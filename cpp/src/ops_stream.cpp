// ops_stream.cpp — streamed scale/maxpool/upsample/add/concat/softmax.
#include "args.hpp"
#include "common.hpp"
#include "gdal_io.hpp"
#include "streaming.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

static int rows_from_budget_simple(int64_t budget_bytes, int C, int W,
                                   int min_extra_rows = 0) {
    if (budget_bytes <= 0) return 1024;
    const int64_t per_row = static_cast<int64_t>(C) * W * 4 * 2;  // slop 2x
    if (per_row <= 0) return 1024;
    int64_t rows = budget_bytes / per_row - min_extra_rows;
    if (rows < 1) rows = 1;
    if (rows > 8192) rows = 8192;
    return static_cast<int>(rows);
}

int run_scale(const Args& a) {
    ReaderDS in(a.ins[0]);
    const int W = in.meta.W, H = in.meta.H, C = in.meta.C;
    WriterDS out(a.out_tif, W, H, C,
                 in.meta.gt, in.meta.has_gt, in.meta.srs, a.co_opts);
    const int Hs = a.strip_rows > 0
        ? std::min(a.strip_rows, H)
        : std::min(H, rows_from_budget_simple(a.mem_bytes, C, W));
    std::vector<float> buf(static_cast<size_t>(C) * Hs * W);
    const float s = a.scale;
    for (int y0 = 0; y0 < H; y0 += Hs) {
        const int y1 = std::min(H, y0 + Hs);
        const int rows = y1 - y0;
        in.read_strip(y0, y1, buf.data());
        const size_t n = static_cast<size_t>(C) * rows * W;
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) buf[i] *= s;
        out.write_strip(y0, y1, buf.data());
    }
    return 0;
}

int run_softmax(const Args& a) {
    ReaderDS in(a.ins[0]);
    const int W = in.meta.W, H = in.meta.H, C = in.meta.C;
    WriterDS out(a.out_tif, W, H, C,
                 in.meta.gt, in.meta.has_gt, in.meta.srs, a.co_opts);
    const int Hs = a.strip_rows > 0
        ? std::min(a.strip_rows, H)
        : std::min(H, rows_from_budget_simple(a.mem_bytes, C, W));
    std::vector<float> buf(static_cast<size_t>(C) * Hs * W);
    for (int y0 = 0; y0 < H; y0 += Hs) {
        const int y1 = std::min(H, y0 + Hs);
        const int rows = y1 - y0;
        in.read_strip(y0, y1, buf.data());
        const size_t plane = static_cast<size_t>(rows) * W;
        #pragma omp parallel for schedule(static)
        for (size_t p = 0; p < plane; ++p) {
            float m = -std::numeric_limits<float>::infinity();
            for (int c = 0; c < C; ++c) {
                float v = buf[static_cast<size_t>(c) * plane + p];
                if (v > m) m = v;
            }
            float s = 0.0f;
            for (int c = 0; c < C; ++c) {
                float e = std::exp(buf[static_cast<size_t>(c) * plane + p] - m);
                buf[static_cast<size_t>(c) * plane + p] = e;
                s += e;
            }
            float inv = 1.0f / s;
            for (int c = 0; c < C; ++c) {
                buf[static_cast<size_t>(c) * plane + p] *= inv;
            }
        }
        out.write_strip(y0, y1, buf.data());
    }
    return 0;
}

int run_add(const Args& a) {
    if (a.ins.size() < 2) { std::cerr << "add: need >=2 inputs\n"; return 2; }
    std::vector<std::unique_ptr<ReaderDS>> rs;
    rs.reserve(a.ins.size());
    for (auto& p : a.ins) rs.push_back(std::make_unique<ReaderDS>(p));
    int H = rs[0]->meta.H, W = rs[0]->meta.W, C = rs[0]->meta.C;
    for (auto& r : rs) {
        H = std::min(H, r->meta.H);
        W = std::min(W, r->meta.W);
        C = std::min(C, r->meta.C);
    }
    // Mismatched widths/channels would require a per-row crop; bail loudly.
    for (auto& r : rs) {
        if (r->meta.W != W || r->meta.C != C) {
            std::cerr << "add: input dims mismatch in W/C (W,C of inputs must match)\n";
            return 2;
        }
    }
    WriterDS out(a.out_tif, W, H, C,
                 rs[0]->meta.gt, rs[0]->meta.has_gt, rs[0]->meta.srs, a.co_opts);
    const int Hs = a.strip_rows > 0
        ? std::min(a.strip_rows, H)
        : std::min(H, rows_from_budget_simple(a.mem_bytes, C * 2, W));
    std::vector<float> acc(static_cast<size_t>(C) * Hs * W);
    std::vector<float> tmp(static_cast<size_t>(C) * Hs * W);
    const Act act = a.activation;
    for (int y0 = 0; y0 < H; y0 += Hs) {
        const int y1 = std::min(H, y0 + Hs);
        const int rows = y1 - y0;
        rs[0]->read_strip(y0, y1, acc.data());
        for (size_t k = 1; k < rs.size(); ++k) {
            rs[k]->read_strip(y0, y1, tmp.data());
            const size_t n = static_cast<size_t>(C) * rows * W;
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < n; ++i) acc[i] += tmp[i];
        }
        if (act != Act::NONE) {
            const size_t n = static_cast<size_t>(C) * rows * W;
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < n; ++i) acc[i] = apply_act(acc[i], act);
        }
        out.write_strip(y0, y1, acc.data());
    }
    return 0;
}

int run_concat(const Args& a) {
    if (a.ins.size() < 2) { std::cerr << "concat: need >=2 inputs\n"; return 2; }
    std::vector<std::unique_ptr<ReaderDS>> rs;
    rs.reserve(a.ins.size());
    for (auto& p : a.ins) rs.push_back(std::make_unique<ReaderDS>(p));
    int H = rs[0]->meta.H, W = rs[0]->meta.W;
    for (auto& r : rs) {
        H = std::min(H, r->meta.H);
        W = std::min(W, r->meta.W);
    }
    int Ctot = 0;
    for (auto& r : rs) Ctot += r->meta.C;
    // Require matching W (per-row cropping in streamed form is not worth it).
    for (auto& r : rs) {
        if (r->meta.W != W) {
            std::cerr << "concat: input widths must match\n"; return 2;
        }
    }
    WriterDS out(a.out_tif, W, H, Ctot,
                 rs[0]->meta.gt, rs[0]->meta.has_gt, rs[0]->meta.srs, a.co_opts);
    const int Hs = a.strip_rows > 0
        ? std::min(a.strip_rows, H)
        : std::min(H, rows_from_budget_simple(a.mem_bytes, Ctot, W));
    std::vector<float> buf(static_cast<size_t>(Ctot) * Hs * W);
    for (int y0 = 0; y0 < H; y0 += Hs) {
        const int y1 = std::min(H, y0 + Hs);
        const int rows = y1 - y0;
        int c_off = 0;
        for (auto& r : rs) {
            std::vector<int> bands(r->meta.C);
            for (int i = 0; i < r->meta.C; ++i) bands[i] = i + 1;
            float* dst0 = &buf[static_cast<size_t>(c_off) * rows * W];
            CPLErr err = r->ds->RasterIO(
                GF_Read, 0, y0, W, rows, dst0,
                W, rows, GDT_Float32, r->meta.C, bands.data(),
                sizeof(float),
                static_cast<GSpacing>(sizeof(float)) * W,
                static_cast<GSpacing>(sizeof(float)) * W * rows,
                nullptr);
            if (err != CE_None) { std::cerr << "concat: read failed\n"; return 2; }
            c_off += r->meta.C;
        }
        out.write_strip(y0, y1, buf.data());
    }
    return 0;
}

int run_upsample(const Args& a) {
    if (a.up_method != "nearest") {
        std::cerr << "upsample method must be 'nearest'\n"; return 2;
    }
    const int F = static_cast<int>(a.scale);
    if (F <= 0) { std::cerr << "upsample --scale must be positive int\n"; return 2; }

    ReaderDS in(a.ins[0]);
    const int W = in.meta.W, H = in.meta.H, C = in.meta.C;
    const int Hout = H * F, Wout = W * F;
    double gt2[6];
    gt_for_upsample(in.meta.gt, F, gt2);

    WriterDS out(a.out_tif, Wout, Hout, C,
                 gt2, in.meta.has_gt, in.meta.srs, a.co_opts);

    // Strip in output rows; each chunk of F output rows uses 1 input row.
    int Hs_out = a.strip_rows > 0
        ? std::min(a.strip_rows, Hout)
        : std::min(Hout, rows_from_budget_simple(a.mem_bytes, C, Wout));
    // Round to multiple of F so strips align with input rows.
    Hs_out = std::max(F, (Hs_out / F) * F);

    const int Hs_in_max = Hs_out / F;
    std::vector<float> in_buf(static_cast<size_t>(C) * Hs_in_max * W);
    std::vector<float> out_buf(static_cast<size_t>(C) * Hs_out * Wout);

    for (int y0_o = 0; y0_o < Hout; y0_o += Hs_out) {
        const int y1_o = std::min(Hout, y0_o + Hs_out);
        const int y0_i = y0_o / F;
        const int y1_i = (y1_o + F - 1) / F;
        const int rows_in = y1_i - y0_i;
        const int rows_out = y1_o - y0_o;
        in.read_strip(y0_i, y1_i, in_buf.data());
        #pragma omp parallel for collapse(2) schedule(static)
        for (int c = 0; c < C; ++c) {
            for (int i = 0; i < rows_out; ++i) {
                const int isrc_global = y0_o + i;
                const int isrc = isrc_global / F - y0_i;
                const float* in_row = &in_buf[(static_cast<size_t>(c) * rows_in + isrc) * W];
                float* out_row = &out_buf[(static_cast<size_t>(c) * rows_out + i) * Wout];
                for (int j = 0; j < Wout; ++j) {
                    out_row[j] = in_row[j / F];
                }
            }
        }
        out.write_strip(y0_o, y1_o, out_buf.data());
    }
    return 0;
}

int run_maxpool(const Args& a) {
    if (a.pool_kernel <= 0 || a.pool_stride <= 0) {
        std::cerr << "maxpool needs --kernel-size and --stride\n"; return 2;
    }
    ReaderDS in(a.ins[0]);
    const int W = in.meta.W, H = in.meta.H, C = in.meta.C;
    const int K = a.pool_kernel, S = a.pool_stride, P = a.pool_padding;
    const int Hout = (H + 2 * P - K) / S + 1;
    const int Wout = (W + 2 * P - K) / S + 1;
    if (Hout <= 0 || Wout <= 0) {
        std::cerr << "invalid maxpool output size\n"; return 2;
    }
    double gt_out[6];
    gt_for_conv(in.meta.gt, K, K, S, P, gt_out);
    WriterDS out(a.out_tif, Wout, Hout, C,
                 gt_out, in.meta.has_gt, in.meta.srs, a.co_opts);

    int Hs_out = a.strip_rows > 0
        ? std::min(a.strip_rows, Hout)
        : std::min(Hout, strip_rows_from_budget(a.mem_bytes, C, C, W, K, S));

    std::vector<float> in_strip;
    std::vector<float> out_buf;
    const float NINF = -std::numeric_limits<float>::infinity();

    for (int y0_o = 0; y0_o < Hout; y0_o += Hs_out) {
        const int y1_o = std::min(Hout, y0_o + Hs_out);
        StripPlan sp = plan_strip(y0_o, y1_o, H, K, S, P);
        in_strip.assign(static_cast<size_t>(C) * sp.H_strip * W, 0.0f);
        if (sp.y1_read > sp.y0_read) {
            const int real_rows = sp.y1_read - sp.y0_read;
            std::vector<int> bands(C);
            for (int i = 0; i < C; ++i) bands[i] = i + 1;
            float* dst0 = &in_strip[static_cast<size_t>(sp.pad_top) * W];
            CPLErr err = in.ds->RasterIO(
                GF_Read, 0, sp.y0_read, W, real_rows, dst0,
                W, real_rows, GDT_Float32, C, bands.data(),
                sizeof(float),
                static_cast<GSpacing>(sizeof(float)) * W,
                static_cast<GSpacing>(sizeof(float)) * W * sp.H_strip,
                nullptr);
            if (err != CE_None) { std::cerr << "maxpool read failed\n"; return 2; }
        }
        out_buf.assign(static_cast<size_t>(C) * sp.Hs_out * Wout, 0.0f);

        // Pool over strip. H-padding handled by the strip's zero rows; W-padding
        // explicit in the inner loop (kx-bound check vs left/right margins).
        const int W_eff = W;
        #pragma omp parallel for collapse(2) schedule(static)
        for (int c = 0; c < C; ++c) {
            for (int i = 0; i < sp.Hs_out; ++i) {
                const float* in_c = &in_strip[static_cast<size_t>(c) * sp.H_strip * W];
                float* out_c = &out_buf[static_cast<size_t>(c) * sp.Hs_out * Wout];
                for (int j = 0; j < Wout; ++j) {
                    float m = NINF;
                    for (int ky = 0; ky < K; ++ky) {
                        const int isrc = i * S + ky;          // strip-local
                        for (int kx = 0; kx < K; ++kx) {
                            const int jsrc = j * S + kx - P;  // image-global
                            if (jsrc < 0 || jsrc >= W_eff) continue;
                            float v = in_c[isrc * W + jsrc];
                            if (v > m) m = v;
                        }
                    }
                    if (m == NINF) m = 0.0f;
                    out_c[i * Wout + j] = m;
                }
            }
        }
        out.write_strip(y0_o, y1_o, out_buf.data());
    }
    return 0;
}
