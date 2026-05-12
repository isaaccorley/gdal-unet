// gdal-conv2d: single-process layer ops using GDAL I/O.
//
// Modes:
//   conv     - 2D convolution (+ optional BN/bias + activation)
//   scale    - elementwise multiply by scalar
//   maxpool  - max pool 2D
//   upsample - nearest-neighbor upsample by integer factor
//   add      - elementwise add of N rasters (+ optional activation)
//   concat   - band-wise concat of N rasters
//   softmax  - per-pixel stable softmax across bands
//
// Float16 on disk, float32 internally. OpenMP over output channels (conv).

#include <gdal.h>
#include <gdal_priv.h>
#include <cpl_string.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <omp.h>

// Activation kinds
enum class Act { NONE, RELU, RELU6, SWISH, GELU, HSWISH, SIGMOID };

static inline float apply_act(float v, Act act) {
    switch (act) {
        case Act::NONE:   return v;
        case Act::RELU:   return v < 0.0f ? 0.0f : v;
        case Act::RELU6:  return v < 0.0f ? 0.0f : (v > 6.0f ? 6.0f : v);
        case Act::SWISH:  return v / (1.0f + std::exp(-v));
        case Act::SIGMOID: return 1.0f / (1.0f + std::exp(-v));
        case Act::HSWISH: {
            float t = v + 3.0f;
            if (t < 0.0f) t = 0.0f; else if (t > 6.0f) t = 6.0f;
            return v * t * (1.0f / 6.0f);
        }
        case Act::GELU: {
            const float k = 0.7978845608028654f;
            float t = k * (v + 0.044715f * v * v * v);
            return 0.5f * v * (1.0f + std::tanh(t));
        }
    }
    return v;
}

static Act parse_act(const std::string& s) {
    if (s == "none")    return Act::NONE;
    if (s == "relu")    return Act::RELU;
    if (s == "relu6")   return Act::RELU6;
    if (s == "swish" || s == "silu") return Act::SWISH;
    if (s == "gelu")    return Act::GELU;
    if (s == "hswish" || s == "hard_swish") return Act::HSWISH;
    if (s == "sigmoid") return Act::SIGMOID;
    std::cerr << "unknown activation: " << s << "\n";
    std::exit(2);
}

// ---------- Float16 helpers ----------
static std::vector<float> read_raw_f16(const std::string& path, size_t expected) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "cannot open " << path << "\n"; std::exit(2); }
    f.seekg(0, std::ios::end);
    size_t nbytes = f.tellg();
    f.seekg(0, std::ios::beg);
    if (nbytes != expected * 2) {
        std::cerr << "size mismatch on " << path << ": got " << nbytes
                  << " bytes, expected " << (expected * 2) << "\n";
        std::exit(2);
    }
    std::vector<uint16_t> raw(expected);
    f.read(reinterpret_cast<char*>(raw.data()), nbytes);
    std::vector<float> out(expected);
    for (size_t i = 0; i < expected; ++i) {
        uint16_t h = raw[i];
        uint32_t sign = (h & 0x8000u) << 16;
        uint32_t exp  = (h & 0x7C00u) >> 10;
        uint32_t mant = (h & 0x03FFu);
        uint32_t f32;
        if (exp == 0) {
            if (mant == 0) {
                f32 = sign;
            } else {
                while (!(mant & 0x0400u)) { mant <<= 1; exp = (exp == 0 ? 0 : exp - 1); --exp; }
                mant &= 0x03FFu;
                f32 = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
            }
        } else if (exp == 0x1F) {
            f32 = sign | 0x7F800000u | (mant << 13);
        } else {
            f32 = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
        }
        float fv;
        std::memcpy(&fv, &f32, 4);
        out[i] = fv;
    }
    return out;
}

// ---------- GDAL I/O helpers ----------
struct Raster {
    int W = 0, H = 0, C = 0;
    std::vector<float> data;  // (C, H, W) row-major
    double gt[6] = {0,1,0,0,0,1};
    bool has_gt = false;
    const OGRSpatialReference* srs = nullptr;
};

static Raster read_raster(const std::string& path) {
    Raster r;
    GDALDataset* ds = static_cast<GDALDataset*>(GDALOpenEx(
        path.c_str(), GDAL_OF_RASTER | GDAL_OF_READONLY, nullptr, nullptr, nullptr));
    if (!ds) { std::cerr << "cannot open input " << path << "\n"; std::exit(2); }
    r.W = ds->GetRasterXSize();
    r.H = ds->GetRasterYSize();
    r.C = ds->GetRasterCount();
    r.has_gt = (ds->GetGeoTransform(r.gt) == CE_None);
    r.srs = ds->GetSpatialRef();
    r.data.assign(static_cast<size_t>(r.C) * r.H * r.W, 0.0f);
    std::vector<int> bands(r.C);
    for (int i=0; i<r.C; ++i) bands[i] = i+1;
    CPLErr err = ds->RasterIO(
        GF_Read, 0, 0, r.W, r.H, r.data.data(),
        r.W, r.H, GDT_Float32, r.C, bands.data(),
        sizeof(float),
        static_cast<GSpacing>(sizeof(float)) * r.W,
        static_cast<GSpacing>(sizeof(float)) * r.W * r.H,
        nullptr);
    if (err != CE_None) { std::cerr << "RasterIO read failed\n"; std::exit(2); }
    GDALClose(ds);
    return r;
}

static void write_raster(const std::string& path, const float* data,
                         int W, int H, int C,
                         const double gt[6], bool has_gt,
                         const OGRSpatialReference* srs,
                         const std::vector<std::string>& co_opts) {
    GDALDriver* drv = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (!drv) { std::cerr << "GTiff driver missing\n"; std::exit(2); }
    char** opts = nullptr;
    for (auto& s : co_opts) opts = CSLAddString(opts, s.c_str());
    GDALDataset* out_ds = drv->Create(path.c_str(), W, H, C, GDT_Float16, opts);
    CSLDestroy(opts);
    if (!out_ds) { std::cerr << "create output failed\n"; std::exit(2); }
    if (has_gt) out_ds->SetGeoTransform(const_cast<double*>(gt));
    if (srs) out_ds->SetSpatialRef(srs);
    std::vector<int> bands(C);
    for (int i=0; i<C; ++i) bands[i] = i+1;
    CPLErr err = out_ds->RasterIO(
        GF_Write, 0, 0, W, H, const_cast<float*>(data),
        W, H, GDT_Float32, C, bands.data(),
        sizeof(float),
        static_cast<GSpacing>(sizeof(float)) * W,
        static_cast<GSpacing>(sizeof(float)) * W * H,
        nullptr);
    if (err != CE_None) { std::cerr << "RasterIO write failed\n"; std::exit(2); }
    GDALClose(out_ds);
}

// ---------- Args ----------
struct Args {
    std::string mode = "conv";
    // conv args
    std::vector<std::string> ins;  // for add/concat: multiple inputs; otherwise size 1
    std::string kernel_path;
    int Cout=0, Cin=0, kH=0, kW=0;
    std::string bn_a_path, bn_b_path, bias_path;
    bool relu = false;
    Act activation = Act::NONE;
    bool depthwise = false;
    int stride = 1;
    int padding = 0;
    // scale
    float scale = 1.0f;
    // maxpool
    int pool_kernel = 0;
    int pool_stride = 0;
    int pool_padding = 0;
    // upsample
    int up_scale = 1;
    std::string up_method = "nearest";
    std::string out_tif;
    int threads = 0;
    std::vector<std::string> co_opts;
};

static void usage() {
    std::cerr <<
      "Usage: gdal-conv2d --mode MODE ... --out <out.tif> [--threads N] [--co KEY=VAL ...]\n"
      "\n"
      "Modes:\n"
      "  conv     --in <in.tif> --kernel <k.bin> --kernel-shape Cout,Cin,kH,kW\n"
      "           [--bn-a a.bin --bn-b b.bin] [--bias b.bin] [--depthwise]\n"
      "           [--stride S] [--padding P] [--activation TYPE | --relu]\n"
      "           activation: none|relu|relu6|swish|silu|gelu|hswish|sigmoid\n"
      "  scale    --in <in.tif> --scale <float>\n"
      "  maxpool  --in <in.tif> --kernel-size N --stride S --padding P\n"
      "  upsample --in <in.tif> --scale F --method nearest\n"
      "  add      --in <in1.tif> --in <in2.tif> [--in <in3.tif> ...] [--activation TYPE]\n"
      "  concat   --in <in1.tif> --in <in2.tif> [--in <in3.tif> ...]\n"
      "  softmax  --in <in.tif>\n";
}

static bool parse_shape(const std::string& s, Args& a) {
    int vs[4] = {0,0,0,0};
    int n = std::sscanf(s.c_str(), "%d,%d,%d,%d", &vs[0],&vs[1],&vs[2],&vs[3]);
    if (n != 4) return false;
    a.Cout=vs[0]; a.Cin=vs[1]; a.kH=vs[2]; a.kW=vs[3];
    return true;
}

static int parse_args(int argc, char** argv, Args& a) {
    for (int i=1; i<argc; ++i) {
        std::string s = argv[i];
        auto need = [&](const char*) {
            if (i+1 >= argc) { usage(); std::exit(2); }
            return std::string(argv[++i]);
        };
        if      (s == "--mode")         a.mode = need("mode");
        else if (s == "--in")           a.ins.push_back(need("in"));
        else if (s == "--kernel")       a.kernel_path  = need("kernel");
        else if (s == "--kernel-shape") { if (!parse_shape(need("shape"), a)) { usage(); return 2; } }
        else if (s == "--bn-a")         a.bn_a_path    = need("bn-a");
        else if (s == "--bn-b")         a.bn_b_path    = need("bn-b");
        else if (s == "--bias")         a.bias_path    = need("bias");
        else if (s == "--relu")         { a.relu = true; a.activation = Act::RELU; }
        else if (s == "--activation")   a.activation = parse_act(need("activation"));
        else if (s == "--depthwise")    a.depthwise = true;
        else if (s == "--stride")       a.stride = std::stoi(need("stride"));
        else if (s == "--padding")      a.padding = std::stoi(need("padding"));
        else if (s == "--kernel-size")  a.pool_kernel = std::stoi(need("kernel-size"));
        else if (s == "--scale")        a.scale = std::stof(need("scale"));
        else if (s == "--method")       a.up_method = need("method");
        else if (s == "--out")          a.out_tif = need("out");
        else if (s == "--threads")      a.threads = std::stoi(need("threads"));
        else if (s == "--co")           a.co_opts.push_back(need("co"));
        else if (s == "-h" || s == "--help") { usage(); std::exit(0); }
        else { std::cerr << "unknown arg: " << s << "\n"; usage(); return 2; }
    }
    // maxpool: --stride/--padding shared with conv. Copy to pool slots.
    if (a.mode == "maxpool") {
        a.pool_stride = a.stride;
        a.pool_padding = a.padding;
    }
    if (a.out_tif.empty() || a.ins.empty()) { usage(); return 2; }
    return 0;
}

// ---------- modes ----------
static int run_conv(const Args& a) {
    GDALDataset* in_ds = static_cast<GDALDataset*>(GDALOpenEx(
        a.ins[0].c_str(), GDAL_OF_RASTER | GDAL_OF_READONLY, nullptr, nullptr, nullptr));
    if (!in_ds) { std::cerr << "cannot open input " << a.ins[0] << "\n"; return 2; }
    const int W = in_ds->GetRasterXSize();
    const int H = in_ds->GetRasterYSize();
    const int Cin_in = in_ds->GetRasterCount();
    if (Cin_in != a.Cin) {
        std::cerr << "input has " << Cin_in << " bands, kernel expects "
                  << a.Cin << "\n"; return 2;
    }
    double gt[6] = {0,1,0,0,0,1};
    bool has_gt = (in_ds->GetGeoTransform(gt) == CE_None);
    const OGRSpatialReference* srs = in_ds->GetSpatialRef();

    std::vector<float> input(static_cast<size_t>(a.Cin) * H * W);
    {
        std::vector<int> bands(a.Cin);
        for (int i=0; i<a.Cin; ++i) bands[i] = i+1;
        CPLErr err = in_ds->RasterIO(
            GF_Read, 0, 0, W, H, input.data(),
            W, H, GDT_Float32, a.Cin, bands.data(),
            sizeof(float),
            static_cast<GSpacing>(sizeof(float)) * W,
            static_cast<GSpacing>(sizeof(float)) * W * H,
            nullptr);
        if (err != CE_None) { std::cerr << "RasterIO read failed\n"; return 2; }
    }

    const int P = a.padding;
    const int Hp = H + 2*P;
    const int Wp = W + 2*P;
    std::vector<float> padded;
    const float* src_ptr = nullptr;
    int Hsrc, Wsrc;
    if (P == 0) {
        src_ptr = input.data();
        Hsrc = H; Wsrc = W;
    } else {
        padded.assign(static_cast<size_t>(a.Cin) * Hp * Wp, 0.0f);
        #pragma omp parallel for collapse(2) schedule(static)
        for (int c = 0; c < a.Cin; ++c) {
            for (int i = 0; i < H; ++i) {
                const float* sline = &input[(static_cast<size_t>(c) * H + i) * W];
                float* dline = &padded[(static_cast<size_t>(c) * Hp + (i + P)) * Wp + P];
                std::memcpy(dline, sline, sizeof(float) * W);
            }
        }
        src_ptr = padded.data();
        Hsrc = Hp; Wsrc = Wp;
        std::vector<float>().swap(input);
    }

    const size_t kn = a.depthwise
        ? static_cast<size_t>(a.Cin) * a.kH * a.kW
        : static_cast<size_t>(a.Cout) * a.Cin * a.kH * a.kW;
    std::vector<float> kernel = read_raw_f16(a.kernel_path, kn);
    std::vector<float> bn_a, bn_b, bias;
    bool have_bn   = !a.bn_a_path.empty();
    bool have_bias = !a.bias_path.empty();
    if (have_bn) {
        bn_a = read_raw_f16(a.bn_a_path, a.Cout);
        bn_b = read_raw_f16(a.bn_b_path, a.Cout);
    }
    if (have_bias) bias = read_raw_f16(a.bias_path, a.Cout);

    const int Hf = Hsrc - a.kH + 1;
    const int Wf = Wsrc - a.kW + 1;
    if (Hf <= 0 || Wf <= 0) {
        std::cerr << "invalid conv output size " << Hf << "x" << Wf << "\n";
        return 2;
    }
    auto stride_count = [&](int n) { return (n + a.stride - 1) / a.stride; };
    const int Hout = stride_count(Hf);
    const int Wout = stride_count(Wf);

    std::vector<float> out(static_cast<size_t>(a.Cout) * Hout * Wout);

    const int S = a.stride;
    const int kH = a.kH, kW = a.kW;
    const size_t in_chan_stride = static_cast<size_t>(Hsrc) * Wsrc;
    const size_t k_chan_stride  = static_cast<size_t>(a.kH) * a.kW;
    const size_t k_outc_stride  = static_cast<size_t>(a.Cin) * k_chan_stride;
    const size_t out_chan_stride = static_cast<size_t>(Hout) * Wout;

    const Act act = a.activation;
    const bool depthwise = a.depthwise;

    #pragma omp parallel for schedule(dynamic, 1)
    for (int oc = 0; oc < a.Cout; ++oc) {
        float* out_oc = &out[oc * out_chan_stride];
        const float* k_oc = depthwise
            ? &kernel[static_cast<size_t>(oc) * k_chan_stride]
            : &kernel[oc * k_outc_stride];
        const int ic_lo = depthwise ? oc : 0;
        const int ic_hi = depthwise ? oc + 1 : a.Cin;
        for (int i = 0; i < Hout; ++i) {
            const int isrc = i * S;
            for (int j = 0; j < Wout; ++j) {
                const int jsrc = j * S;
                float acc = 0.0f;
                for (int ic = ic_lo; ic < ic_hi; ++ic) {
                    const float* in_ic = src_ptr + ic * in_chan_stride;
                    const float* k_ic  = depthwise
                        ? k_oc
                        : k_oc + (ic - ic_lo) * k_chan_stride;
                    for (int ky = 0; ky < kH; ++ky) {
                        const float* in_row = in_ic + (isrc + ky) * Wsrc + jsrc;
                        const float* k_row  = k_ic + ky * kW;
                        for (int kx = 0; kx < kW; ++kx) {
                            acc += k_row[kx] * in_row[kx];
                        }
                    }
                }
                out_oc[i * Wout + j] = acc;
            }
        }
        const float ba = have_bn ? bn_a[oc] : 1.0f;
        const float bb = have_bn ? bn_b[oc] : 0.0f;
        const float bi = have_bias ? bias[oc] : 0.0f;
        const size_t N = static_cast<size_t>(Hout) * Wout;
        if (have_bn) {
            for (size_t p = 0; p < N; ++p) {
                float v = ba * out_oc[p] + bb;
                out_oc[p] = apply_act(v, act);
            }
        } else if (have_bias) {
            for (size_t p = 0; p < N; ++p) {
                float v = out_oc[p] + bi;
                out_oc[p] = apply_act(v, act);
            }
        } else if (act != Act::NONE) {
            for (size_t p = 0; p < N; ++p) {
                out_oc[p] = apply_act(out_oc[p], act);
            }
        }
    }

    // Adjusted geotransform: pixel size *= stride; origin shifted -P*pixel for padding.
    double gt2[6] = {gt[0], gt[1]*S, gt[2]*S, gt[3], gt[4]*S, gt[5]*S};
    gt2[0] = gt[0] - P * gt[1] - P * gt[2];
    gt2[3] = gt[3] - P * gt[4] - P * gt[5];
    gt2[1] = gt[1]*S; gt2[2] = gt[2]*S;
    gt2[4] = gt[4]*S; gt2[5] = gt[5]*S;

    write_raster(a.out_tif, out.data(), Wout, Hout, a.Cout,
                 gt2, has_gt, srs, a.co_opts);
    GDALClose(in_ds);
    return 0;
}

static int run_scale(const Args& a) {
    Raster r = read_raster(a.ins[0]);
    const size_t N = r.data.size();
    const float s = a.scale;
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) r.data[i] *= s;
    write_raster(a.out_tif, r.data.data(), r.W, r.H, r.C,
                 r.gt, r.has_gt, r.srs, a.co_opts);
    return 0;
}

static int run_maxpool(const Args& a) {
    if (a.pool_kernel <= 0 || a.pool_stride <= 0) {
        std::cerr << "maxpool needs --kernel-size and --stride\n"; return 2;
    }
    Raster r = read_raster(a.ins[0]);
    const int K = a.pool_kernel;
    const int S = a.pool_stride;
    const int P = a.pool_padding;
    // PyTorch nn.MaxPool2d output size formula (floor):
    //   out = floor((H + 2P - K) / S) + 1
    const int Hout = (r.H + 2*P - K) / S + 1;
    const int Wout = (r.W + 2*P - K) / S + 1;
    if (Hout <= 0 || Wout <= 0) {
        std::cerr << "invalid maxpool output size\n"; return 2;
    }
    std::vector<float> out(static_cast<size_t>(r.C) * Hout * Wout);
    const float NINF = -std::numeric_limits<float>::infinity();
    #pragma omp parallel for schedule(static)
    for (int c = 0; c < r.C; ++c) {
        const float* in_c = &r.data[static_cast<size_t>(c) * r.H * r.W];
        float* out_c = &out[static_cast<size_t>(c) * Hout * Wout];
        for (int i = 0; i < Hout; ++i) {
            for (int j = 0; j < Wout; ++j) {
                float m = NINF;
                for (int ky = 0; ky < K; ++ky) {
                    int isrc = i * S + ky - P;
                    if (isrc < 0 || isrc >= r.H) continue;
                    for (int kx = 0; kx < K; ++kx) {
                        int jsrc = j * S + kx - P;
                        if (jsrc < 0 || jsrc >= r.W) continue;
                        float v = in_c[isrc * r.W + jsrc];
                        if (v > m) m = v;
                    }
                }
                if (m == NINF) m = 0.0f;
                out_c[i * Wout + j] = m;
            }
        }
    }
    double gt2[6];
    std::memcpy(gt2, r.gt, sizeof(gt2));
    // shift origin for padding (origin moves UP/LEFT by P), then pixel scaled by S
    gt2[0] = r.gt[0] - P * r.gt[1] - P * r.gt[2];
    gt2[3] = r.gt[3] - P * r.gt[4] - P * r.gt[5];
    gt2[1] = r.gt[1]*S; gt2[2] = r.gt[2]*S;
    gt2[4] = r.gt[4]*S; gt2[5] = r.gt[5]*S;
    write_raster(a.out_tif, out.data(), Wout, Hout, r.C,
                 gt2, r.has_gt, r.srs, a.co_opts);
    return 0;
}

static int run_upsample(const Args& a) {
    if (a.up_method != "nearest") {
        std::cerr << "upsample method must be 'nearest'\n"; return 2;
    }
    int F = static_cast<int>(a.scale);
    if (F <= 0) { std::cerr << "upsample --scale must be positive int\n"; return 2; }
    Raster r = read_raster(a.ins[0]);
    int Hout = r.H * F, Wout = r.W * F;
    std::vector<float> out(static_cast<size_t>(r.C) * Hout * Wout);
    #pragma omp parallel for schedule(static)
    for (int c = 0; c < r.C; ++c) {
        const float* in_c = &r.data[static_cast<size_t>(c) * r.H * r.W];
        float* out_c = &out[static_cast<size_t>(c) * Hout * Wout];
        for (int i = 0; i < Hout; ++i) {
            int isrc = i / F;
            for (int j = 0; j < Wout; ++j) {
                int jsrc = j / F;
                out_c[i * Wout + j] = in_c[isrc * r.W + jsrc];
            }
        }
    }
    double gt2[6];
    std::memcpy(gt2, r.gt, sizeof(gt2));
    gt2[1] = r.gt[1] / F; gt2[2] = r.gt[2] / F;
    gt2[4] = r.gt[4] / F; gt2[5] = r.gt[5] / F;
    write_raster(a.out_tif, out.data(), Wout, Hout, r.C,
                 gt2, r.has_gt, r.srs, a.co_opts);
    return 0;
}

static int run_add(const Args& a) {
    if (a.ins.size() < 2) { std::cerr << "add: need >=2 inputs\n"; return 2; }
    Raster acc = read_raster(a.ins[0]);
    for (size_t k = 1; k < a.ins.size(); ++k) {
        Raster r = read_raster(a.ins[k]);
        // tolerate small mismatches: crop to min
        int H = std::min(acc.H, r.H);
        int W = std::min(acc.W, r.W);
        int C = std::min(acc.C, r.C);
        if (H != acc.H || W != acc.W || C != acc.C) {
            // rewrite acc cropped
            std::vector<float> cropped(static_cast<size_t>(C) * H * W);
            for (int c = 0; c < C; ++c)
                for (int i = 0; i < H; ++i)
                    std::memcpy(&cropped[(c*H + i)*W],
                                &acc.data[(static_cast<size_t>(c)*acc.H + i)*acc.W],
                                sizeof(float) * W);
            acc.data = std::move(cropped);
            acc.H = H; acc.W = W; acc.C = C;
        }
        #pragma omp parallel for schedule(static)
        for (int c = 0; c < C; ++c) {
            float* a_c = &acc.data[static_cast<size_t>(c) * H * W];
            const float* b_c = &r.data[static_cast<size_t>(c) * r.H * r.W];
            for (int i = 0; i < H; ++i)
                for (int j = 0; j < W; ++j)
                    a_c[i*W + j] += b_c[i*r.W + j];
        }
    }
    Act act = a.activation;
    if (act != Act::NONE) {
        size_t N = acc.data.size();
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < N; ++i) acc.data[i] = apply_act(acc.data[i], act);
    }
    write_raster(a.out_tif, acc.data.data(), acc.W, acc.H, acc.C,
                 acc.gt, acc.has_gt, acc.srs, a.co_opts);
    return 0;
}

static int run_concat(const Args& a) {
    if (a.ins.size() < 2) { std::cerr << "concat: need >=2 inputs\n"; return 2; }
    std::vector<Raster> rs;
    rs.reserve(a.ins.size());
    for (auto& p : a.ins) rs.push_back(read_raster(p));
    int H = rs[0].H, W = rs[0].W;
    for (auto& r : rs) { H = std::min(H, r.H); W = std::min(W, r.W); }
    int Ctot = 0;
    for (auto& r : rs) Ctot += r.C;
    std::vector<float> out(static_cast<size_t>(Ctot) * H * W);
    int co = 0;
    for (auto& r : rs) {
        for (int c = 0; c < r.C; ++c) {
            for (int i = 0; i < H; ++i) {
                std::memcpy(&out[(static_cast<size_t>(co + c)*H + i)*W],
                            &r.data[(static_cast<size_t>(c)*r.H + i)*r.W],
                            sizeof(float) * W);
            }
        }
        co += r.C;
    }
    write_raster(a.out_tif, out.data(), W, H, Ctot,
                 rs[0].gt, rs[0].has_gt, rs[0].srs, a.co_opts);
    return 0;
}

static int run_softmax(const Args& a) {
    Raster r = read_raster(a.ins[0]);
    const int C = r.C, H = r.H, W = r.W;
    const size_t plane = static_cast<size_t>(H) * W;
    #pragma omp parallel for schedule(static)
    for (size_t p = 0; p < plane; ++p) {
        float m = -std::numeric_limits<float>::infinity();
        for (int c = 0; c < C; ++c) {
            float v = r.data[static_cast<size_t>(c)*plane + p];
            if (v > m) m = v;
        }
        float s = 0.0f;
        for (int c = 0; c < C; ++c) {
            float e = std::exp(r.data[static_cast<size_t>(c)*plane + p] - m);
            r.data[static_cast<size_t>(c)*plane + p] = e;
            s += e;
        }
        float inv = 1.0f / s;
        for (int c = 0; c < C; ++c) {
            r.data[static_cast<size_t>(c)*plane + p] *= inv;
        }
    }
    write_raster(a.out_tif, r.data.data(), W, H, C,
                 r.gt, r.has_gt, r.srs, a.co_opts);
    return 0;
}

int main(int argc, char** argv) {
    Args a;
    if (int r = parse_args(argc, argv, a)) return r;
    if (a.threads > 0) omp_set_num_threads(a.threads);
    GDALAllRegister();

    if (a.mode == "conv") {
        if (a.kernel_path.empty() || a.Cout==0 || a.Cin==0 || a.kH==0 || a.kW==0) {
            std::cerr << "conv mode requires --kernel and --kernel-shape\n";
            return 2;
        }
        if (!a.bias_path.empty() && !a.bn_a_path.empty()) {
            std::cerr << "error: --bias and --bn-a are mutually exclusive\n"; return 2;
        }
        if (a.stride < 1) { std::cerr << "stride must be >= 1\n"; return 2; }
        if (a.depthwise && a.Cout != a.Cin) {
            std::cerr << "depthwise requires Cout==Cin\n"; return 2;
        }
        return run_conv(a);
    }
    if (a.mode == "scale")    return run_scale(a);
    if (a.mode == "maxpool")  return run_maxpool(a);
    if (a.mode == "upsample") return run_upsample(a);
    if (a.mode == "add")      return run_add(a);
    if (a.mode == "concat")   return run_concat(a);
    if (a.mode == "softmax")  return run_softmax(a);
    std::cerr << "unknown mode: " << a.mode << "\n";
    usage();
    return 2;
}
