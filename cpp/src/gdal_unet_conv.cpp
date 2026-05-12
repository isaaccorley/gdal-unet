// gdal-unet-conv: single-process conv+BN+ReLU+(stride/padding) layer using GDAL I/O.
//
// Replaces the chunked-diagonal `gdal raster pipeline | calc` cascade that
// fires ~70 subprocesses per conv layer. Naive triple-loop conv, OpenMP over
// output channels, scalar inner loops. Float16 on disk, float32 internally.

#include <gdal.h>
#include <gdal_priv.h>
#include <cpl_string.h>

#include <algorithm>
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

// ---------- Float16 helpers (IEEE-754 binary16) ----------
// We use GDT_Float32 for RasterIO and pass GDT_Float32 buffers but the on-disk
// type may be Float16 (GDT_Float16, available in GDAL 3.11+).

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
    // half -> float (IEEE-754 binary16). Use a simple software conversion so
    // we don't depend on _Float16/__fp16 availability.
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
                // subnormal -> normalize
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

struct Args {
    std::string in_tif;
    std::string kernel_path;
    int Cout=0, Cin=0, kH=0, kW=0;
    std::string bn_a_path, bn_b_path, bias_path;
    bool relu = false;
    int stride = 1;
    int padding = 0;
    std::string out_tif;
    int threads = 0;
    std::vector<std::string> co_opts;
};

static void usage() {
    std::cerr << "Usage: gdal-unet-conv --in <in.tif> --kernel <k.bin> "
                 "--kernel-shape Cout,Cin,kH,kW [--bn-a a.bin --bn-b b.bin] "
                 "[--bias bias.bin] [--relu] [--stride 1|2] [--padding P] "
                 "--out <out.tif> [--threads N] [--co KEY=VAL ...]\n";
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
        auto need = [&](const char* /*name*/) {
            if (i+1 >= argc) { usage(); std::exit(2); }
            return std::string(argv[++i]);
        };
        if      (s == "--in")           a.in_tif       = need("in");
        else if (s == "--kernel")       a.kernel_path  = need("kernel");
        else if (s == "--kernel-shape") { if (!parse_shape(need("shape"), a)) { usage(); return 2; } }
        else if (s == "--bn-a")         a.bn_a_path    = need("bn-a");
        else if (s == "--bn-b")         a.bn_b_path    = need("bn-b");
        else if (s == "--bias")         a.bias_path    = need("bias");
        else if (s == "--relu")         a.relu = true;
        else if (s == "--stride")       a.stride = std::stoi(need("stride"));
        else if (s == "--padding")      a.padding = std::stoi(need("padding"));
        else if (s == "--out")          a.out_tif = need("out");
        else if (s == "--threads")      a.threads = std::stoi(need("threads"));
        else if (s == "--co")           a.co_opts.push_back(need("co"));
        else if (s == "-h" || s == "--help") { usage(); std::exit(0); }
        else { std::cerr << "unknown arg: " << s << "\n"; usage(); return 2; }
    }
    if (a.in_tif.empty() || a.kernel_path.empty() || a.out_tif.empty() ||
        a.Cout==0 || a.Cin==0 || a.kH==0 || a.kW==0) {
        usage(); return 2;
    }
    if (!a.bias_path.empty() && !a.bn_a_path.empty()) {
        std::cerr << "error: --bias and --bn-a are mutually exclusive\n"; return 2;
    }
    if (a.stride != 1 && a.stride != 2) { std::cerr << "stride must be 1 or 2\n"; return 2; }
    return 0;
}

int main(int argc, char** argv) {
    Args a;
    if (int r = parse_args(argc, argv, a)) return r;

    if (a.threads > 0) omp_set_num_threads(a.threads);
    GDALAllRegister();

    // ---- Open input ----
    GDALDataset* in_ds = static_cast<GDALDataset*>(GDALOpenEx(
        a.in_tif.c_str(), GDAL_OF_RASTER | GDAL_OF_READONLY, nullptr, nullptr, nullptr));
    if (!in_ds) { std::cerr << "cannot open input " << a.in_tif << "\n"; return 2; }
    const int W = in_ds->GetRasterXSize();
    const int H = in_ds->GetRasterYSize();
    const int Cin_in = in_ds->GetRasterCount();
    if (Cin_in != a.Cin) {
        std::cerr << "input has " << Cin_in << " bands, kernel expects "
                  << a.Cin << "\n"; return 2;
    }

    // Geotransform / SRS
    double gt[6] = {0,1,0,0,0,1};
    bool has_gt = (in_ds->GetGeoTransform(gt) == CE_None);
    const OGRSpatialReference* srs = in_ds->GetSpatialRef();

    // ---- Read full input into float32 (Cin, H, W) ----
    std::vector<float> input(static_cast<size_t>(a.Cin) * H * W);
    {
        // RasterIO band-interleaved read.
        std::vector<int> bands(a.Cin);
        for (int i=0; i<a.Cin; ++i) bands[i] = i+1;
        CPLErr err = in_ds->RasterIO(
            GF_Read, 0, 0, W, H, input.data(),
            W, H, GDT_Float32, a.Cin, bands.data(),
            /*nPixelSpace=*/ sizeof(float),
            /*nLineSpace=*/  static_cast<GSpacing>(sizeof(float)) * W,
            /*nBandSpace=*/  static_cast<GSpacing>(sizeof(float)) * W * H,
            nullptr);
        if (err != CE_None) { std::cerr << "RasterIO read failed\n"; return 2; }
    }

    // ---- Pad input ----
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
        // free input now
        std::vector<float>().swap(input);
    }

    // ---- Load kernel + optional bn / bias ----
    const size_t kn = static_cast<size_t>(a.Cout) * a.Cin * a.kH * a.kW;
    std::vector<float> kernel = read_raw_f16(a.kernel_path, kn);
    std::vector<float> bn_a, bn_b, bias;
    bool have_bn   = !a.bn_a_path.empty();
    bool have_bias = !a.bias_path.empty();
    if (have_bn) {
        bn_a = read_raw_f16(a.bn_a_path, a.Cout);
        bn_b = read_raw_f16(a.bn_b_path, a.Cout);
    }
    if (have_bias) bias = read_raw_f16(a.bias_path, a.Cout);

    // ---- Output shape after conv (with stride) ----
    // Valid conv on padded input -> (Hsrc - kH + 1, Wsrc - kW + 1) at stride 1.
    const int Hf = Hsrc - a.kH + 1;   // full-resolution output rows
    const int Wf = Wsrc - a.kW + 1;
    if (Hf <= 0 || Wf <= 0) {
        std::cerr << "invalid conv output size " << Hf << "x" << Wf << "\n";
        return 2;
    }
    // stride: sample positions 0, stride, 2*stride, ...
    auto stride_count = [&](int n) { return (n + a.stride - 1) / a.stride; };
    const int Hout = stride_count(Hf);
    const int Wout = stride_count(Wf);

    // ---- Allocate output ----
    std::vector<float> out(static_cast<size_t>(a.Cout) * Hout * Wout);

    // ---- Conv ----
    // For each output channel oc (parallel), each output pixel (i,j):
    //   sum over (ic, ky, kx) of kernel[oc, ic, ky, kx] * src[ic, i*S+ky, j*S+kx]
    const int S = a.stride;
    const int kH = a.kH, kW = a.kW;
    const size_t in_chan_stride = static_cast<size_t>(Hsrc) * Wsrc;
    const size_t k_chan_stride  = static_cast<size_t>(a.kH) * a.kW;
    const size_t k_outc_stride  = static_cast<size_t>(a.Cin) * k_chan_stride;
    const size_t out_chan_stride = static_cast<size_t>(Hout) * Wout;

    #pragma omp parallel for schedule(dynamic, 1)
    for (int oc = 0; oc < a.Cout; ++oc) {
        float* out_oc = &out[oc * out_chan_stride];
        const float* k_oc = &kernel[oc * k_outc_stride];
        for (int i = 0; i < Hout; ++i) {
            const int isrc = i * S;
            for (int j = 0; j < Wout; ++j) {
                const int jsrc = j * S;
                float acc = 0.0f;
                for (int ic = 0; ic < a.Cin; ++ic) {
                    const float* in_ic = src_ptr + ic * in_chan_stride;
                    const float* k_ic  = k_oc + ic * k_chan_stride;
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

        // bias / BN / ReLU per output channel
        const float ba = have_bn ? bn_a[oc] : 1.0f;
        const float bb = have_bn ? bn_b[oc] : 0.0f;
        const float bi = have_bias ? bias[oc] : 0.0f;
        const bool do_relu = a.relu;
        const size_t N = static_cast<size_t>(Hout) * Wout;
        if (have_bn) {
            for (size_t p = 0; p < N; ++p) {
                float v = ba * out_oc[p] + bb;
                if (do_relu && v < 0.0f) v = 0.0f;
                out_oc[p] = v;
            }
        } else if (have_bias) {
            for (size_t p = 0; p < N; ++p) {
                float v = out_oc[p] + bi;
                if (do_relu && v < 0.0f) v = 0.0f;
                out_oc[p] = v;
            }
        } else if (do_relu) {
            for (size_t p = 0; p < N; ++p) {
                if (out_oc[p] < 0.0f) out_oc[p] = 0.0f;
            }
        }
    }

    // ---- Write output as Float16 GeoTIFF ----
    GDALDriver* drv = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (!drv) { std::cerr << "GTiff driver missing\n"; return 2; }

    char** opts = nullptr;
    for (auto& s : a.co_opts) opts = CSLAddString(opts, s.c_str());

    GDALDataset* out_ds = drv->Create(
        a.out_tif.c_str(), Wout, Hout, a.Cout, GDT_Float16, opts);
    CSLDestroy(opts);
    if (!out_ds) { std::cerr << "create output failed\n"; return 2; }

    if (has_gt) {
        // Adjust geotransform for stride: pixel size *= stride.
        // (Origin stays the same because we sample starting at (0,0).)
        double gt2[6] = {gt[0], gt[1]*S, gt[2]*S, gt[3], gt[4]*S, gt[5]*S};
        // Account for padding: padding shifts the "0,0" sample of the conv
        // window UP/LEFT by P pixels in the original grid. We undo that so
        // the output sits on the same map grid as the input (matches the
        // Python pad+conv+crop behavior).
        // origin' = origin - P * pixel_size_in_x/y
        gt2[0] = gt[0] - P * gt[1] - P * gt[2];
        gt2[3] = gt[3] - P * gt[4] - P * gt[5];
        // also pixel sizes scaled by stride:
        gt2[1] = gt[1]*S; gt2[2] = gt[2]*S;
        gt2[4] = gt[4]*S; gt2[5] = gt[5]*S;
        out_ds->SetGeoTransform(gt2);
    }
    if (srs) out_ds->SetSpatialRef(srs);

    {
        std::vector<int> bands(a.Cout);
        for (int i=0; i<a.Cout; ++i) bands[i] = i+1;
        CPLErr err = out_ds->RasterIO(
            GF_Write, 0, 0, Wout, Hout, out.data(),
            Wout, Hout, GDT_Float32, a.Cout, bands.data(),
            sizeof(float),
            static_cast<GSpacing>(sizeof(float)) * Wout,
            static_cast<GSpacing>(sizeof(float)) * Wout * Hout,
            nullptr);
        if (err != CE_None) { std::cerr << "RasterIO write failed\n"; return 2; }
    }
    GDALClose(out_ds);
    GDALClose(in_ds);
    return 0;
}
