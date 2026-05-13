// gdal-conv2d: single-process layer ops using GDAL I/O + oneDNN.
//
// Modes:
//   conv     - 2D convolution (+ optional BN/bias + activation), oneDNN
//   scale    - elementwise multiply by scalar
//   maxpool  - max pool 2D
//   upsample - nearest-neighbor upsample by integer factor
//   add      - elementwise add of N rasters (+ optional activation)
//   concat   - band-wise concat of N rasters
//   softmax  - per-pixel stable softmax across bands
//
// Float16 on disk, float32 internally. All modes stream in row strips
// sized by --mem-mb (default 2048) so very large rasters never load
// fully in RAM.

#include "args.hpp"
#include "common.hpp"

#include <gdal.h>
#include <gdal_priv.h>
#include <omp.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

static void usage() {
    std::cerr <<
      "Usage: gdal-conv2d --mode MODE ... --out <out.tif>\n"
      "        [--threads N] [--mem-mb MB | --strip-rows N] [--co KEY=VAL ...]\n"
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
      "  softmax  --in <in.tif>\n"
      "\n"
      "Streaming:\n"
      "  --mem-mb MB       per-op memory budget for strip buffers (default 2048)\n"
      "  --strip-rows N    explicit output strip height (overrides --mem-mb)\n";
}

static bool parse_shape(const std::string& s, Args& a) {
    int vs[4] = {0, 0, 0, 0};
    int n = std::sscanf(s.c_str(), "%d,%d,%d,%d", &vs[0], &vs[1], &vs[2], &vs[3]);
    if (n != 4) return false;
    a.Cout = vs[0]; a.Cin = vs[1]; a.kH = vs[2]; a.kW = vs[3];
    return true;
}

static int parse_args(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto need = [&](const char*) {
            if (i + 1 >= argc) { usage(); std::exit(2); }
            return std::string(argv[++i]);
        };
        if      (s == "--mode")         a.mode = need("mode");
        else if (s == "--in")           a.ins.push_back(need("in"));
        else if (s == "--kernel")       a.kernel_path = need("kernel");
        else if (s == "--kernel-shape") { if (!parse_shape(need("shape"), a)) { usage(); return 2; } }
        else if (s == "--bn-a")         a.bn_a_path = need("bn-a");
        else if (s == "--bn-b")         a.bn_b_path = need("bn-b");
        else if (s == "--bias")         a.bias_path = need("bias");
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
        else if (s == "--mem-mb")       a.mem_bytes = static_cast<int64_t>(std::stoll(need("mem-mb"))) * 1024 * 1024;
        else if (s == "--strip-rows")   a.strip_rows = std::stoi(need("strip-rows"));
        else if (s == "-h" || s == "--help") { usage(); std::exit(0); }
        else { std::cerr << "unknown arg: " << s << "\n"; usage(); return 2; }
    }
    if (a.mode == "maxpool") {
        a.pool_stride = a.stride;
        a.pool_padding = a.padding;
    }
    if (a.out_tif.empty() || a.ins.empty()) { usage(); return 2; }
    return 0;
}

int main(int argc, char** argv) {
    Args a;
    if (int r = parse_args(argc, argv, a)) return r;
    if (a.threads > 0) {
        omp_set_num_threads(a.threads);
        // oneDNN picks up OMP_NUM_THREADS / omp_set_num_threads automatically
        // when built with the OpenMP runtime.
    }
    GDALAllRegister();

    if (a.mode == "conv") {
        if (a.kernel_path.empty() || a.Cout == 0 || a.Cin == 0 || a.kH == 0 || a.kW == 0) {
            std::cerr << "conv mode requires --kernel and --kernel-shape\n";
            return 2;
        }
        if (!a.bias_path.empty() && !a.bn_a_path.empty()) {
            std::cerr << "error: --bias and --bn-a are mutually exclusive\n"; return 2;
        }
        if (a.stride < 1) { std::cerr << "stride must be >= 1\n"; return 2; }
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
