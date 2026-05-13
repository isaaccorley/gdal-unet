// args.hpp — CLI argument struct shared by main + op implementations.
#pragma once

#include "common.hpp"

#include <cstdint>
#include <string>
#include <vector>

struct Args {
    std::string mode = "conv";
    // inputs: conv/scale/maxpool/upsample/softmax use ins[0]; add/concat use all.
    std::vector<std::string> ins;

    // conv
    std::string kernel_path;
    int Cout = 0, Cin = 0, kH = 0, kW = 0;
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

    // streaming
    int64_t mem_bytes = 2LL * 1024 * 1024 * 1024;  // --mem-mb default 2048
    int strip_rows = 0;                            // --strip-rows override (0=auto)
};

// Op entry points.
int run_conv(const Args& a);
int run_scale(const Args& a);
int run_maxpool(const Args& a);
int run_upsample(const Args& a);
int run_add(const Args& a);
int run_concat(const Args& a);
int run_softmax(const Args& a);
