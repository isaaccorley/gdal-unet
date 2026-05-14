// conv_dnnl.cpp — streamed conv (+ optional BN/bias + activation) via oneDNN.
//
// Strategy:
//  - BN params fold into weights/bias so we ship a single conv+bias+act primitive.
//  - Strip the output in H. Each output strip [y0_o,y1_o) needs input rows
//    [y0_o*S - P, (y1_o-1)*S - P + kH); rows outside [0,H) are zero-filled
//    in the strip buffer. H padding is therefore implicit; only W padding
//    is passed to oneDNN.
//  - One primitive per distinct (Hs_out) used: typically 1 (interior strips)
//    plus 1 for the final, possibly-shorter strip.
//  - Weights are reordered once to oneDNN's preferred layout via format_tag::any.

#include "args.hpp"
#include "common.hpp"
#include "gdal_io.hpp"
#include "streaming.hpp"

#include <oneapi/dnnl/dnnl.hpp>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <unordered_map>
#include <vector>

using namespace dnnl;

namespace {

algorithm post_op_for(Act act, float& alpha, float& beta) {
    alpha = 0.f; beta = 0.f;
    switch (act) {
        case Act::RELU:    return algorithm::eltwise_relu;
        case Act::RELU6:   alpha = 0.f; beta = 6.f; return algorithm::eltwise_clip;
        case Act::SWISH:   alpha = 1.f; return algorithm::eltwise_swish;
        case Act::GELU:    return algorithm::eltwise_gelu_tanh;
        case Act::HSWISH:  alpha = 1.f / 6.f; beta = 0.5f; return algorithm::eltwise_hardswish;
        case Act::SIGMOID: return algorithm::eltwise_logistic;
        case Act::NONE:    return algorithm::undef;
    }
    return algorithm::undef;
}

struct ConvCtx {
    engine eng{engine::kind::cpu, 0};
    stream strm{eng};
    memory weights_mem;       // in pd-preferred layout
    memory bias_mem;          // plain x
    memory::desc weights_md;  // concrete layout chosen on first build
    bool weights_ready = false;
    bool depthwise = false;
    int Cin = 0, Cout = 0, kH = 0, kW = 0, S = 1, P = 0;
    Act act = Act::NONE;

    // Cache one primitive per output-strip height.
    struct Entry {
        convolution_forward::primitive_desc pd;
        convolution_forward prim;
    };
    std::unordered_map<int, Entry> cache;
};

memory::desc make_src_md(int Cin, int H_strip, int W) {
    return memory::desc({1, Cin, H_strip, W}, memory::data_type::f32, memory::format_tag::nchw);
}

memory::desc make_dst_md(int Cout, int Hout, int Wout) {
    return memory::desc({1, Cout, Hout, Wout}, memory::data_type::f32, memory::format_tag::nchw);
}

memory::desc make_wei_md(bool depthwise, int Cin, int Cout, int kH, int kW,
                         memory::format_tag tag) {
    if (depthwise) {
        return memory::desc({Cin, 1, 1, kH, kW}, memory::data_type::f32, tag);
    }
    return memory::desc({Cout, Cin, kH, kW}, memory::data_type::f32, tag);
}

}  // namespace

static ConvCtx::Entry& get_or_build(ConvCtx& ctx, int Hs_out, int W_src) {
    auto it = ctx.cache.find(Hs_out);
    if (it != ctx.cache.end()) return it->second;

    const int H_strip = (Hs_out - 1) * ctx.S + ctx.kH;
    const int Wout = (W_src + 2 * ctx.P - ctx.kW) / ctx.S + 1;

    auto src_md = make_src_md(ctx.Cin, H_strip, W_src);
    auto wei_md_req = ctx.weights_ready
        ? ctx.weights_md
        : make_wei_md(ctx.depthwise, ctx.Cin, ctx.Cout,
                      ctx.kH, ctx.kW, memory::format_tag::any);
    auto bia_md = memory::desc({ctx.Cout}, memory::data_type::f32, memory::format_tag::x);
    auto dst_md = make_dst_md(ctx.Cout, Hs_out, Wout);

    primitive_attr attr;
    if (ctx.act != Act::NONE) {
        post_ops po;
        float alpha, beta;
        algorithm alg = post_op_for(ctx.act, alpha, beta);
        po.append_eltwise(alg, alpha, beta);
        attr.set_post_ops(po);
    }

    memory::dims strides{ctx.S, ctx.S};
    memory::dims pad_l{0, ctx.P};   // H is fully materialized in strip buffer
    memory::dims pad_r{0, ctx.P};

    auto pd = convolution_forward::primitive_desc(
        ctx.eng, prop_kind::forward_inference, algorithm::convolution_auto,
        src_md, wei_md_req, bia_md, dst_md,
        strides, pad_l, pad_r, attr);

    // Reorder weights to pd-preferred layout on first build only.
    if (!ctx.weights_ready) {
        memory new_wei(pd.weights_desc(), ctx.eng);
        reorder(ctx.weights_mem, new_wei).execute(
            ctx.strm, ctx.weights_mem, new_wei);
        ctx.strm.wait();
        ctx.weights_mem = std::move(new_wei);
        ctx.weights_md = pd.weights_desc();
        ctx.weights_ready = true;
    }

    ConvCtx::Entry e{pd, convolution_forward(pd)};
    auto [ins, _] = ctx.cache.emplace(Hs_out, std::move(e));
    return ins->second;
}

int run_conv(const Args& a) {
    if (a.depthwise && a.Cout != a.Cin) {
        std::cerr << "depthwise requires Cout==Cin\n"; return 2;
    }

    ReaderDS in(a.ins[0]);
    if (in.meta.C != a.Cin) {
        std::cerr << "input has " << in.meta.C << " bands, kernel expects "
                  << a.Cin << "\n"; return 2;
    }
    const int W = in.meta.W;
    const int H = in.meta.H;

    // --- load kernel + (optional) BN/bias, fold BN into weights/bias ---
    const size_t kn = a.depthwise
        ? static_cast<size_t>(a.Cin) * a.kH * a.kW
        : static_cast<size_t>(a.Cout) * a.Cin * a.kH * a.kW;
    std::vector<float> kernel = read_raw_f16(a.kernel_path, kn);
    std::vector<float> bias_buf(a.Cout, 0.0f);

    const bool have_bn   = !a.bn_a_path.empty();
    const bool have_bias = !a.bias_path.empty();
    if (have_bn) {
        auto bn_a = read_raw_f16(a.bn_a_path, a.Cout);
        auto bn_b = read_raw_f16(a.bn_b_path, a.Cout);
        const size_t per_oc = a.depthwise
            ? static_cast<size_t>(a.kH) * a.kW
            : static_cast<size_t>(a.Cin) * a.kH * a.kW;
        for (int oc = 0; oc < a.Cout; ++oc) {
            float* w = &kernel[static_cast<size_t>(oc) * per_oc];
            for (size_t i = 0; i < per_oc; ++i) w[i] *= bn_a[oc];
            bias_buf[oc] = bn_b[oc];
        }
    } else if (have_bias) {
        auto b = read_raw_f16(a.bias_path, a.Cout);
        for (int oc = 0; oc < a.Cout; ++oc) bias_buf[oc] = b[oc];
    }

    // --- ctx setup ---
    ConvCtx ctx;
    ctx.depthwise = a.depthwise;
    ctx.Cin = a.Cin; ctx.Cout = a.Cout;
    ctx.kH = a.kH;   ctx.kW = a.kW;
    ctx.S  = a.stride; ctx.P = a.padding;
    ctx.act = a.activation;

    // Stage weights in a plain layout, then get_or_build will reorder.
    auto wei_md_plain = make_wei_md(a.depthwise, a.Cin, a.Cout, a.kH, a.kW,
                                    a.depthwise ? memory::format_tag::goihw
                                                : memory::format_tag::oihw);
    ctx.weights_mem = memory(wei_md_plain, ctx.eng);
    std::memcpy(ctx.weights_mem.get_data_handle(), kernel.data(),
                kernel.size() * sizeof(float));

    auto bia_md = memory::desc({a.Cout}, memory::data_type::f32, memory::format_tag::x);
    ctx.bias_mem = memory(bia_md, ctx.eng);
    std::memcpy(ctx.bias_mem.get_data_handle(), bias_buf.data(),
                bias_buf.size() * sizeof(float));

    // --- output sizing ---
    const int Hout = (H + 2 * a.padding - a.kH) / a.stride + 1;
    const int Wout = (W + 2 * a.padding - a.kW) / a.stride + 1;
    if (Hout <= 0 || Wout <= 0) {
        std::cerr << "invalid conv output size " << Hout << "x" << Wout << "\n";
        return 2;
    }

    double gt_out[6];
    gt_for_conv(in.meta.gt, a.kH, a.kW, a.stride, a.padding, gt_out);

    WriterDS out(a.out_tif, Wout, Hout, a.Cout,
                 gt_out, in.meta.has_gt, in.meta.srs, a.co_opts);

    // --- pick output strip height from budget ---
    int Hs_out = a.strip_rows > 0
        ? a.strip_rows
        : strip_rows_from_budget(a.mem_bytes, a.Cin, a.Cout, W, a.kH, a.stride);
    Hs_out = std::min(Hs_out, Hout);

    // --- iterate strips ---
    std::vector<float> in_strip;   // C * H_strip * W
    std::vector<float> out_strip;  // Cout * Hs_out_actual * Wout

    for (int y0_o = 0; y0_o < Hout; y0_o += Hs_out) {
        const int y1_o = std::min(Hout, y0_o + Hs_out);
        StripPlan sp = plan_strip(y0_o, y1_o, H, a.kH, a.stride, a.padding);

        const size_t in_n = static_cast<size_t>(a.Cin) * sp.H_strip * W;
        in_strip.assign(in_n, 0.0f);

        if (sp.y1_read > sp.y0_read) {
            // Read directly into the strip buffer, skipping pad_top zero rows
            // per channel via GDAL band/line spacing — no temp copy.
            const int real_rows = sp.y1_read - sp.y0_read;
            std::vector<int> bands(a.Cin);
            for (int i = 0; i < a.Cin; ++i) bands[i] = i + 1;
            float* dst0 = &in_strip[static_cast<size_t>(sp.pad_top) * W];
            CPLErr err = in.ds->RasterIO(
                GF_Read, 0, sp.y0_read, W, real_rows, dst0,
                W, real_rows, GDT_Float32, a.Cin, bands.data(),
                sizeof(float),
                static_cast<GSpacing>(sizeof(float)) * W,
                static_cast<GSpacing>(sizeof(float)) * W * sp.H_strip,
                nullptr);
            if (err != CE_None) { std::cerr << "RasterIO read failed\n"; return 2; }
        }

        const int Hs_actual = sp.Hs_out;
        out_strip.assign(static_cast<size_t>(a.Cout) * Hs_actual * Wout, 0.0f);

        auto& e = get_or_build(ctx, Hs_actual, W);

        memory src_mem(e.pd.src_desc(), ctx.eng, in_strip.data());
        memory dst_mem(e.pd.dst_desc(), ctx.eng, out_strip.data());

        e.prim.execute(ctx.strm, {
            {DNNL_ARG_SRC, src_mem},
            {DNNL_ARG_WEIGHTS, ctx.weights_mem},
            {DNNL_ARG_BIAS, ctx.bias_mem},
            {DNNL_ARG_DST, dst_mem},
        });
        ctx.strm.wait();

        out.write_strip(y0_o, y1_o, out_strip.data());
    }

    return 0;
}
