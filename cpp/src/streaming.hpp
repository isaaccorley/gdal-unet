// streaming.hpp — strip-plan computation for conv/pool-like ops.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstddef>

// Plan for a single output strip [y0_o, y1_o) of a conv-like op with
// kernel kH, stride S, padding P over an input image of height H.
//
// We feed the kernel a contiguous in-memory strip of input rows of
// height H_strip = (Hs_out-1)*S + kH, where rows that lie outside the
// real image are zero-filled. This lets us use a single primitive with
// padding_t = padding_b = 0 in H for every interior strip; only the
// last strip may have a smaller Hs_out and need its own primitive.
struct StripPlan {
    int y0_out;       // first output row
    int y1_out;       // one past last output row
    int Hs_out;       // y1_out - y0_out
    int H_strip;      // rows in the input-side strip buffer
    int y0_read;      // first input row read from file (>=0)
    int y1_read;      // one past last input row read (<=H)
    int pad_top;      // rows of zeros prepended into strip buffer
    int pad_bot;      // rows of zeros appended into strip buffer
};

inline StripPlan plan_strip(int y0_out, int y1_out,
                            int H_in, int kH, int S, int P) {
    StripPlan p;
    p.y0_out = y0_out;
    p.y1_out = y1_out;
    p.Hs_out = y1_out - y0_out;
    p.H_strip = (p.Hs_out - 1) * S + kH;

    const int in_top = y0_out * S - P;             // inclusive
    const int in_bot = in_top + p.H_strip;         // exclusive
    p.pad_top = std::max(0, -in_top);
    p.pad_bot = std::max(0, in_bot - H_in);
    p.y0_read = std::max(0, in_top);
    p.y1_read = std::min(H_in, in_bot);
    return p;
}

// Choose default output strip height from a memory budget (bytes).
// Estimates per-strip working set as roughly:
//   input  buffer: Cin * H_strip * W * 4
//   output buffer: Cout * Hs_out * W * 4
// Solves Hs_out * (Cin*S + Cout) * W * 4 + Cin*(kH - S)*W*4 <= budget
// (slop factor 2x for oneDNN scratch + GDAL block buffers).
inline int strip_rows_from_budget(int64_t budget_bytes,
                                  int Cin, int Cout,
                                  int W, int kH, int S) {
    if (budget_bytes <= 0) return 1024;
    const int64_t slop = 2;
    const int64_t per_row = static_cast<int64_t>(W) * 4 *
                            (static_cast<int64_t>(Cin) * S + Cout) * slop;
    const int64_t fixed = static_cast<int64_t>(Cin) *
                          std::max(0, kH - S) *
                          static_cast<int64_t>(W) * 4 * slop;
    if (per_row <= 0) return 1024;
    int64_t rows = (budget_bytes - fixed) / per_row;
    if (rows < 1) rows = 1;
    if (rows > 8192) rows = 8192;  // cap for primitive cache + diminishing returns
    return static_cast<int>(rows);
}
