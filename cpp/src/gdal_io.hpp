// gdal_io.hpp — GDAL helpers for streamed strip I/O.
#pragma once

#include <gdal.h>
#include <gdal_priv.h>
#include <cpl_string.h>
#include <ogr_spatialref.h>

#include <cstring>
#include <iostream>
#include <string>
#include <vector>

struct RasterMeta {
    int W = 0, H = 0, C = 0;
    double gt[6] = {0, 1, 0, 0, 0, 1};
    bool has_gt = false;
    const OGRSpatialReference* srs = nullptr;
};

// RAII open for reading.
struct ReaderDS {
    GDALDataset* ds = nullptr;
    RasterMeta meta;

    explicit ReaderDS(const std::string& path) {
        ds = static_cast<GDALDataset*>(GDALOpenEx(
            path.c_str(), GDAL_OF_RASTER | GDAL_OF_READONLY, nullptr, nullptr, nullptr));
        if (!ds) { std::cerr << "cannot open input " << path << "\n"; std::exit(2); }
        meta.W = ds->GetRasterXSize();
        meta.H = ds->GetRasterYSize();
        meta.C = ds->GetRasterCount();
        meta.has_gt = (ds->GetGeoTransform(meta.gt) == CE_None);
        meta.srs = ds->GetSpatialRef();
    }
    ~ReaderDS() { if (ds) GDALClose(ds); }
    ReaderDS(const ReaderDS&) = delete;
    ReaderDS& operator=(const ReaderDS&) = delete;

    // Read rows [y0,y1) into NCHW float32 buffer of size C*(y1-y0)*W.
    void read_strip(int y0, int y1, float* dst) const {
        const int rows = y1 - y0;
        std::vector<int> bands(meta.C);
        for (int i = 0; i < meta.C; ++i) bands[i] = i + 1;
        CPLErr err = ds->RasterIO(
            GF_Read, 0, y0, meta.W, rows, dst,
            meta.W, rows, GDT_Float32, meta.C, bands.data(),
            sizeof(float),
            static_cast<GSpacing>(sizeof(float)) * meta.W,
            static_cast<GSpacing>(sizeof(float)) * meta.W * rows,
            nullptr);
        if (err != CE_None) { std::cerr << "RasterIO read failed\n"; std::exit(2); }
    }
};

// RAII create + RAII strip write.
struct WriterDS {
    GDALDataset* ds = nullptr;
    int W = 0, H = 0, C = 0;

    WriterDS(const std::string& path, int W_, int H_, int C_,
             const double gt[6], bool has_gt,
             const OGRSpatialReference* srs,
             const std::vector<std::string>& co_opts) : W(W_), H(H_), C(C_) {
        GDALDriver* drv = GetGDALDriverManager()->GetDriverByName("GTiff");
        if (!drv) { std::cerr << "GTiff driver missing\n"; std::exit(2); }
        char** opts = nullptr;
        for (auto& s : co_opts) opts = CSLAddString(opts, s.c_str());
        ds = drv->Create(path.c_str(), W, H, C, GDT_Float16, opts);
        CSLDestroy(opts);
        if (!ds) { std::cerr << "create output failed\n"; std::exit(2); }
        if (has_gt) ds->SetGeoTransform(const_cast<double*>(gt));
        if (srs) ds->SetSpatialRef(srs);
    }
    ~WriterDS() { if (ds) GDALClose(ds); }
    WriterDS(const WriterDS&) = delete;
    WriterDS& operator=(const WriterDS&) = delete;

    // Write NCHW strip rows [y0,y1).
    void write_strip(int y0, int y1, const float* src) {
        const int rows = y1 - y0;
        std::vector<int> bands(C);
        for (int i = 0; i < C; ++i) bands[i] = i + 1;
        CPLErr err = ds->RasterIO(
            GF_Write, 0, y0, W, rows, const_cast<float*>(src),
            W, rows, GDT_Float32, C, bands.data(),
            sizeof(float),
            static_cast<GSpacing>(sizeof(float)) * W,
            static_cast<GSpacing>(sizeof(float)) * W * rows,
            nullptr);
        if (err != CE_None) { std::cerr << "RasterIO write failed\n"; std::exit(2); }
    }
};

// Adjust geotransform for a conv-/pool-like op with kernel (kH,kW),
// stride S, padding P.
//
// Output pixel (0,0) covers S input-pixel-widths centered on the receptive
// field. Its corner sits at (K - S)/2 - P input-pixel units from input
// pixel (0,0)'s corner. For the "same" convolution case (S=1, K odd,
// P=(K-1)/2) this collapses to zero — output transform == input transform.
inline void gt_for_conv(const double in_gt[6],
                        int kH, int kW, int S, int P,
                        double out_gt[6]) {
    const double shift_x = (kW - S) * 0.5 - P;   // along col direction
    const double shift_y = (kH - S) * 0.5 - P;   // along row direction
    out_gt[0] = in_gt[0] + shift_x * in_gt[1] + shift_y * in_gt[2];
    out_gt[3] = in_gt[3] + shift_x * in_gt[4] + shift_y * in_gt[5];
    out_gt[1] = in_gt[1] * S;
    out_gt[2] = in_gt[2] * S;
    out_gt[4] = in_gt[4] * S;
    out_gt[5] = in_gt[5] * S;
}

// Adjust geotransform for nearest upsample by factor F.
inline void gt_for_upsample(const double in_gt[6], int F, double out_gt[6]) {
    std::memcpy(out_gt, in_gt, 6 * sizeof(double));
    out_gt[1] /= F; out_gt[2] /= F;
    out_gt[4] /= F; out_gt[5] /= F;
}
