// common.hpp — shared types, f16 helpers, activations.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

enum class Act { NONE, RELU, RELU6, SWISH, GELU, HSWISH, SIGMOID };

inline Act parse_act(const std::string& s) {
    if (s == "none")                          return Act::NONE;
    if (s == "relu")                          return Act::RELU;
    if (s == "relu6")                         return Act::RELU6;
    if (s == "swish" || s == "silu")          return Act::SWISH;
    if (s == "gelu")                          return Act::GELU;
    if (s == "hswish" || s == "hard_swish")   return Act::HSWISH;
    if (s == "sigmoid")                       return Act::SIGMOID;
    std::cerr << "unknown activation: " << s << "\n";
    std::exit(2);
}

inline float apply_act(float v, Act act) {
    switch (act) {
        case Act::NONE:    return v;
        case Act::RELU:    return v < 0.0f ? 0.0f : v;
        case Act::RELU6:   return v < 0.0f ? 0.0f : (v > 6.0f ? 6.0f : v);
        case Act::SWISH:   return v / (1.0f + std::exp(-v));
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

// Decode raw float16 blob (little-endian IEEE half) to float32.
inline std::vector<float> read_raw_f16(const std::string& path, size_t expected) {
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
