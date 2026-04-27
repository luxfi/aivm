// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// ai_inference_int8.metal — Metal kernel for the AI precompile (Mode 1/3).
//
// Determinism contract: byte-equal CPU↔Metal for the tiny classifier
// (32 -> 16 -> 1). No tensor cores, no FP16, no parallel reductions.
// One thread per output element of layer 1 OR layer 2; layers run as two
// dispatches so that layer 1's hidden activations are fully resident
// before layer 2 reads them.
//
// Buffer layout:
//   buffer(0)  W1[32*16] int8 row-major
//   buffer(1)  B1[16] int32
//   buffer(2)  W2[16] int8
//   buffer(3)  B2[1] int32
//   buffer(4)  X[N*32] int8 (input batch)
//   buffer(5)  H[N*16] int8 (hidden buffer)
//   buffer(6)  Y[N*1] int8 (output)
//   buffer(7)  Params { uint n; int shift1; int shift2; }
//
// Saturation: clamp [-128, 127]. Right-shift is sign-aware on int32. Same
// reduction order (k=0..K-1) as the CPU reference.

#include <metal_stdlib>
using namespace metal;

struct AIPrecompileParams {
    uint n;
    int  shift1;
    int  shift2;
};

inline int sat_i8(int v) {
    if (v >  127) return  127;
    if (v < -128) return -128;
    return v;
}

kernel void ai_int8_layer1(
    device const char*  W1     [[buffer(0)]],
    device const int*   B1     [[buffer(1)]],
    device const char*  X      [[buffer(4)]],
    device char*        H      [[buffer(5)]],
    constant AIPrecompileParams& P [[buffer(7)]],
    uint2 tid [[thread_position_in_grid]])
{
    const uint sample = tid.y;
    const uint i      = tid.x;
    if (sample >= P.n) return;
    if (i >= 16u) return;

    int acc = B1[i];
    for (uint k = 0u; k < 32u; ++k) {
        int w = (int)W1[i * 32u + k];                  // int8 sign-extended
        int x = (int)X[sample * 32u + k];              // int8 sign-extended
        acc += w * x;
    }
    acc = acc >> P.shift1;                              // sign-aware shift
    H[sample * 16u + i] = (char)sat_i8(acc);
}

kernel void ai_int8_layer2(
    device const char*  W2     [[buffer(2)]],
    device const int*   B2     [[buffer(3)]],
    device const char*  H      [[buffer(5)]],
    device char*        Y      [[buffer(6)]],
    constant AIPrecompileParams& P [[buffer(7)]],
    uint2 tid [[thread_position_in_grid]])
{
    const uint sample = tid.y;
    const uint i      = tid.x;
    if (sample >= P.n) return;
    if (i >= 1u) return;

    int acc = B2[i];
    for (uint k = 0u; k < 16u; ++k) {
        int w = (int)W2[i * 16u + k];
        int h = (int)H[sample * 16u + k];
        acc += w * h;
    }
    acc = acc >> P.shift2;
    Y[sample * 1u + i] = (char)sat_i8(acc);
}
