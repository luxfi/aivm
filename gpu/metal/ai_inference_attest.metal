// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// ai_inference_attest.metal — Mode 2 / Mode 3 attestation hooks.
//
// For Mode 2 (Confidential), the kernel runs the same int8 GEMM as Mode 1
// but writes its output into a sealed device-private buffer. The host buffer
// is overwritten with the salt-bound commitment only — the raw output never
// leaves the device.
//
// For Mode 3 (Verifiable), the kernel emits the raw output AND the
// commitment in parallel. Sampled replay happens host-side (one thread,
// CPU reference) — this kernel only provides the per-call hash that the
// host folds into the attestation_root.
//
// keccak256 implementation matches aivm_kernels_common.h.metal byte-for-byte.

#include <metal_stdlib>
using namespace metal;

constant ulong RC[24] = {
    0x0000000000000001UL, 0x0000000000008082UL,
    0x800000000000808AUL, 0x8000000080008000UL,
    0x000000000000808BUL, 0x0000000080000001UL,
    0x8000000080008081UL, 0x8000000000008009UL,
    0x000000000000008AUL, 0x0000000000000088UL,
    0x0000000080008009UL, 0x000000008000000AUL,
    0x000000008000808BUL, 0x800000000000008BUL,
    0x8000000000008089UL, 0x8000000000008003UL,
    0x8000000000008002UL, 0x8000000000000080UL,
    0x000000000000800AUL, 0x800000008000000AUL,
    0x8000000080008081UL, 0x8000000000008080UL,
    0x0000000080000001UL, 0x8000000080008008UL,
};

constant uint KROT[25] = {
     0,  1, 62, 28, 27,
    36, 44,  6, 55, 20,
     3, 10, 43, 25, 39,
    41, 45, 15, 21,  8,
    18,  2, 61, 56, 14,
};

inline ulong rotl64_dev(ulong x, uint n) {
    return (x << (n & 63u)) | (x >> ((64u - n) & 63u));
}

void keccak_f1600_dev(thread ulong* s) {
    for (uint round = 0; round < 24u; ++round) {
        ulong c[5];
        for (uint x = 0; x < 5u; ++x)
            c[x] = s[x] ^ s[x+5] ^ s[x+10] ^ s[x+15] ^ s[x+20];
        ulong d[5];
        for (uint x = 0; x < 5u; ++x)
            d[x] = c[(x + 4u) % 5u] ^ rotl64_dev(c[(x + 1u) % 5u], 1u);
        for (uint y = 0; y < 25u; y += 5u)
            for (uint x = 0; x < 5u; ++x) s[y + x] ^= d[x];
        ulong b[25];
        for (uint y = 0; y < 5u; ++y)
            for (uint x = 0; x < 5u; ++x) {
                uint i = x + 5u * y;
                uint j = y + 5u * ((2u * x + 3u * y) % 5u);
                b[j] = rotl64_dev(s[i], KROT[i]);
            }
        for (uint y = 0; y < 25u; y += 5u) {
            ulong t0=b[y+0], t1=b[y+1], t2=b[y+2], t3=b[y+3], t4=b[y+4];
            s[y+0] = t0 ^ ((~t1) & t2);
            s[y+1] = t1 ^ ((~t2) & t3);
            s[y+2] = t2 ^ ((~t3) & t4);
            s[y+3] = t3 ^ ((~t4) & t0);
            s[y+4] = t4 ^ ((~t0) & t1);
        }
        s[0] ^= RC[round];
    }
}

// Hash arbitrary-length data on the device — used by attestation only.
inline void keccak256_dev(device const uchar* data, uint len, thread uchar* out) {
    thread ulong s[25] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    const uint rate = 136u;
    uint off = 0u;
    while ((len - off) >= rate) {
        for (uint i = 0u; i < rate; ++i) {
            uint lane = i / 8u, sh = (i % 8u) * 8u;
            s[lane] ^= ((ulong)data[off + i]) << sh;
        }
        keccak_f1600_dev(s);
        off += rate;
    }
    thread uchar block[136];
    for (uint i = 0u; i < rate; ++i) block[i] = 0u;
    uint rem = len - off;
    for (uint i = 0u; i < rem; ++i) block[i] = data[off + i];
    block[rem]      ^= 0x01u;
    block[rate - 1u] ^= 0x80u;
    for (uint i = 0u; i < rate; ++i) {
        uint lane = i / 8u, sh = (i % 8u) * 8u;
        s[lane] ^= ((ulong)block[i]) << sh;
    }
    keccak_f1600_dev(s);
    for (uint i = 0u; i < 32u; ++i) {
        uint lane = i / 8u, sh = (i % 8u) * 8u;
        out[i] = (uchar)((s[lane] >> sh) & 0xFFu);
    }
}

// Compute commitment = keccak(salt[0..32] || data[0..len]) on-device.
// Buffers: salt 32 bytes, data len bytes, scratch (salt||data) into device pool.
kernel void ai_attest_commit(
    device const uchar* salt    [[buffer(0)]],
    device const uchar* data    [[buffer(1)]],
    constant uint& len          [[buffer(2)]],
    device uchar* commitment    [[buffer(3)]],
    device uchar* scratch       [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0u) return;
    for (uint i = 0u; i < 32u; ++i) scratch[i] = salt[i];
    for (uint i = 0u; i < len; ++i) scratch[32u + i] = data[i];
    thread uchar out[32];
    keccak256_dev(scratch, 32u + len, out);
    for (uint i = 0u; i < 32u; ++i) commitment[i] = out[i];
}
