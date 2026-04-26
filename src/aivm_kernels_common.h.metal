// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// aivm_kernels_common.h.metal — shared device code for the four AIVM kernels.
//
// Layout structs MUST match aivm_gpu_layout.hpp byte-for-byte. keccak256 is
// the same Keccak-f[1600] / 0x01 / 0x80 padding used by the CPU reference
// (aivm_cpu_reference.cpp). Determinism across CPU / Metal / CUDA / WGSL
// hinges on this exact byte-for-byte recipe.

#pragma once

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Layout structs — must match aivm_gpu_layout.hpp byte-for-byte.
// =============================================================================

struct alignas(16) Attestation {
    uchar  tee_quote_digest[32];   // 0
    uchar  measurement[32];        // 32
    uchar  attesting_key[48];      // 64
    ulong  expiry_ns;              // 112
    uint   kind;                   // 120
    uint   evidence_offset;        // 124
    uint   evidence_len;           // 128
    uint   status;                 // 132
    uint   occupied;               // 136
    uint   _pad0;                  // 140 -> 144
};

struct alignas(16) ModelRegistryEntry {
    uchar  model_root[32];         // 0
    uchar  weight_hash[32];        // 32
    uchar  license_root[32];       // 64
    uchar  owner_addr[20];         // 96
    uint   _pad0;                  // 116
    ulong  version;                // 120
    ulong  parameter_count;        // 128
    uint   modality;               // 136
    uint   occupied;               // 140
    ulong  _pad1;                  // 144
    ulong  _pad2;                  // 152 -> 160
};

struct alignas(16) AuditAnchor {
    uchar  commit_root[32];                  // 0
    uchar  parent_root[32];                  // 32
    uchar  validator_set_root_at_commit[32]; // 64
    ulong  height;                           // 96
    ulong  timestamp_ns;                     // 104
    uint   occupied;                         // 112
    uint   _pad0;                            // 116
    ulong  _pad1;                            // 120 -> 128
};

struct alignas(16) AIVMEpochState {
    ulong  current_epoch;                  // 0
    ulong  next_epoch_height;              // 8
    ulong  total_active_attestations;      // 16
    uint   active_model_count;             // 24
    uint   expired_attestation_count;      // 28
    uchar  attestation_root[32];           // 32
    uchar  model_registry_root[32];        // 64
    uchar  audit_root[32];                 // 96
    uchar  aivm_state_root[32];            // 128 -> 160
};

struct alignas(16) AIVMRoundDescriptor {
    ulong  chain_id;
    ulong  round;
    ulong  timestamp_ns;
    ulong  epoch;
    uint   mode;
    uint   attestation_op_count;
    uint   model_op_count;
    uint   anchor_op_count;
    uint   closing_flag;
    uint   _pad0;
    ulong  _pad1;
    uchar  parent_aivm_root[32];
};

struct alignas(16) AttestationOp {
    uchar  tee_quote_digest[32];   // 0
    uchar  measurement[32];        // 32
    uchar  attesting_key[48];      // 64
    ulong  expiry_ns;              // 112
    uint   kind;                   // 120
    uint   evidence_offset;        // 124
    uint   evidence_len;           // 128
    uint   epoch;                  // 132
    uint   _pad0;                  // 136
    uint   _pad1;                  // 140 -> 144
};

struct alignas(16) ModelOp {
    uchar  model_root[32];        // 0
    uchar  weight_hash[32];       // 32
    uchar  license_root[32];      // 64
    uchar  owner_addr[20];        // 96
    uint   _pad0;                 // 116
    ulong  parameter_count;       // 120
    uint   modality;              // 128
    uint   kind;                  // 132
    uint   epoch;                 // 136
    uint   _pad1;                 // 140
    ulong  _pad2;                 // 144
    ulong  _pad3;                 // 152 -> 160
};

struct alignas(16) AnchorOp {
    uchar  commit_root[32];                  // 0
    uchar  parent_root[32];                  // 32
    uchar  validator_set_root_at_commit[32]; // 64
    ulong  height;                           // 96
    ulong  timestamp_ns;                     // 104
    uint   epoch;                            // 112
    uint   _pad0;                            // 116
    ulong  _pad1;                            // 120 -> 128
};

struct alignas(16) AIVMTransitionResult {
    uint   status;                  // 0
    uint   attestation_apply_count; // 4
    uint   model_apply_count;       // 8
    uint   anchor_apply_count;      // 12
    uint   active_attestations;     // 16
    uint   expired_attestations;    // 20
    uint   model_count;             // 24
    uint   anchor_count;            // 28
    ulong  epoch;                   // 32
    ulong  total_models;            // 40
    ulong  total_anchors;           // 48
    ulong  _pad0;                   // 56
    uchar  attestation_root[32];    // 64
    uchar  model_registry_root[32]; // 96
    uchar  audit_root[32];          // 128
    uchar  aivm_state_root[32];     // 160 -> 192
};

// =============================================================================
// Constants — match aivm_gpu_layout.hpp / aivm_cpu_reference.cpp
// =============================================================================

constant uint kAttStatusVerified = 0x2u;
constant uint kAttStatusExpired  = 0x4u;

constant uint kModeAttestation   = 0u;
constant uint kModeProvenance    = 1u;
constant uint kModeAnchor        = 2u;
constant uint kModeEpoch         = 3u;
constant uint kModeFullRound     = 4u;

constant uint kModelOpRegister      = 0u;
constant uint kModelOpUpdateWeights = 1u;
constant uint kModelOpUpdateLicense = 2u;
constant uint kModelOpTransfer      = 3u;

// =============================================================================
// keccak256 — bit-identical to aivm_cpu_reference.cpp
// =============================================================================

constant ulong kKeccakRC[24] = {
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

constant uint kKeccakRot[25] = {
     0,  1, 62, 28, 27,
    36, 44,  6, 55, 20,
     3, 10, 43, 25, 39,
    41, 45, 15, 21,  8,
    18,  2, 61, 56, 14,
};

inline ulong rotl64(ulong x, uint n) {
    // Match aivm_cpu_reference.cpp: mask shift count to avoid UB at n=0.
    // kKeccakRot[0] == 0, so this path is hit every round.
    return (x << (n & 63u)) | (x >> ((64u - n) & 63u));
}

inline void keccak_f1600(thread ulong* s) {
    for (uint round = 0; round < 24u; ++round) {
        ulong c[5];
        for (uint x = 0; x < 5u; ++x)
            c[x] = s[x] ^ s[x+5] ^ s[x+10] ^ s[x+15] ^ s[x+20];
        ulong d[5];
        for (uint x = 0; x < 5u; ++x)
            d[x] = c[(x + 4u) % 5u] ^ rotl64(c[(x + 1u) % 5u], 1u);
        for (uint y = 0; y < 25u; y += 5u)
            for (uint x = 0; x < 5u; ++x)
                s[y + x] ^= d[x];
        ulong b[25];
        for (uint y = 0; y < 5u; ++y)
            for (uint x = 0; x < 5u; ++x) {
                uint i = x + 5u * y;
                uint j = y + 5u * ((2u * x + 3u * y) % 5u);
                b[j] = rotl64(s[i], kKeccakRot[i]);
            }
        for (uint y = 0; y < 25u; y += 5u) {
            ulong t0 = b[y+0], t1 = b[y+1], t2 = b[y+2], t3 = b[y+3], t4 = b[y+4];
            s[y+0] = t0 ^ ((~t1) & t2);
            s[y+1] = t1 ^ ((~t2) & t3);
            s[y+2] = t2 ^ ((~t3) & t4);
            s[y+3] = t3 ^ ((~t4) & t0);
            s[y+4] = t4 ^ ((~t0) & t1);
        }
        s[0] ^= kKeccakRC[round];
    }
}

inline void keccak256(thread const uchar* data, ulong len, thread uchar* out) {
    ulong s[25] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    const uint rate = 136u;
    ulong off = 0;
    while (len - off >= rate) {
        for (uint i = 0; i < rate; ++i) {
            uint lane = i / 8u, sh = (i % 8u) * 8u;
            s[lane] ^= ((ulong)data[off + i]) << sh;
        }
        keccak_f1600(s);
        off += rate;
    }
    uchar block[136] = {};
    ulong rem = len - off;
    for (ulong i = 0; i < rem; ++i) block[i] = data[off + i];
    block[rem]      ^= 0x01;
    block[rate - 1] ^= 0x80;
    for (uint i = 0; i < rate; ++i) {
        uint lane = i / 8u, sh = (i % 8u) * 8u;
        s[lane] ^= ((ulong)block[i]) << sh;
    }
    keccak_f1600(s);
    for (uint i = 0; i < 32u; ++i) {
        uint lane = i / 8u, sh = (i % 8u) * 8u;
        out[i] = (uchar)((s[lane] >> sh) & 0xFFu);
    }
}

inline void absorb_u32(thread uchar* dst, uint off, uint v) {
    for (uint k = 0; k < 4u; ++k) dst[off + k] = (uchar)((v >> (k*8u)) & 0xFFu);
}
inline void absorb_u64(thread uchar* dst, uint off, ulong v) {
    for (uint k = 0; k < 8u; ++k) dst[off + k] = (uchar)((v >> (k*8u)) & 0xFFu);
}

// =============================================================================
// Open-addressing helpers
// =============================================================================

inline ulong key_from_digest(thread const uchar* d) {
    ulong k = 0;
    for (uint i = 0; i < 8u; ++i) k |= ((ulong)d[i]) << (i*8u);
    return k;
}
inline ulong key_from_digest_dev(device const uchar* d) {
    ulong k = 0;
    for (uint i = 0; i < 8u; ++i) k |= ((ulong)d[i]) << (i*8u);
    return k;
}

inline uint hash_index(ulong k, uint mask) {
    ulong h = 0xcbf29ce484222325UL;
    h = (h ^ k) * 0x100000001b3UL;
    return (uint)h & mask;
}

inline bool digest_equal_dev_thr(device const uchar* a, thread const uchar* b) {
    for (uint i = 0; i < 32u; ++i) if (a[i] != b[i]) return false;
    return true;
}
inline bool digest_zero_thr(thread const uchar* a) {
    for (uint i = 0; i < 32u; ++i) if (a[i] != 0u) return false;
    return true;
}
inline bool key48_zero_thr(thread const uchar* a) {
    for (uint i = 0; i < 48u; ++i) if (a[i] != 0u) return false;
    return true;
}
inline bool digest_zero_dev(device const uchar* a) {
    for (uint i = 0; i < 32u; ++i) if (a[i] != 0u) return false;
    return true;
}

inline uint attestation_locate(device Attestation* tab, uint count,
                               thread const uchar* digest,
                               bool insert_if_missing)
{
    uint mask = count - 1u;
    ulong key = key_from_digest(digest);
    uint idx  = hash_index(key, mask);
    for (uint probe = 0; probe < count; ++probe) {
        device Attestation& s = tab[idx];
        if (s.occupied == 0u) {
            if (insert_if_missing) {
                for (uint k = 0; k < 32u; ++k) s.tee_quote_digest[k] = digest[k];
                for (uint k = 0; k < 32u; ++k) s.measurement[k] = 0u;
                for (uint k = 0; k < 48u; ++k) s.attesting_key[k] = 0u;
                s.expiry_ns = 0u;
                s.kind = 0u;
                s.evidence_offset = 0u;
                s.evidence_len = 0u;
                s.status = 0u;
                s.occupied = 1u;
                return idx;
            }
            return 0xFFFFFFFFu;
        }
        if (digest_equal_dev_thr(s.tee_quote_digest, digest)) return idx;
        idx = (idx + 1u) & mask;
    }
    return 0xFFFFFFFFu;
}

inline uint model_locate(device ModelRegistryEntry* tab, uint count,
                         thread const uchar* model_root,
                         bool insert_if_missing)
{
    uint mask = count - 1u;
    ulong key = key_from_digest(model_root);
    uint idx  = hash_index(key, mask);
    for (uint probe = 0; probe < count; ++probe) {
        device ModelRegistryEntry& s = tab[idx];
        if (s.occupied == 0u) {
            if (insert_if_missing) {
                for (uint k = 0; k < 32u; ++k) s.model_root[k] = model_root[k];
                for (uint k = 0; k < 32u; ++k) s.weight_hash[k] = 0u;
                for (uint k = 0; k < 32u; ++k) s.license_root[k] = 0u;
                for (uint k = 0; k < 20u; ++k) s.owner_addr[k] = 0u;
                s.version = 0u;
                s.parameter_count = 0u;
                s.modality = 0u;
                s.occupied = 1u;
                return idx;
            }
            return 0xFFFFFFFFu;
        }
        if (digest_equal_dev_thr(s.model_root, model_root)) return idx;
        idx = (idx + 1u) & mask;
    }
    return 0xFFFFFFFFu;
}
