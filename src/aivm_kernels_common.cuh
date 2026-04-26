// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// aivm_kernels_common.cuh — shared device code for the four CUDA AIVM kernels.
// Layout MUST match aivm_gpu_layout.hpp byte-for-byte.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace aivm::cuda {

struct alignas(16) Attestation {
    uint8_t  tee_quote_digest[32];
    uint8_t  measurement[32];
    uint8_t  attesting_key[48];
    uint64_t expiry_ns;
    uint32_t kind;
    uint32_t evidence_offset;
    uint32_t evidence_len;
    uint32_t status;
    uint32_t occupied;
    uint32_t _pad0;
};

struct alignas(16) ModelRegistryEntry {
    uint8_t  model_root[32];
    uint8_t  weight_hash[32];
    uint8_t  license_root[32];
    uint8_t  owner_addr[20];
    uint32_t _pad0;
    uint64_t version;
    uint64_t parameter_count;
    uint32_t modality;
    uint32_t occupied;
    uint64_t _pad1;
    uint64_t _pad2;
};

struct alignas(16) AuditAnchor {
    uint8_t  commit_root[32];
    uint8_t  parent_root[32];
    uint8_t  validator_set_root_at_commit[32];
    uint64_t height;
    uint64_t timestamp_ns;
    uint32_t occupied;
    uint32_t _pad0;
    uint64_t _pad1;
};

struct alignas(16) AIVMEpochState {
    uint64_t current_epoch;
    uint64_t next_epoch_height;
    uint64_t total_active_attestations;
    uint32_t active_model_count;
    uint32_t expired_attestation_count;
    uint8_t  attestation_root[32];
    uint8_t  model_registry_root[32];
    uint8_t  audit_root[32];
    uint8_t  aivm_state_root[32];
};

struct alignas(16) AIVMRoundDescriptor {
    uint64_t chain_id;
    uint64_t round;
    uint64_t timestamp_ns;
    uint64_t epoch;
    uint32_t mode;
    uint32_t attestation_op_count;
    uint32_t model_op_count;
    uint32_t anchor_op_count;
    uint32_t closing_flag;
    uint32_t _pad0;
    uint64_t _pad1;
    uint8_t  parent_aivm_root[32];
};

struct alignas(16) AttestationOp {
    uint8_t  tee_quote_digest[32];
    uint8_t  measurement[32];
    uint8_t  attesting_key[48];
    uint64_t expiry_ns;
    uint32_t kind;
    uint32_t evidence_offset;
    uint32_t evidence_len;
    uint32_t epoch;
    uint32_t _pad0;
    uint32_t _pad1;
};

struct alignas(16) ModelOp {
    uint8_t  model_root[32];
    uint8_t  weight_hash[32];
    uint8_t  license_root[32];
    uint8_t  owner_addr[20];
    uint32_t _pad0;
    uint64_t parameter_count;
    uint32_t modality;
    uint32_t kind;
    uint32_t epoch;
    uint32_t _pad1;
    uint64_t _pad2;
    uint64_t _pad3;
};

struct alignas(16) AnchorOp {
    uint8_t  commit_root[32];
    uint8_t  parent_root[32];
    uint8_t  validator_set_root_at_commit[32];
    uint64_t height;
    uint64_t timestamp_ns;
    uint32_t epoch;
    uint32_t _pad0;
    uint64_t _pad1;
};

struct alignas(16) AIVMTransitionResult {
    uint32_t status;
    uint32_t attestation_apply_count;
    uint32_t model_apply_count;
    uint32_t anchor_apply_count;
    uint32_t active_attestations;
    uint32_t expired_attestations;
    uint32_t model_count;
    uint32_t anchor_count;
    uint64_t epoch;
    uint64_t total_models;
    uint64_t total_anchors;
    uint64_t _pad0;
    uint8_t  attestation_root[32];
    uint8_t  model_registry_root[32];
    uint8_t  audit_root[32];
    uint8_t  aivm_state_root[32];
};

constexpr uint32_t kAttStatusVerified = 0x2u;
constexpr uint32_t kAttStatusExpired  = 0x4u;

constexpr uint32_t kModeAttestation   = 0u;
constexpr uint32_t kModeProvenance    = 1u;
constexpr uint32_t kModeAnchor        = 2u;
constexpr uint32_t kModeEpoch         = 3u;
constexpr uint32_t kModeFullRound     = 4u;

constexpr uint32_t kModelOpRegister      = 0u;
constexpr uint32_t kModelOpUpdateWeights = 1u;
constexpr uint32_t kModelOpUpdateLicense = 2u;
constexpr uint32_t kModelOpTransfer      = 3u;

__constant__ static const uint64_t kKeccakRC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL,
    0x800000000000808AULL, 0x8000000080008000ULL,
    0x000000000000808BULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008AULL, 0x0000000000000088ULL,
    0x0000000080008009ULL, 0x000000008000000AULL,
    0x000000008000808BULL, 0x800000000000008BULL,
    0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800AULL, 0x800000008000000AULL,
    0x8000000080008081ULL, 0x8000000000008080ULL,
    0x0000000080000001ULL, 0x8000000080008008ULL,
};

__constant__ static const uint32_t kKeccakRot[25] = {
     0,  1, 62, 28, 27,
    36, 44,  6, 55, 20,
     3, 10, 43, 25, 39,
    41, 45, 15, 21,  8,
    18,  2, 61, 56, 14,
};

__device__ inline uint64_t rotl64(uint64_t x, uint32_t n) {
    return (x << n) | (x >> (64u - n));
}

__device__ inline void keccak_f1600(uint64_t* s) {
    for (uint32_t round = 0; round < 24u; ++round) {
        uint64_t c[5];
        for (uint32_t x = 0; x < 5u; ++x)
            c[x] = s[x] ^ s[x+5] ^ s[x+10] ^ s[x+15] ^ s[x+20];
        uint64_t d[5];
        for (uint32_t x = 0; x < 5u; ++x)
            d[x] = c[(x + 4u) % 5u] ^ rotl64(c[(x + 1u) % 5u], 1u);
        for (uint32_t y = 0; y < 25u; y += 5u)
            for (uint32_t x = 0; x < 5u; ++x)
                s[y + x] ^= d[x];
        uint64_t b[25];
        for (uint32_t y = 0; y < 5u; ++y)
            for (uint32_t x = 0; x < 5u; ++x) {
                uint32_t i = x + 5u * y;
                uint32_t j = y + 5u * ((2u * x + 3u * y) % 5u);
                b[j] = rotl64(s[i], kKeccakRot[i]);
            }
        for (uint32_t y = 0; y < 25u; y += 5u) {
            uint64_t t0 = b[y+0], t1 = b[y+1], t2 = b[y+2], t3 = b[y+3], t4 = b[y+4];
            s[y+0] = t0 ^ ((~t1) & t2);
            s[y+1] = t1 ^ ((~t2) & t3);
            s[y+2] = t2 ^ ((~t3) & t4);
            s[y+3] = t3 ^ ((~t4) & t0);
            s[y+4] = t4 ^ ((~t0) & t1);
        }
        s[0] ^= kKeccakRC[round];
    }
}

__device__ inline void keccak256(const uint8_t* data, uint64_t len, uint8_t* out) {
    uint64_t s[25] = {0};
    constexpr uint32_t rate = 136u;
    uint64_t off = 0;
    while (len - off >= rate) {
        for (uint32_t i = 0; i < rate; ++i) {
            uint32_t lane = i / 8u, sh = (i % 8u) * 8u;
            s[lane] ^= ((uint64_t)data[off + i]) << sh;
        }
        keccak_f1600(s);
        off += rate;
    }
    uint8_t block[136] = {0};
    uint64_t rem = len - off;
    for (uint64_t i = 0; i < rem; ++i) block[i] = data[off + i];
    block[rem]      ^= 0x01;
    block[rate - 1] ^= 0x80;
    for (uint32_t i = 0; i < rate; ++i) {
        uint32_t lane = i / 8u, sh = (i % 8u) * 8u;
        s[lane] ^= ((uint64_t)block[i]) << sh;
    }
    keccak_f1600(s);
    for (uint32_t i = 0; i < 32u; ++i) {
        uint32_t lane = i / 8u, sh = (i % 8u) * 8u;
        out[i] = (uint8_t)((s[lane] >> sh) & 0xFFu);
    }
}

__device__ inline void absorb_u32(uint8_t* dst, uint32_t off, uint32_t v) {
    for (uint32_t k = 0; k < 4u; ++k) dst[off + k] = (uint8_t)((v >> (k*8u)) & 0xFFu);
}
__device__ inline void absorb_u64(uint8_t* dst, uint32_t off, uint64_t v) {
    for (uint32_t k = 0; k < 8u; ++k) dst[off + k] = (uint8_t)((v >> (k*8u)) & 0xFFu);
}

__device__ inline uint64_t key_from_digest(const uint8_t* d) {
    uint64_t k = 0;
    for (uint32_t i = 0; i < 8u; ++i) k |= ((uint64_t)d[i]) << (i*8u);
    return k;
}

__device__ inline uint32_t hash_index(uint64_t k, uint32_t mask) {
    uint64_t h = 0xcbf29ce484222325ULL;
    h = (h ^ k) * 0x100000001b3ULL;
    return (uint32_t)h & mask;
}

__device__ inline bool digest_equal32(const uint8_t* a, const uint8_t* b) {
    for (uint32_t i = 0; i < 32u; ++i) if (a[i] != b[i]) return false;
    return true;
}
__device__ inline bool digest_zero32(const uint8_t* a) {
    for (uint32_t i = 0; i < 32u; ++i) if (a[i] != 0u) return false;
    return true;
}
__device__ inline bool key48_zero(const uint8_t* a) {
    for (uint32_t i = 0; i < 48u; ++i) if (a[i] != 0u) return false;
    return true;
}

__device__ inline uint32_t attestation_locate(Attestation* tab, uint32_t count,
                                              const uint8_t* digest,
                                              bool insert_if_missing)
{
    uint32_t mask = count - 1u;
    uint64_t key  = key_from_digest(digest);
    uint32_t idx  = hash_index(key, mask);
    for (uint32_t probe = 0; probe < count; ++probe) {
        Attestation& s = tab[idx];
        if (s.occupied == 0u) {
            if (insert_if_missing) {
                for (uint32_t k = 0; k < 32u; ++k) s.tee_quote_digest[k] = digest[k];
                for (uint32_t k = 0; k < 32u; ++k) s.measurement[k]      = 0u;
                for (uint32_t k = 0; k < 48u; ++k) s.attesting_key[k]    = 0u;
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
        if (digest_equal32(s.tee_quote_digest, digest)) return idx;
        idx = (idx + 1u) & mask;
    }
    return 0xFFFFFFFFu;
}

__device__ inline uint32_t model_locate(ModelRegistryEntry* tab, uint32_t count,
                                        const uint8_t* model_root,
                                        bool insert_if_missing)
{
    uint32_t mask = count - 1u;
    uint64_t key  = key_from_digest(model_root);
    uint32_t idx  = hash_index(key, mask);
    for (uint32_t probe = 0; probe < count; ++probe) {
        ModelRegistryEntry& s = tab[idx];
        if (s.occupied == 0u) {
            if (insert_if_missing) {
                for (uint32_t k = 0; k < 32u; ++k) s.model_root[k]   = model_root[k];
                for (uint32_t k = 0; k < 32u; ++k) s.weight_hash[k]  = 0u;
                for (uint32_t k = 0; k < 32u; ++k) s.license_root[k] = 0u;
                for (uint32_t k = 0; k < 20u; ++k) s.owner_addr[k]   = 0u;
                s.version = 0u;
                s.parameter_count = 0u;
                s.modality = 0u;
                s.occupied = 1u;
                return idx;
            }
            return 0xFFFFFFFFu;
        }
        if (digest_equal32(s.model_root, model_root)) return idx;
        idx = (idx + 1u) & mask;
    }
    return 0xFFFFFFFFu;
}

}  // namespace aivm::cuda
