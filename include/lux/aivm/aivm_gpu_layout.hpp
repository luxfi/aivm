// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file aivm_gpu_layout.hpp
/// Shared host/GPU memory layouts for the AIVM (AI-VM, A-Chain) transition
/// substrate.
///
/// **Scope** — this module is the **GPU-native transition substrate for the
/// A-Chain**. State (TEE attestations, AI model registry, audit anchors)
/// lives on the GPU and the canonical transition logic also runs on the GPU.
/// Pairs with PVM (P-Chain), CEVM (C-Chain), XVM (X-Chain), BridgeVM
/// (B-Chain), MPCVM (M-Chain) under the same LP-137 contract.
///
///   aivm/                          the substrate this header describes
///   aivm/src/aivm_cpu_reference.cpp deterministic CPU oracle (this file's twin)
///   aivm/src/aivm_*.cu             CUDA transition kernels
///   aivm/src/aivm_*.metal          Metal transition kernels
///   aivm/src/aivm_*.wgsl           WGSL transition kernels
///
/// Round model — one AIVMTransitionRound covers:
///   * AttestationApply  : register/refresh TEE attestations (SGX/SEV/TDX/NV)
///   * ProvenanceApply   : register AI models, rotate weight hashes,
///                         update license commitments
///   * AnchorApply       : append commit anchor with parent-root chaining
///   * EpochTransition   : finalise per-epoch attestation set, prune expired
///                         entries, emit aivm_state_root for cross-chain
///                         binding
///
/// All offsets here MUST match aivm_*.metal, aivm_*.cu, and aivm_*.wgsl
/// byte-for-byte. The CPU reference in aivm_cpu_reference.cpp consumes the
/// same arenas in the same order and produces identical roots — it is the
/// equivalence oracle for cross-backend determinism (CPU vs Metal vs CUDA
/// vs WGSL).

#pragma once

#include <cstdint>

namespace aivm::gpu {

// =============================================================================
// Residency class tags (mirrors PVM)
// =============================================================================

enum class ResidencyClass : uint32_t {
    DeviceHot = 0,
    HostCold  = 1,
};

// =============================================================================
// Attestation kinds — TEE quote families
// =============================================================================

enum class AttestationKind : uint32_t {
    SGX     = 0,   ///< Intel SGX
    SEV     = 1,   ///< AMD SEV-SNP
    TDX     = 2,   ///< Intel TDX
    NvTrust = 3,   ///< NVIDIA NVTrust (H100/H200/B100/B200/GB200)
};

// =============================================================================
// Attestation arena (DeviceHot) — open-addressing keyed by tee_quote_digest[0..8]
// =============================================================================
//
// Each slot binds a TEE quote to an attesting public key with an expiry.
// The kernel verifies the quote signature in batches by linking against
// quasar_bls_verifier / quasar_ringtail_verifier / quasar_groth16_verifier
// (host-side decision based on kind), and only commits the entry if the
// signature verifies AND the quote has not expired.
//
// status bits:
//   0x1  occupied      — slot in use (set on insert)
//   0x2  verified      — quote signature checked + bound to attesting_key
//   0x4  expired       — pruned at epoch close

struct alignas(16) Attestation {
    uint8_t  tee_quote_digest[32];   ///< 0    keccak of the original quote blob
    uint8_t  measurement[32];        ///< 32   measurement / runtime hash (PCR0..)
    uint8_t  attesting_key[48];      ///< 64   BLS12-381 G1 compressed (or padded smaller key)
    uint64_t expiry_ns;              ///< 112  unix nanoseconds; 0 = never
    uint32_t kind;                   ///< 120  AttestationKind
    uint32_t evidence_offset;        ///< 124  byte offset into evidence pool (host-managed)
    uint32_t evidence_len;           ///< 128  byte length of evidence blob
    uint32_t status;                 ///< 132  status bits above
    uint32_t occupied;               ///< 136  0=free, 1=occupied
    uint32_t _pad0;                  ///< 140  -> 144
};
static_assert(sizeof(Attestation) == 144, "Attestation layout drift");
static_assert(alignof(Attestation) == 16, "Attestation alignment drift");

inline constexpr uint32_t kDefaultAttestationSlots = 1024u;

// =============================================================================
// Model registry arena (DeviceHot) — open-addressing keyed by model_root[0..8]
// =============================================================================
//
// Each slot describes a registered AI model:
//   * model_root  — Merkle root over weights + tokenizer + config
//   * weight_hash — keccak of the canonical weight tensor blob
//   * license_root — Merkle root of license commitments
//   * owner_addr  — 20-byte EVM address of registrant
//   * version     — monotonic version counter (for weight rotation)
//   * parameter_count — total trainable parameter count
//   * modality    — text/vision/audio/multi

enum class Modality : uint32_t {
    Text   = 0,
    Vision = 1,
    Audio  = 2,
    Multi  = 3,
};

struct alignas(16) ModelRegistryEntry {
    uint8_t  model_root[32];        ///< 0    Merkle root over weights + tokenizer + config
    uint8_t  weight_hash[32];       ///< 32   keccak of canonical weight blob
    uint8_t  license_root[32];      ///< 64   Merkle root of license commitments
    uint8_t  owner_addr[20];        ///< 96   EVM address (20 bytes)
    uint32_t _pad0;                 ///< 116
    uint64_t version;               ///< 120  monotonic version
    uint64_t parameter_count;       ///< 128  total parameters
    uint32_t modality;              ///< 136  Modality
    uint32_t occupied;              ///< 140  0=free, 1=occupied
    uint64_t _pad1;                 ///< 144
    uint64_t _pad2;                 ///< 152  -> 160
};
static_assert(sizeof(ModelRegistryEntry) == 160, "ModelRegistryEntry layout drift");
static_assert(alignof(ModelRegistryEntry) == 16, "ModelRegistryEntry alignment drift");

inline constexpr uint32_t kDefaultModelSlots = 512u;

// =============================================================================
// Audit anchor arena (DeviceHot) — append-only ring-buffer of commit roots
// =============================================================================
//
// Each slot binds (commit_root, parent_root, validator_set_root_at_commit,
// height, timestamp_ns) into the audit chain. parent_root chains entries
// to form a binary commitment chain — verifying the audit_root is
// equivalent to verifying every commit ever anchored.

struct alignas(16) AuditAnchor {
    uint8_t  commit_root[32];                  ///< 0
    uint8_t  parent_root[32];                  ///< 32
    uint8_t  validator_set_root_at_commit[32]; ///< 64
    uint64_t height;                           ///< 96
    uint64_t timestamp_ns;                     ///< 104
    uint32_t occupied;                         ///< 112  0=free, 1=occupied
    uint32_t _pad0;                            ///< 116
    uint64_t _pad1;                            ///< 120  -> 128
};
static_assert(sizeof(AuditAnchor) == 128, "AuditAnchor layout drift");
static_assert(alignof(AuditAnchor) == 16, "AuditAnchor alignment drift");

inline constexpr uint32_t kDefaultAnchorSlots = 4096u;

// =============================================================================
// Epoch state (DeviceHot)
// =============================================================================

struct alignas(16) AIVMEpochState {
    uint64_t current_epoch;                  ///< 0
    uint64_t next_epoch_height;              ///< 8
    uint64_t total_active_attestations;      ///< 16
    uint32_t active_model_count;             ///< 24
    uint32_t expired_attestation_count;      ///< 28
    uint8_t  attestation_root[32];           ///< 32
    uint8_t  model_registry_root[32];        ///< 64
    uint8_t  audit_root[32];                 ///< 96
    uint8_t  aivm_state_root[32];            ///< 128 -> 160
};
static_assert(sizeof(AIVMEpochState) == 160, "AIVMEpochState layout drift");
static_assert(alignof(AIVMEpochState) == 16, "AIVMEpochState alignment drift");

// =============================================================================
// Round descriptor (host -> GPU, written once per round)
// =============================================================================

enum class AIVMTransitionMode : uint32_t {
    AttestationApply = 0,
    ProvenanceApply  = 1,
    AnchorApply      = 2,
    EpochTransition  = 3,
    FullRound        = 4,   ///< chain all four in canonical order
};

struct alignas(16) AIVMRoundDescriptor {
    uint64_t chain_id;            ///< 0   A-Chain canonical id
    uint64_t round;               ///< 8   monotonic
    uint64_t timestamp_ns;        ///< 16
    uint64_t epoch;               ///< 24
    uint32_t mode;                ///< 32  AIVMTransitionMode
    uint32_t attestation_op_count;///< 36
    uint32_t model_op_count;      ///< 40
    uint32_t anchor_op_count;     ///< 44
    uint32_t closing_flag;        ///< 48  1 = run EpochTransition after the others
    uint32_t _pad0;               ///< 52
    uint64_t _pad1;               ///< 56
    uint8_t  parent_aivm_root[32];///< 64  -> 96
};
static_assert(sizeof(AIVMRoundDescriptor) == 96, "AIVMRoundDescriptor layout drift");
static_assert(alignof(AIVMRoundDescriptor) == 16, "AIVMRoundDescriptor alignment drift");

// =============================================================================
// Operation streams (host-supplied input)
// =============================================================================
//
// AttestationOp — a single attestation submission. Layout matches
// Attestation byte-for-byte so we can reuse the keccak leaf format for
// the verifier's binding hash.

struct alignas(16) AttestationOp {
    uint8_t  tee_quote_digest[32];   ///< 0
    uint8_t  measurement[32];        ///< 32
    uint8_t  attesting_key[48];      ///< 64
    uint64_t expiry_ns;              ///< 112
    uint32_t kind;                   ///< 120  AttestationKind
    uint32_t evidence_offset;        ///< 124
    uint32_t evidence_len;           ///< 128
    uint32_t epoch;                  ///< 132
    uint32_t _pad0;                  ///< 136
    uint32_t _pad1;                  ///< 140  -> 144
};
static_assert(sizeof(AttestationOp) == 144, "AttestationOp layout drift");
static_assert(alignof(AttestationOp) == 16, "AttestationOp alignment drift");

// ModelOp kinds.
enum class ModelOpKind : uint32_t {
    Register      = 0,
    UpdateWeights = 1,
    UpdateLicense = 2,
    Transfer      = 3,
};

struct alignas(16) ModelOp {
    uint8_t  model_root[32];        ///< 0
    uint8_t  weight_hash[32];       ///< 32
    uint8_t  license_root[32];      ///< 64
    uint8_t  owner_addr[20];        ///< 96
    uint32_t _pad0;                 ///< 116
    uint64_t parameter_count;       ///< 120
    uint32_t modality;              ///< 128
    uint32_t kind;                  ///< 132  ModelOpKind
    uint32_t epoch;                 ///< 136
    uint32_t _pad1;                 ///< 140
    uint64_t _pad2;                 ///< 144
    uint64_t _pad3;                 ///< 152 -> 160
};
static_assert(sizeof(ModelOp) == 160, "ModelOp layout drift");
static_assert(alignof(ModelOp) == 16, "ModelOp alignment drift");

struct alignas(16) AnchorOp {
    uint8_t  commit_root[32];                  ///< 0
    uint8_t  parent_root[32];                  ///< 32
    uint8_t  validator_set_root_at_commit[32]; ///< 64
    uint64_t height;                           ///< 96
    uint64_t timestamp_ns;                     ///< 104
    uint32_t epoch;                            ///< 112
    uint32_t _pad0;                            ///< 116
    uint64_t _pad1;                            ///< 120 -> 128
};
static_assert(sizeof(AnchorOp) == 128, "AnchorOp layout drift");
static_assert(alignof(AnchorOp) == 16, "AnchorOp alignment drift");

// =============================================================================
// Round result (GPU -> host)
// =============================================================================

struct alignas(16) AIVMTransitionResult {
    uint32_t status;                  ///< 0   0=in-progress, 1=finalized, 2=needs_state, 3=failed
    uint32_t attestation_apply_count; ///< 4   attestation ops applied (verified + accepted)
    uint32_t model_apply_count;       ///< 8
    uint32_t anchor_apply_count;      ///< 12
    uint32_t active_attestations;     ///< 16
    uint32_t expired_attestations;    ///< 20
    uint32_t model_count;             ///< 24
    uint32_t anchor_count;            ///< 28
    uint64_t epoch;                   ///< 32
    uint64_t total_models;            ///< 40
    uint64_t total_anchors;           ///< 48
    uint64_t _pad0;                   ///< 56
    uint8_t  attestation_root[32];    ///< 64
    uint8_t  model_registry_root[32]; ///< 96
    uint8_t  audit_root[32];          ///< 128
    uint8_t  aivm_state_root[32];     ///< 160 -> 192
};
static_assert(sizeof(AIVMTransitionResult) == 192, "AIVMTransitionResult layout drift");
static_assert(alignof(AIVMTransitionResult) == 16, "AIVMTransitionResult alignment drift");

}  // namespace aivm::gpu
