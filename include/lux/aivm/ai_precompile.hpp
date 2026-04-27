// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file ai_precompile.hpp
/// AIVM AI precompile service (v0.59.1, AI/ML v0.1).
///
/// Five operations, three execution modes. Model-bound — not arbitrary GPU
/// inference in EVM. The host caller selects the model by `model_hash`, the
/// service verifies the signature, runs a fixed quantized kernel, and binds
/// the (model_hash, input_commitment, output_commitment) tuple into the
/// AIVM state root via `AIPrecompileArtifact`.
///
/// Operations
///   ModelVerify       0x0a01 — verify model_hash + signature; load weights
///                              into the device-warm pool
///   InferenceCommit   0x0a02 — commit input -> input_commitment Hash
///   InferenceExecute  0x0a03 — run inference; return output_commitment
///   ResultAttest      0x0a04 — produce attestation over (model_hash, input,
///                              output, policy_hash, timestamp) -> attestation_root
///   AuditCommit       0x0a05 — append to audit log (folded into AIVM
///                              audit_root via AnchorOp by the caller)
///
/// Execution modes
///   DeterministicInteger  — int8/int4 quantized, fixed accumulator (int32),
///                           fixed reduction order (canonical row-major), fixed
///                           saturation. Byte-equal across CPU + Metal +
///                           CUDA + WGSL backends.
///   ConfidentialInference — same kernel as Mode 1, but the host never sees
///                           the raw input or output. Only commitments are
///                           returned. Sealed device buffer holds the result.
///                           Real TEE quote stand-in lands in CC v0.1
///                           (sibling agent).
///   VerifiableInference   — Mode 1 + sampled deterministic replay (1-of-K,
///                           K=10, seeded by round_id). If any sampled input
///                           diverges, status=failed. zkML proofs land in
///                           v0.2.
///
/// AIPrecompileArtifact composition
///   The artifact is bound into the AIVM execution_root keccak input by the
///   caller (cevm precompile dispatch). Host code MUST forward the artifact
///   bytes verbatim into the AnchorOp.commit_root preimage so the audit_root
///   reflects every inference. See test_artifact_root_composition.

#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace aivm::ai {

using Hash = std::array<uint8_t, 32>;

// =============================================================================
// Execution modes
// =============================================================================

enum class AIExecutionMode : uint8_t {
    DeterministicInteger  = 0,
    ConfidentialInference = 1,
    VerifiableInference   = 2,
};

// =============================================================================
// Precompile op codes
// =============================================================================

enum class AIPrecompileOp : uint16_t {
    ModelVerify      = 0x0a01,
    InferenceCommit  = 0x0a02,
    InferenceExecute = 0x0a03,
    ResultAttest     = 0x0a04,
    AuditCommit      = 0x0a05,
};

// =============================================================================
// Status codes
// =============================================================================

enum class AIStatus : uint16_t {
    Ok                  = 0,
    UnknownModel        = 1,
    SignatureRejected   = 2,
    InputTooLarge       = 3,
    OutputCapacity      = 4,
    DivergenceDetected  = 5,  // Mode 3 — sampled replay caught a bit-flip
    InternalError       = 6,
};

// =============================================================================
// Artifact bound into execution_root + audit_root
// =============================================================================
//
// Flag bits:
//   bit 0 — DeterministicInteger
//   bit 1 — ConfidentialInference (sealed input/output, only commitments)
//   bit 2 — VerifiableInference   (sampled replay verified)

struct alignas(16) AIPrecompileArtifact {
    Hash     model_hash;
    Hash     model_config_hash;
    Hash     input_commitment;
    Hash     output_commitment;
    Hash     attestation_root;     // zero unless mode == Confidential or Verifiable
    Hash     policy_root;
    uint32_t batch_count;
    uint32_t flags;

    /// Canonical 200-byte preimage for binding into execution_root / audit_root.
    /// The caller folds these bytes into AnchorOp.commit_root keccak input.
    static constexpr std::size_t kPreimageBytes =
        32 + 32 + 32 + 32 + 32 + 32 + 4 + 4;

    void encode_preimage(uint8_t out[kPreimageBytes]) const noexcept;
    Hash root() const noexcept;  // keccak(encode_preimage())
};
static_assert(sizeof(AIPrecompileArtifact) == 208,
              "AIPrecompileArtifact layout drift");

// =============================================================================
// Request / response
// =============================================================================

struct AIInferenceCall {
    Hash            model_hash;
    AIExecutionMode mode;
    uint32_t        input_offset;     // into the host-supplied call data blob
    uint32_t        input_len;
    uint32_t        output_offset;
    uint32_t        output_capacity;
    Hash            policy_hash;
    uint64_t        round_id;         // seeds Mode 3 sampling
    uint64_t        timestamp_ns;     // bound into attestation_root
    uint8_t         salt[32];         // per-call fresh randomness
};

struct AIInferenceResult {
    AIStatus status;
    uint32_t output_len;
    Hash     input_commitment;
    Hash     output_commitment;
    Hash     attestation_root;        // zero unless mode == Confidential or Verifiable
};

// =============================================================================
// Tiny classifier fixture (32 -> 16 -> 1, int8 weights, int8 inputs)
// =============================================================================
//
// File format (little-endian, packed):
//   magic[4]      = "AIVM"
//   version[4]    = 1
//   layers[2]     = 2
//   in_dim[2]     = 32
//   hidden[2]     = 16
//   out_dim[2]    = 1
//   _pad[16]      = 0
//   W1[32*16]     int8 weights, row-major (input row-major)
//   B1[16]        int32 little-endian bias
//   W2[16*1]      int8 weights
//   B2[1]         int32 little-endian bias
//   shift1[1]     int8 right-shift after layer 1 saturation
//   shift2[1]     int8 right-shift after layer 2 saturation

struct TinyClassifier {
    static constexpr uint32_t kInDim  = 32;
    static constexpr uint32_t kHidden = 16;
    static constexpr uint32_t kOutDim = 1;

    std::array<int8_t,  kInDim * kHidden>  w1{};
    std::array<int32_t, kHidden>           b1{};
    std::array<int8_t,  kHidden * kOutDim> w2{};
    std::array<int32_t, kOutDim>           b2{};
    int8_t shift1 = 7;
    int8_t shift2 = 7;
    Hash   model_hash{};
    Hash   model_config_hash{};
};

/// Generate the canonical fixture (deterministic, seeded). Same bytes every
/// call — this is the v0.1 test vector.
TinyClassifier make_tiny_classifier_fixture();

/// Serialize fixture to bytes. Round-trip stable.
std::vector<uint8_t> tiny_classifier_to_bytes(const TinyClassifier& m);

/// Parse fixture bytes. Returns empty hash on malformed input.
TinyClassifier tiny_classifier_from_bytes(const std::vector<uint8_t>& bytes);

// =============================================================================
// AIPrecompile service
// =============================================================================
//
// Stateful — verified models are cached by model_hash. Construct once, call
// many times. Thread-safe at the verify/run level.

class AIPrecompile {
public:
    AIPrecompile();
    ~AIPrecompile();

    AIPrecompile(const AIPrecompile&) = delete;
    AIPrecompile& operator=(const AIPrecompile&) = delete;

    // Op 0x0a01 — verify and admit a model. signature is a stand-in: any
    // non-zero 64-byte signature is accepted as long as the model's
    // self-consistent hash matches model_hash. Real Ed25519 / BLS lands in
    // CC v0.1.
    AIStatus model_verify(const std::vector<uint8_t>& model_bytes,
                          const Hash& model_hash,
                          const std::array<uint8_t, 64>& signature);

    // Op 0x0a02 — commit to input bytes. Returns Hash = keccak(salt || input).
    static Hash inference_commit(const uint8_t salt[32],
                                 const uint8_t* input, std::size_t len);

    // Op 0x0a03 — run inference. Writes up to `call.output_capacity` bytes
    // into `output` (host-visible). For Mode 2 (Confidential), the host
    // buffer is zeroed after execute and only the commitment is exposed.
    AIInferenceResult inference_execute(const AIInferenceCall& call,
                                        const uint8_t* input,
                                        uint8_t* output);

    // Op 0x0a04 — produce attestation root binding model, input, output,
    // policy, and timestamp. v0.1 stand-in for real TEE quote.
    static Hash result_attest(const Hash& model_hash,
                              const Hash& input_commitment,
                              const Hash& output_commitment,
                              const Hash& policy_hash,
                              uint64_t timestamp_ns);

    // Op 0x0a05 — derive audit-commit hash. Caller folds this into a
    // standard AIVM AnchorOp.commit_root and pushes through the existing
    // audit_root pipeline.
    static Hash audit_commit(const AIPrecompileArtifact& artifact);

    // Run the full op-chain and return the artifact for execution_root
    // binding. This is the single-shot entry point the EVM precompile uses.
    AIInferenceResult execute(const AIInferenceCall& call,
                              const uint8_t* input,
                              uint8_t* output,
                              AIPrecompileArtifact& artifact_out);

    // Optional Metal acceleration. If set, Mode 1 / Mode 2 / Mode 3 dispatch
    // their GEMM through the Metal driver. The byte-equal contract still
    // holds: Metal output must match CPU output for the same call.
    bool has_metal_engine() const noexcept;
    bool enable_metal_engine();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace aivm::ai
