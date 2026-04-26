// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file aivm_cpu_reference.hpp
/// CPU reference implementation of the AIVM transition kernels — the
/// differential-fuzz oracle for cross-backend determinism (CPU vs Metal
/// vs CUDA vs WGSL must produce byte-identical roots on the same input).
///
/// The reference processes input ops in canonical order:
///   1. AttestationApply   (attestation_ops, in order)
///   2. ProvenanceApply    (model_ops, in order)
///   3. AnchorApply        (anchor_ops, in order)
///   4. EpochTransition    (closes the epoch if desc.closing_flag != 0)
///
/// State carries forward across run_reference() calls only via the
/// attestation/model/anchor arenas the caller threads through — the
/// reference is pure on (state, ops) and does not retain hidden state.

#pragma once

#include "aivm_gpu_layout.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace aivm::gpu::ref {

struct AIVMReferenceState {
    std::vector<Attestation>         attestations;  ///< sized to kDefaultAttestationSlots
    std::vector<ModelRegistryEntry>  models;        ///< sized to kDefaultModelSlots
    std::vector<AuditAnchor>         anchors;       ///< sized to kDefaultAnchorSlots
    AIVMEpochState                   epoch{};

    static AIVMReferenceState empty();
};

AIVMTransitionResult run_reference(AIVMReferenceState& state,
                                   const AIVMRoundDescriptor& desc,
                                   std::span<const AttestationOp> attestation_ops,
                                   std::span<const ModelOp>       model_ops,
                                   std::span<const AnchorOp>      anchor_ops);

}  // namespace aivm::gpu::ref
