// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// aivm_transition.cu — CUDA peer of aivm_transition.metal.

#include "aivm_kernels_common.cuh"

namespace aivm::cuda {

extern "C" __global__ void aivm_epoch_transition(
    const AIVMRoundDescriptor* desc,
    Attestation*               attestations,
    ModelRegistryEntry*        models,
    AuditAnchor*               anchors,
    AIVMEpochState*            epoch,
    AIVMTransitionResult*      result,
    uint32_t                   att_count,
    uint32_t                   model_count,
    uint32_t                   anchor_count)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // -- mark expired against round timestamp --
    for (uint32_t i = 0; i < att_count; ++i) {
        Attestation& a = attestations[i];
        if (a.occupied == 0u) continue;
        if (a.expiry_ns != 0u && a.expiry_ns <= desc->timestamp_ns) {
            a.status |= kAttStatusExpired;
        }
    }

    // -- attestation_root + counts --
    uint8_t acc[32]; for (uint32_t k = 0; k < 32u; ++k) acc[k] = 0u;
    uint32_t active = 0u, expired = 0u;
    for (uint32_t i = 0; i < att_count; ++i) {
        Attestation& a = attestations[i];
        if (a.occupied == 0u) continue;
        bool exp = (a.expiry_ns != 0u && a.expiry_ns <= desc->timestamp_ns)
                || (a.status & kAttStatusExpired) != 0u;
        bool ver = (a.status & kAttStatusVerified) != 0u;
        if (exp) ++expired;
        else if (ver) ++active;

        uint8_t leaf[32 + 32 + 48 + 8 + 4 + 4 + 4 + 4 + 4 + 4];
        uint32_t o = 0;
        for (uint32_t k = 0; k < 32u; ++k) leaf[o + k] = a.tee_quote_digest[k]; o += 32;
        for (uint32_t k = 0; k < 32u; ++k) leaf[o + k] = a.measurement[k];      o += 32;
        for (uint32_t k = 0; k < 48u; ++k) leaf[o + k] = a.attesting_key[k];    o += 48;
        absorb_u64(leaf, o, a.expiry_ns);       o += 8;
        absorb_u32(leaf, o, a.kind);            o += 4;
        absorb_u32(leaf, o, a.evidence_offset); o += 4;
        absorb_u32(leaf, o, a.evidence_len);    o += 4;
        absorb_u32(leaf, o, a.status);          o += 4;
        absorb_u32(leaf, o, exp ? 1u : 0u);     o += 4;
        absorb_u32(leaf, o, i);                 o += 4;

        uint8_t leaf_hash[32];
        keccak256(leaf, o, leaf_hash);
        uint8_t buf[64];
        for (uint32_t k = 0; k < 32u; ++k) buf[k]      = acc[k];
        for (uint32_t k = 0; k < 32u; ++k) buf[32 + k] = leaf_hash[k];
        keccak256(buf, 64, acc);
    }
    for (uint32_t k = 0; k < 32u; ++k) epoch->attestation_root[k] = acc[k];

    // -- model_registry_root --
    for (uint32_t k = 0; k < 32u; ++k) acc[k] = 0u;
    uint32_t mcount = 0u;
    for (uint32_t i = 0; i < model_count; ++i) {
        ModelRegistryEntry& m = models[i];
        if (m.occupied == 0u) continue;
        ++mcount;

        uint8_t leaf[32 + 32 + 32 + 20 + 8 + 8 + 4 + 4];
        uint32_t o = 0;
        for (uint32_t k = 0; k < 32u; ++k) leaf[o + k] = m.model_root[k];   o += 32;
        for (uint32_t k = 0; k < 32u; ++k) leaf[o + k] = m.weight_hash[k];  o += 32;
        for (uint32_t k = 0; k < 32u; ++k) leaf[o + k] = m.license_root[k]; o += 32;
        for (uint32_t k = 0; k < 20u; ++k) leaf[o + k] = m.owner_addr[k];   o += 20;
        absorb_u64(leaf, o, m.version);         o += 8;
        absorb_u64(leaf, o, m.parameter_count); o += 8;
        absorb_u32(leaf, o, m.modality);        o += 4;
        absorb_u32(leaf, o, i);                 o += 4;

        uint8_t leaf_hash[32];
        keccak256(leaf, o, leaf_hash);
        uint8_t buf[64];
        for (uint32_t k = 0; k < 32u; ++k) buf[k]      = acc[k];
        for (uint32_t k = 0; k < 32u; ++k) buf[32 + k] = leaf_hash[k];
        keccak256(buf, 64, acc);
    }
    for (uint32_t k = 0; k < 32u; ++k) epoch->model_registry_root[k] = acc[k];

    // -- audit_root --
    for (uint32_t k = 0; k < 32u; ++k) acc[k] = 0u;
    uint32_t acount = 0u;
    for (uint32_t i = 0; i < anchor_count; ++i) {
        AuditAnchor& a = anchors[i];
        if (a.occupied == 0u) continue;
        ++acount;

        uint8_t leaf[32 + 32 + 32 + 8 + 8 + 4];
        uint32_t o = 0;
        for (uint32_t k = 0; k < 32u; ++k) leaf[o + k] = a.commit_root[k]; o += 32;
        for (uint32_t k = 0; k < 32u; ++k) leaf[o + k] = a.parent_root[k]; o += 32;
        for (uint32_t k = 0; k < 32u; ++k) leaf[o + k] = a.validator_set_root_at_commit[k]; o += 32;
        absorb_u64(leaf, o, a.height);       o += 8;
        absorb_u64(leaf, o, a.timestamp_ns); o += 8;
        absorb_u32(leaf, o, i);              o += 4;

        uint8_t leaf_hash[32];
        keccak256(leaf, o, leaf_hash);
        uint8_t buf[64];
        for (uint32_t k = 0; k < 32u; ++k) buf[k]      = acc[k];
        for (uint32_t k = 0; k < 32u; ++k) buf[32 + k] = leaf_hash[k];
        keccak256(buf, 64, acc);
    }
    for (uint32_t k = 0; k < 32u; ++k) epoch->audit_root[k] = acc[k];

    // -- epoch metadata --
    epoch->active_model_count        = mcount;
    epoch->expired_attestation_count = expired;
    epoch->total_active_attestations = active;
    uint64_t target_epoch = (desc->closing_flag != 0u) ? desc->epoch + 1u : desc->epoch;
    if (desc->closing_flag != 0u) {
        epoch->current_epoch = target_epoch;
    }

    // -- composed aivm_state_root --
    uint8_t composed[32 + 32 + 32 + 32 + 8 + 4 + 4 + 4];
    uint32_t o = 0;
    for (uint32_t k = 0; k < 32u; ++k) composed[o + k] = desc->parent_aivm_root[k]; o += 32;
    for (uint32_t k = 0; k < 32u; ++k) composed[o + k] = epoch->attestation_root[k]; o += 32;
    for (uint32_t k = 0; k < 32u; ++k) composed[o + k] = epoch->model_registry_root[k]; o += 32;
    for (uint32_t k = 0; k < 32u; ++k) composed[o + k] = epoch->audit_root[k];       o += 32;
    absorb_u64(composed, o, epoch->current_epoch); o += 8;
    absorb_u32(composed, o, active);               o += 4;
    absorb_u32(composed, o, mcount);               o += 4;
    absorb_u32(composed, o, acount);               o += 4;

    uint8_t state_root_local[32];
    keccak256(composed, o, state_root_local);
    for (uint32_t k = 0; k < 32u; ++k) epoch->aivm_state_root[k] = state_root_local[k];

    // -- write result --
    for (uint32_t k = 0; k < 32u; ++k) result->attestation_root[k]    = epoch->attestation_root[k];
    for (uint32_t k = 0; k < 32u; ++k) result->model_registry_root[k] = epoch->model_registry_root[k];
    for (uint32_t k = 0; k < 32u; ++k) result->audit_root[k]          = epoch->audit_root[k];
    for (uint32_t k = 0; k < 32u; ++k) result->aivm_state_root[k]     = epoch->aivm_state_root[k];
    result->active_attestations  = active;
    result->expired_attestations = expired;
    result->model_count          = mcount;
    result->anchor_count         = acount;
    result->total_models         = (uint64_t)mcount;
    result->total_anchors        = (uint64_t)acount;
    result->epoch                = epoch->current_epoch;
    result->status               = 1u;
}

}  // namespace aivm::cuda
