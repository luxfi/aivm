// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// aivm_transition.metal — EpochTransitionAIVM kernel.
//
// Marks expired attestations against the round timestamp, computes the
// three component roots, then composes aivm_state_root:
//
//   attestation_root     =  fold-keccak over occupied Attestation leaves
//   model_registry_root  =  fold-keccak over occupied ModelRegistryEntry leaves
//   audit_root           =  fold-keccak over occupied AuditAnchor leaves
//   aivm_state_root      =  keccak(parent || att_root || model_root ||
//                                  audit_root || epoch || active ||
//                                  model_count || anchor_count)
//
// aivm_state_root is the linkage that completes LP-137 GPU-native
// compliance for the A-Chain.

#include "aivm_kernels_common.h.metal"

kernel void aivm_epoch_transition(
    device const AIVMRoundDescriptor* desc           [[buffer(0)]],
    device Attestation*               attestations   [[buffer(1)]],
    device ModelRegistryEntry*        models         [[buffer(2)]],
    device AuditAnchor*               anchors        [[buffer(3)]],
    device AIVMEpochState*            epoch          [[buffer(4)]],
    device AIVMTransitionResult*      result         [[buffer(5)]],
    constant uint&                    att_count      [[buffer(6)]],
    constant uint&                    model_count    [[buffer(7)]],
    constant uint&                    anchor_count   [[buffer(8)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0u) return;

    // -- mark expired against round timestamp --
    for (uint i = 0; i < att_count; ++i) {
        device Attestation& a = attestations[i];
        if (a.occupied == 0u) continue;
        if (a.expiry_ns != 0u && a.expiry_ns <= desc->timestamp_ns) {
            a.status |= kAttStatusExpired;
        }
    }

    // -- attestation_root + counts --
    uchar acc[32]; for (uint k = 0; k < 32u; ++k) acc[k] = 0u;
    uint active = 0u, expired = 0u;
    for (uint i = 0; i < att_count; ++i) {
        device Attestation& a = attestations[i];
        if (a.occupied == 0u) continue;
        bool exp = (a.expiry_ns != 0u && a.expiry_ns <= desc->timestamp_ns)
                || (a.status & kAttStatusExpired) != 0u;
        bool ver = (a.status & kAttStatusVerified) != 0u;
        if (exp) ++expired;
        else if (ver) ++active;

        uchar leaf[32 + 32 + 48 + 8 + 4 + 4 + 4 + 4 + 4 + 4];
        uint o = 0;
        for (uint k = 0; k < 32u; ++k) leaf[o + k] = a.tee_quote_digest[k]; o += 32;
        for (uint k = 0; k < 32u; ++k) leaf[o + k] = a.measurement[k];      o += 32;
        for (uint k = 0; k < 48u; ++k) leaf[o + k] = a.attesting_key[k];    o += 48;
        absorb_u64(leaf, o, a.expiry_ns);       o += 8;
        absorb_u32(leaf, o, a.kind);            o += 4;
        absorb_u32(leaf, o, a.evidence_offset); o += 4;
        absorb_u32(leaf, o, a.evidence_len);    o += 4;
        absorb_u32(leaf, o, a.status);          o += 4;
        absorb_u32(leaf, o, exp ? 1u : 0u);     o += 4;
        absorb_u32(leaf, o, i);                 o += 4;

        uchar leaf_hash[32];
        keccak256(leaf, o, leaf_hash);
        uchar buf[64];
        for (uint k = 0; k < 32u; ++k) buf[k]      = acc[k];
        for (uint k = 0; k < 32u; ++k) buf[32 + k] = leaf_hash[k];
        keccak256(buf, 64, acc);
    }
    for (uint k = 0; k < 32u; ++k) epoch->attestation_root[k] = acc[k];

    // -- model_registry_root --
    for (uint k = 0; k < 32u; ++k) acc[k] = 0u;
    uint mcount = 0u;
    for (uint i = 0; i < model_count; ++i) {
        device ModelRegistryEntry& m = models[i];
        if (m.occupied == 0u) continue;
        ++mcount;

        uchar leaf[32 + 32 + 32 + 20 + 8 + 8 + 4 + 4];
        uint o = 0;
        for (uint k = 0; k < 32u; ++k) leaf[o + k] = m.model_root[k];   o += 32;
        for (uint k = 0; k < 32u; ++k) leaf[o + k] = m.weight_hash[k];  o += 32;
        for (uint k = 0; k < 32u; ++k) leaf[o + k] = m.license_root[k]; o += 32;
        for (uint k = 0; k < 20u; ++k) leaf[o + k] = m.owner_addr[k];   o += 20;
        absorb_u64(leaf, o, m.version);         o += 8;
        absorb_u64(leaf, o, m.parameter_count); o += 8;
        absorb_u32(leaf, o, m.modality);        o += 4;
        absorb_u32(leaf, o, i);                 o += 4;

        uchar leaf_hash[32];
        keccak256(leaf, o, leaf_hash);
        uchar buf[64];
        for (uint k = 0; k < 32u; ++k) buf[k]      = acc[k];
        for (uint k = 0; k < 32u; ++k) buf[32 + k] = leaf_hash[k];
        keccak256(buf, 64, acc);
    }
    for (uint k = 0; k < 32u; ++k) epoch->model_registry_root[k] = acc[k];

    // -- audit_root --
    for (uint k = 0; k < 32u; ++k) acc[k] = 0u;
    uint acount = 0u;
    for (uint i = 0; i < anchor_count; ++i) {
        device AuditAnchor& a = anchors[i];
        if (a.occupied == 0u) continue;
        ++acount;

        uchar leaf[32 + 32 + 32 + 8 + 8 + 4];
        uint o = 0;
        for (uint k = 0; k < 32u; ++k) leaf[o + k] = a.commit_root[k]; o += 32;
        for (uint k = 0; k < 32u; ++k) leaf[o + k] = a.parent_root[k]; o += 32;
        for (uint k = 0; k < 32u; ++k) leaf[o + k] = a.validator_set_root_at_commit[k]; o += 32;
        absorb_u64(leaf, o, a.height);       o += 8;
        absorb_u64(leaf, o, a.timestamp_ns); o += 8;
        absorb_u32(leaf, o, i);              o += 4;

        uchar leaf_hash[32];
        keccak256(leaf, o, leaf_hash);
        uchar buf[64];
        for (uint k = 0; k < 32u; ++k) buf[k]      = acc[k];
        for (uint k = 0; k < 32u; ++k) buf[32 + k] = leaf_hash[k];
        keccak256(buf, 64, acc);
    }
    for (uint k = 0; k < 32u; ++k) epoch->audit_root[k] = acc[k];

    // -- epoch metadata --
    epoch->active_model_count        = mcount;
    epoch->expired_attestation_count = expired;
    epoch->total_active_attestations = active;
    ulong target_epoch = (desc->closing_flag != 0u) ? desc->epoch + 1u : desc->epoch;
    if (desc->closing_flag != 0u) {
        epoch->current_epoch = target_epoch;
    }

    // -- composed aivm_state_root --
    uchar composed[32 + 32 + 32 + 32 + 8 + 4 + 4 + 4];
    uint o = 0;
    for (uint k = 0; k < 32u; ++k) composed[o + k] = desc->parent_aivm_root[k]; o += 32;
    for (uint k = 0; k < 32u; ++k) composed[o + k] = epoch->attestation_root[k]; o += 32;
    for (uint k = 0; k < 32u; ++k) composed[o + k] = epoch->model_registry_root[k]; o += 32;
    for (uint k = 0; k < 32u; ++k) composed[o + k] = epoch->audit_root[k];       o += 32;
    absorb_u64(composed, o, epoch->current_epoch); o += 8;
    absorb_u32(composed, o, active);               o += 4;
    absorb_u32(composed, o, mcount);               o += 4;
    absorb_u32(composed, o, acount);               o += 4;

    uchar state_root_local[32];
    keccak256(composed, o, state_root_local);
    for (uint k = 0; k < 32u; ++k) epoch->aivm_state_root[k] = state_root_local[k];

    // -- write result --
    for (uint k = 0; k < 32u; ++k) result->attestation_root[k]    = epoch->attestation_root[k];
    for (uint k = 0; k < 32u; ++k) result->model_registry_root[k] = epoch->model_registry_root[k];
    for (uint k = 0; k < 32u; ++k) result->audit_root[k]          = epoch->audit_root[k];
    for (uint k = 0; k < 32u; ++k) result->aivm_state_root[k]     = epoch->aivm_state_root[k];
    result->active_attestations  = active;
    result->expired_attestations = expired;
    result->model_count          = mcount;
    result->anchor_count         = acount;
    result->total_models         = (ulong)mcount;
    result->total_anchors        = (ulong)acount;
    result->epoch                = epoch->current_epoch;
    result->status               = 1u;
}
