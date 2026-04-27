// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// aivm_provenance.metal — ProvenanceApply kernels (v0.59 parallel).
//
// ModelOp semantics (Register / UpdateWeights / UpdateLicense / Transfer)
// are not last-writer-wins per model — UpdateWeights bumps the version
// counter cumulatively. Pass 1 therefore must replay all ops in canonical
// order and update the arena in place. Pass 2 is a no-op writeback —
// retained for API symmetry with attestation/anchor and to keep the
// dispatch shape consistent.

#include "aivm_kernels_common.h.metal"

constant uint kSentinelSlotProv = 0xFFFFFFFFu;

kernel void aivm_provenance_locate(
    device const AIVMRoundDescriptor* desc        [[buffer(0)]],
    device const ModelOp*             ops         [[buffer(1)]],
    device ModelRegistryEntry*        models      [[buffer(2)]],
    device atomic_uint*               applied_out [[buffer(3)]],
    constant uint&                    model_count [[buffer(4)]],
    device uint*                      op_slot     [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0u) return;

    uint applied = 0u;
    uint count = desc->model_op_count;
    for (uint i = 0; i < count; ++i) {
        const device ModelOp& op = ops[i];

        uchar mroot_thr[32];
        for (uint k = 0; k < 32u; ++k) mroot_thr[k] = op.model_root[k];
        uchar whash_thr[32];
        for (uint k = 0; k < 32u; ++k) whash_thr[k] = op.weight_hash[k];

        if (digest_zero_thr(mroot_thr)) { op_slot[i] = kSentinelSlotProv; continue; }
        if (digest_zero_thr(whash_thr)) { op_slot[i] = kSentinelSlotProv; continue; }

        if (op.kind == kModelOpRegister) {
            uint idx = model_locate(models, model_count, mroot_thr, true);
            if (idx == 0xFFFFFFFFu) { op_slot[i] = kSentinelSlotProv; continue; }
            device ModelRegistryEntry& s = models[idx];
            for (uint k = 0; k < 32u; ++k) s.weight_hash[k]  = op.weight_hash[k];
            for (uint k = 0; k < 32u; ++k) s.license_root[k] = op.license_root[k];
            for (uint k = 0; k < 20u; ++k) s.owner_addr[k]   = op.owner_addr[k];
            s.parameter_count = op.parameter_count;
            s.modality        = op.modality;
            s.version         = 1u;
            op_slot[i] = idx;
            ++applied;
        } else if (op.kind == kModelOpUpdateWeights) {
            uint idx = model_locate(models, model_count, mroot_thr, false);
            if (idx == 0xFFFFFFFFu) { op_slot[i] = kSentinelSlotProv; continue; }
            device ModelRegistryEntry& s = models[idx];
            for (uint k = 0; k < 32u; ++k) s.weight_hash[k] = op.weight_hash[k];
            ulong v = s.version + 1u;
            if (v < s.version) v = 0xFFFFFFFFFFFFFFFFUL;
            s.version = v;
            if (op.parameter_count != 0u) s.parameter_count = op.parameter_count;
            op_slot[i] = idx;
            ++applied;
        } else if (op.kind == kModelOpUpdateLicense) {
            uint idx = model_locate(models, model_count, mroot_thr, false);
            if (idx == 0xFFFFFFFFu) { op_slot[i] = kSentinelSlotProv; continue; }
            device ModelRegistryEntry& s = models[idx];
            for (uint k = 0; k < 32u; ++k) s.license_root[k] = op.license_root[k];
            op_slot[i] = idx;
            ++applied;
        } else if (op.kind == kModelOpTransfer) {
            uint idx = model_locate(models, model_count, mroot_thr, false);
            if (idx == 0xFFFFFFFFu) { op_slot[i] = kSentinelSlotProv; continue; }
            device ModelRegistryEntry& s = models[idx];
            for (uint k = 0; k < 20u; ++k) s.owner_addr[k] = op.owner_addr[k];
            op_slot[i] = idx;
            ++applied;
        } else {
            op_slot[i] = kSentinelSlotProv;
        }
    }
    atomic_store_explicit(applied_out, applied, memory_order_relaxed);
}
