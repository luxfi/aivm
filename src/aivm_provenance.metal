// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// aivm_provenance.metal — ProvenanceApply kernel.
//
// Walks model_ops in canonical order. Supports four ops:
//   * Register      — first registration of a model (model_root must be new)
//   * UpdateWeights — rotate weight_hash, bump version
//   * UpdateLicense — rotate license_root
//   * Transfer      — change owner_addr

#include "aivm_kernels_common.h.metal"

kernel void aivm_provenance_apply(
    device const AIVMRoundDescriptor* desc        [[buffer(0)]],
    device const ModelOp*             ops         [[buffer(1)]],
    device ModelRegistryEntry*        models      [[buffer(2)]],
    device atomic_uint*               applied_out [[buffer(3)]],
    constant uint&                    model_count [[buffer(4)]],
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

        if (digest_zero_thr(mroot_thr)) continue;
        if (digest_zero_thr(whash_thr)) continue;

        if (op.kind == kModelOpRegister) {
            uint idx = model_locate(models, model_count, mroot_thr, true);
            if (idx == 0xFFFFFFFFu) continue;
            device ModelRegistryEntry& s = models[idx];
            for (uint k = 0; k < 32u; ++k) s.weight_hash[k]  = op.weight_hash[k];
            for (uint k = 0; k < 32u; ++k) s.license_root[k] = op.license_root[k];
            for (uint k = 0; k < 20u; ++k) s.owner_addr[k]   = op.owner_addr[k];
            s.parameter_count = op.parameter_count;
            s.modality        = op.modality;
            s.version         = 1u;
            ++applied;
        } else if (op.kind == kModelOpUpdateWeights) {
            uint idx = model_locate(models, model_count, mroot_thr, false);
            if (idx == 0xFFFFFFFFu) continue;
            device ModelRegistryEntry& s = models[idx];
            for (uint k = 0; k < 32u; ++k) s.weight_hash[k] = op.weight_hash[k];
            ulong v = s.version + 1u;
            if (v < s.version) v = 0xFFFFFFFFFFFFFFFFUL; // saturating
            s.version = v;
            if (op.parameter_count != 0u) s.parameter_count = op.parameter_count;
            ++applied;
        } else if (op.kind == kModelOpUpdateLicense) {
            uint idx = model_locate(models, model_count, mroot_thr, false);
            if (idx == 0xFFFFFFFFu) continue;
            device ModelRegistryEntry& s = models[idx];
            for (uint k = 0; k < 32u; ++k) s.license_root[k] = op.license_root[k];
            ++applied;
        } else if (op.kind == kModelOpTransfer) {
            uint idx = model_locate(models, model_count, mroot_thr, false);
            if (idx == 0xFFFFFFFFu) continue;
            device ModelRegistryEntry& s = models[idx];
            for (uint k = 0; k < 20u; ++k) s.owner_addr[k] = op.owner_addr[k];
            ++applied;
        }
    }
    atomic_store_explicit(applied_out, applied, memory_order_relaxed);
}
