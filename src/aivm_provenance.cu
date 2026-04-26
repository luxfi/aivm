// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// aivm_provenance.cu — CUDA peer of aivm_provenance.metal.

#include "aivm_kernels_common.cuh"

namespace aivm::cuda {

extern "C" __global__ void aivm_provenance_apply(
    const AIVMRoundDescriptor* desc,
    const ModelOp*             ops,
    ModelRegistryEntry*        models,
    uint32_t*                  applied_out,
    uint32_t                   model_count)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    uint32_t applied = 0;
    uint32_t count = desc->model_op_count;
    for (uint32_t i = 0; i < count; ++i) {
        const ModelOp& op = ops[i];
        if (digest_zero32(op.model_root)) continue;
        if (digest_zero32(op.weight_hash)) continue;

        if (op.kind == kModelOpRegister) {
            uint32_t idx = model_locate(models, model_count, op.model_root, true);
            if (idx == 0xFFFFFFFFu) continue;
            ModelRegistryEntry& s = models[idx];
            for (uint32_t k = 0; k < 32u; ++k) s.weight_hash[k]  = op.weight_hash[k];
            for (uint32_t k = 0; k < 32u; ++k) s.license_root[k] = op.license_root[k];
            for (uint32_t k = 0; k < 20u; ++k) s.owner_addr[k]   = op.owner_addr[k];
            s.parameter_count = op.parameter_count;
            s.modality        = op.modality;
            s.version         = 1u;
            ++applied;
        } else if (op.kind == kModelOpUpdateWeights) {
            uint32_t idx = model_locate(models, model_count, op.model_root, false);
            if (idx == 0xFFFFFFFFu) continue;
            ModelRegistryEntry& s = models[idx];
            for (uint32_t k = 0; k < 32u; ++k) s.weight_hash[k] = op.weight_hash[k];
            uint64_t v = s.version + 1u;
            if (v < s.version) v = 0xFFFFFFFFFFFFFFFFULL;
            s.version = v;
            if (op.parameter_count != 0u) s.parameter_count = op.parameter_count;
            ++applied;
        } else if (op.kind == kModelOpUpdateLicense) {
            uint32_t idx = model_locate(models, model_count, op.model_root, false);
            if (idx == 0xFFFFFFFFu) continue;
            ModelRegistryEntry& s = models[idx];
            for (uint32_t k = 0; k < 32u; ++k) s.license_root[k] = op.license_root[k];
            ++applied;
        } else if (op.kind == kModelOpTransfer) {
            uint32_t idx = model_locate(models, model_count, op.model_root, false);
            if (idx == 0xFFFFFFFFu) continue;
            ModelRegistryEntry& s = models[idx];
            for (uint32_t k = 0; k < 20u; ++k) s.owner_addr[k] = op.owner_addr[k];
            ++applied;
        }
    }
    *applied_out = applied;
}

}  // namespace aivm::cuda
