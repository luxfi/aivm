// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// aivm_cuda_launchers.cu — host-side <<<1,1>>> launchers for the four AIVM
// CUDA kernel entry points. Kept in its own translation unit so the host
// driver (aivm_gpu_engine_cuda.cpp) stays pure C++ and can be compiled by
// the host C++ compiler without nvcc.

#include "aivm_kernels_common.cuh"

#include "lux/aivm/aivm_gpu_layout.hpp"

namespace aivm::gpu {

namespace cuda_decl {
extern "C" __global__ void aivm_attestation_apply(
    const ::aivm::cuda::AIVMRoundDescriptor*, const ::aivm::cuda::AttestationOp*,
    ::aivm::cuda::Attestation*, uint32_t*, uint32_t);
extern "C" __global__ void aivm_provenance_apply(
    const ::aivm::cuda::AIVMRoundDescriptor*, const ::aivm::cuda::ModelOp*,
    ::aivm::cuda::ModelRegistryEntry*, uint32_t*, uint32_t);
extern "C" __global__ void aivm_anchor_apply(
    const ::aivm::cuda::AIVMRoundDescriptor*, const ::aivm::cuda::AnchorOp*,
    ::aivm::cuda::AuditAnchor*, uint32_t*, uint32_t);
extern "C" __global__ void aivm_epoch_transition(
    const ::aivm::cuda::AIVMRoundDescriptor*, ::aivm::cuda::Attestation*,
    ::aivm::cuda::ModelRegistryEntry*, ::aivm::cuda::AuditAnchor*,
    ::aivm::cuda::AIVMEpochState*, ::aivm::cuda::AIVMTransitionResult*,
    uint32_t, uint32_t, uint32_t);
}  // namespace cuda_decl

void launch_aivm_attestation_apply(
    const AIVMRoundDescriptor* desc, const AttestationOp* ops,
    Attestation* attestations, uint32_t* applied_out, uint32_t att_count)
{
    cuda_decl::aivm_attestation_apply<<<1, 1>>>(
        reinterpret_cast<const ::aivm::cuda::AIVMRoundDescriptor*>(desc),
        reinterpret_cast<const ::aivm::cuda::AttestationOp*>(ops),
        reinterpret_cast<::aivm::cuda::Attestation*>(attestations),
        applied_out, att_count);
}

void launch_aivm_provenance_apply(
    const AIVMRoundDescriptor* desc, const ModelOp* ops,
    ModelRegistryEntry* models, uint32_t* applied_out, uint32_t model_count)
{
    cuda_decl::aivm_provenance_apply<<<1, 1>>>(
        reinterpret_cast<const ::aivm::cuda::AIVMRoundDescriptor*>(desc),
        reinterpret_cast<const ::aivm::cuda::ModelOp*>(ops),
        reinterpret_cast<::aivm::cuda::ModelRegistryEntry*>(models),
        applied_out, model_count);
}

void launch_aivm_anchor_apply(
    const AIVMRoundDescriptor* desc, const AnchorOp* ops,
    AuditAnchor* anchors, uint32_t* applied_out, uint32_t anchor_count)
{
    cuda_decl::aivm_anchor_apply<<<1, 1>>>(
        reinterpret_cast<const ::aivm::cuda::AIVMRoundDescriptor*>(desc),
        reinterpret_cast<const ::aivm::cuda::AnchorOp*>(ops),
        reinterpret_cast<::aivm::cuda::AuditAnchor*>(anchors),
        applied_out, anchor_count);
}

void launch_aivm_epoch_transition(
    const AIVMRoundDescriptor* desc, Attestation* attestations,
    ModelRegistryEntry* models, AuditAnchor* anchors,
    AIVMEpochState* epoch, AIVMTransitionResult* result,
    uint32_t att_count, uint32_t model_count, uint32_t anchor_count)
{
    cuda_decl::aivm_epoch_transition<<<1, 1>>>(
        reinterpret_cast<const ::aivm::cuda::AIVMRoundDescriptor*>(desc),
        reinterpret_cast<::aivm::cuda::Attestation*>(attestations),
        reinterpret_cast<::aivm::cuda::ModelRegistryEntry*>(models),
        reinterpret_cast<::aivm::cuda::AuditAnchor*>(anchors),
        reinterpret_cast<::aivm::cuda::AIVMEpochState*>(epoch),
        reinterpret_cast<::aivm::cuda::AIVMTransitionResult*>(result),
        att_count, model_count, anchor_count);
}

}  // namespace aivm::gpu
