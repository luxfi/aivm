// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// aivm_anchor.cu — CUDA peer of aivm_anchor.metal.

#include "aivm_kernels_common.cuh"

namespace aivm::cuda {

extern "C" __global__ void aivm_anchor_apply(
    const AIVMRoundDescriptor* desc,
    const AnchorOp*            ops,
    AuditAnchor*               anchors,
    uint32_t*                  applied_out,
    uint32_t                   anchor_count)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    uint32_t cursor = 0u;
    while (cursor < anchor_count && anchors[cursor].occupied != 0u) ++cursor;

    uint32_t applied = 0u;
    uint32_t count = desc->anchor_op_count;
    for (uint32_t i = 0; i < count; ++i) {
        const AnchorOp& op = ops[i];
        if (digest_zero32(op.commit_root)) continue;
        if (cursor >= anchor_count) break;

        if (cursor > 0u) {
            AuditAnchor& prev = anchors[cursor - 1u];
            bool ok = true;
            for (uint32_t k = 0; k < 32u; ++k) {
                if (op.parent_root[k] != prev.commit_root[k]) { ok = false; break; }
            }
            if (!ok) continue;
            if (op.height <= prev.height) continue;
        }

        AuditAnchor& dst = anchors[cursor];
        for (uint32_t k = 0; k < 32u; ++k) dst.commit_root[k] = op.commit_root[k];
        for (uint32_t k = 0; k < 32u; ++k) dst.parent_root[k] = op.parent_root[k];
        for (uint32_t k = 0; k < 32u; ++k) dst.validator_set_root_at_commit[k]
            = op.validator_set_root_at_commit[k];
        dst.height       = op.height;
        dst.timestamp_ns = op.timestamp_ns;
        dst.occupied     = 1u;
        ++cursor;
        ++applied;
    }
    *applied_out = applied;
}

}  // namespace aivm::cuda
