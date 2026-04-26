// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// aivm_anchor.metal — AnchorApply kernel.
//
// Append-only ring buffer of audit anchors. Each new anchor must:
//   * have non-zero commit_root
//   * if there is a previous anchor, its parent_root must equal that
//     previous anchor's commit_root (chain integrity)
//   * have strictly monotonic height vs the previous anchor

#include "aivm_kernels_common.h.metal"

kernel void aivm_anchor_apply(
    device const AIVMRoundDescriptor* desc         [[buffer(0)]],
    device const AnchorOp*            ops          [[buffer(1)]],
    device AuditAnchor*               anchors      [[buffer(2)]],
    device atomic_uint*               applied_out  [[buffer(3)]],
    constant uint&                    anchor_count [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0u) return;

    // Locate cursor — first non-occupied slot.
    uint cursor = 0u;
    while (cursor < anchor_count && anchors[cursor].occupied != 0u) ++cursor;

    uint applied = 0u;
    uint count = desc->anchor_op_count;
    for (uint i = 0; i < count; ++i) {
        const device AnchorOp& op = ops[i];

        if (digest_zero_dev(op.commit_root)) continue;
        if (cursor >= anchor_count) break;

        if (cursor > 0u) {
            device AuditAnchor& prev = anchors[cursor - 1u];
            // parent_root must equal prev.commit_root
            bool ok = true;
            for (uint k = 0; k < 32u; ++k) {
                if (op.parent_root[k] != prev.commit_root[k]) { ok = false; break; }
            }
            if (!ok) continue;
            if (op.height <= prev.height) continue;
        }

        device AuditAnchor& dst = anchors[cursor];
        for (uint k = 0; k < 32u; ++k) dst.commit_root[k] = op.commit_root[k];
        for (uint k = 0; k < 32u; ++k) dst.parent_root[k] = op.parent_root[k];
        for (uint k = 0; k < 32u; ++k) dst.validator_set_root_at_commit[k]
            = op.validator_set_root_at_commit[k];
        dst.height       = op.height;
        dst.timestamp_ns = op.timestamp_ns;
        dst.occupied     = 1u;
        ++cursor;
        ++applied;
    }
    atomic_store_explicit(applied_out, applied, memory_order_relaxed);
}
