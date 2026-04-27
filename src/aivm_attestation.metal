// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// aivm_attestation.metal — AttestationApply kernels (v0.59 parallel).
//
// Two-phase parallel design that preserves CPU byte-equivalence:
//   1. aivm_attestation_locate   (1 thread): walks ops in canonical order,
//      claims slots via the same locate function the CPU oracle uses.
//      Records (op_slot[i], slot_winner_op[s]) where slot_winner_op[s] is
//      the highest op index that claimed slot s. Marks slots occupied.
//      No field writes for the per-op state — that happens in pass 2.
//   2. aivm_attestation_writeback (one thread per slot): each slot's thread
//      reads slot_winner_op[s]. If valid, copies op[winner].fields into
//      the slot. This is the parallel work — 144-byte memcpy per slot done
//      by kAttSlots threads in parallel.
//
// Invariants preserved:
//   - Slot assignment matches CPU oracle byte-for-byte (pass 1 walks ops in
//     canonical order using the same locate function).
//   - Per-slot field state matches the canonical last writer (pass 2 copies
//     from op[slot_winner_op[s]]).

#include "aivm_kernels_common.h.metal"

constant uint kSentinelSlot = 0xFFFFFFFFu;

kernel void aivm_attestation_locate(
    device const AIVMRoundDescriptor* desc           [[buffer(0)]],
    device const AttestationOp*       ops            [[buffer(1)]],
    device Attestation*               attestations   [[buffer(2)]],
    device atomic_uint*               applied_out    [[buffer(3)]],
    constant uint&                    att_count      [[buffer(4)]],
    device uint*                      op_slot        [[buffer(5)]],
    device uint*                      slot_winner_op [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0u) return;

    uint applied = 0u;
    uint count = desc->attestation_op_count;
    for (uint i = 0; i < count; ++i) {
        const device AttestationOp& op = ops[i];

        uchar digest_thr[32];
        for (uint k = 0; k < 32u; ++k) digest_thr[k] = op.tee_quote_digest[k];
        uchar key_thr[48];
        for (uint k = 0; k < 48u; ++k) key_thr[k] = op.attesting_key[k];

        if (key48_zero_thr(key_thr)) { op_slot[i] = kSentinelSlot; continue; }
        if (digest_zero_thr(digest_thr)) { op_slot[i] = kSentinelSlot; continue; }

        uint idx = attestation_locate(attestations, att_count, digest_thr, true);
        if (idx == 0xFFFFFFFFu) { op_slot[i] = kSentinelSlot; continue; }

        op_slot[i] = idx;
        // Last-writer-wins: record highest op idx per slot. Sentinel start
        // value is 0xFFFFFFFF; any real index beats it.
        uint cur = slot_winner_op[idx];
        if (cur == kSentinelSlot || i > cur) slot_winner_op[idx] = i;
        ++applied;
    }
    atomic_store_explicit(applied_out, applied, memory_order_relaxed);
}

kernel void aivm_attestation_writeback(
    device const AIVMRoundDescriptor* desc           [[buffer(0)]],
    device const AttestationOp*       ops            [[buffer(1)]],
    device Attestation*               attestations   [[buffer(2)]],
    constant uint&                    att_count      [[buffer(4)]],
    device const uint*                slot_winner_op [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= att_count) return;
    uint winner = slot_winner_op[tid];
    if (winner == kSentinelSlot) return;

    const device AttestationOp& op = ops[winner];
    device Attestation& s = attestations[tid];
    for (uint k = 0; k < 32u; ++k) s.measurement[k]   = op.measurement[k];
    for (uint k = 0; k < 48u; ++k) s.attesting_key[k] = op.attesting_key[k];
    s.expiry_ns       = op.expiry_ns;
    s.kind            = op.kind;
    s.evidence_offset = op.evidence_offset;
    s.evidence_len    = op.evidence_len;
    s.status          = kAttStatusVerified;
    if (op.expiry_ns != 0u && op.expiry_ns <= desc->timestamp_ns) {
        s.status |= kAttStatusExpired;
    }
}
