// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// aivm_attestation.metal — AttestationApply kernel.
//
// Walks attestation_ops in canonical order. For each op:
//   * reject if attesting_key is all-zero (host-side verifier zeros the key
//     when the BLS / Ringtail / Groth16 / SLH-DSA quote signature fails)
//   * reject if tee_quote_digest is zero
//   * insert/update the attestation entry; mark verified
//   * mark expired if expiry_ns is set and <= timestamp_ns
//
// Single-threadgroup (1x1x1) execution preserves byte-for-byte determinism
// with the CPU reference.

#include "aivm_kernels_common.h.metal"

kernel void aivm_attestation_apply(
    device const AIVMRoundDescriptor* desc        [[buffer(0)]],
    device const AttestationOp*       ops         [[buffer(1)]],
    device Attestation*               attestations[[buffer(2)]],
    device atomic_uint*               applied_out [[buffer(3)]],
    constant uint&                    att_count   [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0u) return;

    uint applied = 0u;
    uint count = desc->attestation_op_count;
    for (uint i = 0; i < count; ++i) {
        const device AttestationOp& op = ops[i];

        // Stage op fields into thread memory so we can use thread-pointer
        // helpers below (Metal doesn't auto-coerce device->thread pointers).
        uchar digest_thr[32];
        for (uint k = 0; k < 32u; ++k) digest_thr[k] = op.tee_quote_digest[k];
        uchar key_thr[48];
        for (uint k = 0; k < 48u; ++k) key_thr[k] = op.attesting_key[k];

        if (key48_zero_thr(key_thr)) continue;
        if (digest_zero_thr(digest_thr)) continue;

        uint idx = attestation_locate(attestations, att_count, digest_thr, true);
        if (idx == 0xFFFFFFFFu) continue;

        device Attestation& s = attestations[idx];
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
        ++applied;
    }
    atomic_store_explicit(applied_out, applied, memory_order_relaxed);
}
