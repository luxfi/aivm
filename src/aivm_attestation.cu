// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// aivm_attestation.cu — CUDA peer of aivm_attestation.metal.

#include "aivm_kernels_common.cuh"

namespace aivm::cuda {

extern "C" __global__ void aivm_attestation_apply(
    const AIVMRoundDescriptor* desc,
    const AttestationOp*       ops,
    Attestation*               attestations,
    uint32_t*                  applied_out,
    uint32_t                   att_count)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    uint32_t applied = 0;
    uint32_t count = desc->attestation_op_count;
    for (uint32_t i = 0; i < count; ++i) {
        const AttestationOp& op = ops[i];
        if (key48_zero(op.attesting_key)) continue;
        if (digest_zero32(op.tee_quote_digest)) continue;

        uint32_t idx = attestation_locate(attestations, att_count,
                                          op.tee_quote_digest, true);
        if (idx == 0xFFFFFFFFu) continue;

        Attestation& s = attestations[idx];
        for (uint32_t k = 0; k < 32u; ++k) s.measurement[k]   = op.measurement[k];
        for (uint32_t k = 0; k < 48u; ++k) s.attesting_key[k] = op.attesting_key[k];
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
    *applied_out = applied;
}

}  // namespace aivm::cuda
