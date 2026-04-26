// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file aivm_layout_test.cpp
/// AIVM v0.58 — layout invariants for cross-backend determinism.

#include "lux/aivm/aivm_gpu_layout.hpp"
#include "lux/aivm/aivm_cpu_reference.hpp"

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace aivm::gpu;

namespace {

int g_passed = 0;
int g_failed = 0;

#define EXPECT(name, cond)                                                  \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::printf("  FAIL[%s]: %s\n", (name), #cond);                 \
            std::fflush(stdout);                                            \
            ++g_failed;                                                     \
            return;                                                         \
        }                                                                   \
    } while (0)

#define PASS(name)                                                          \
    do {                                                                    \
        std::printf("  ok  : %s\n", (name));                                \
        std::fflush(stdout);                                                \
        ++g_passed;                                                         \
    } while (0)

void test_attestation_layout()
{
    EXPECT("Attestation.size",  sizeof(Attestation) == 144);
    EXPECT("Attestation.align", alignof(Attestation) == 16);
    EXPECT("Att.digest.off",    offsetof(Attestation, tee_quote_digest) == 0);
    EXPECT("Att.measurement.off", offsetof(Attestation, measurement)    == 32);
    EXPECT("Att.attkey.off",    offsetof(Attestation, attesting_key)    == 64);
    EXPECT("Att.expiry.off",    offsetof(Attestation, expiry_ns)        == 112);
    EXPECT("Att.kind.off",      offsetof(Attestation, kind)             == 120);
    EXPECT("Att.evoff.off",     offsetof(Attestation, evidence_offset)  == 124);
    EXPECT("Att.evlen.off",     offsetof(Attestation, evidence_len)     == 128);
    EXPECT("Att.status.off",    offsetof(Attestation, status)           == 132);
    EXPECT("Att.occupied.off",  offsetof(Attestation, occupied)         == 136);
    PASS("Attestation layout");
}

void test_model_layout()
{
    EXPECT("Model.size",  sizeof(ModelRegistryEntry) == 160);
    EXPECT("Model.align", alignof(ModelRegistryEntry) == 16);
    EXPECT("Model.mroot.off",   offsetof(ModelRegistryEntry, model_root)   == 0);
    EXPECT("Model.whash.off",   offsetof(ModelRegistryEntry, weight_hash)  == 32);
    EXPECT("Model.lroot.off",   offsetof(ModelRegistryEntry, license_root) == 64);
    EXPECT("Model.owner.off",   offsetof(ModelRegistryEntry, owner_addr)   == 96);
    EXPECT("Model.version.off", offsetof(ModelRegistryEntry, version)      == 120);
    EXPECT("Model.params.off",  offsetof(ModelRegistryEntry, parameter_count) == 128);
    EXPECT("Model.modality.off",offsetof(ModelRegistryEntry, modality)     == 136);
    EXPECT("Model.occupied.off",offsetof(ModelRegistryEntry, occupied)     == 140);
    PASS("ModelRegistryEntry layout");
}

void test_anchor_layout()
{
    EXPECT("Anchor.size",  sizeof(AuditAnchor) == 128);
    EXPECT("Anchor.align", alignof(AuditAnchor) == 16);
    EXPECT("Anchor.commit.off",  offsetof(AuditAnchor, commit_root) == 0);
    EXPECT("Anchor.parent.off",  offsetof(AuditAnchor, parent_root) == 32);
    EXPECT("Anchor.vroot.off",   offsetof(AuditAnchor, validator_set_root_at_commit) == 64);
    EXPECT("Anchor.height.off",  offsetof(AuditAnchor, height)       == 96);
    EXPECT("Anchor.ts.off",      offsetof(AuditAnchor, timestamp_ns) == 104);
    EXPECT("Anchor.occupied.off",offsetof(AuditAnchor, occupied)     == 112);
    PASS("AuditAnchor layout");
}

void test_epoch_state_layout()
{
    EXPECT("EpochState.size",  sizeof(AIVMEpochState) == 160);
    EXPECT("EpochState.align", alignof(AIVMEpochState) == 16);
    EXPECT("EpochState.curr.off",  offsetof(AIVMEpochState, current_epoch)              == 0);
    EXPECT("EpochState.next.off",  offsetof(AIVMEpochState, next_epoch_height)          == 8);
    EXPECT("EpochState.total.off", offsetof(AIVMEpochState, total_active_attestations)  == 16);
    EXPECT("EpochState.actc.off",  offsetof(AIVMEpochState, active_model_count)         == 24);
    EXPECT("EpochState.exp.off",   offsetof(AIVMEpochState, expired_attestation_count)  == 28);
    EXPECT("EpochState.aroot.off", offsetof(AIVMEpochState, attestation_root)           == 32);
    EXPECT("EpochState.mroot.off", offsetof(AIVMEpochState, model_registry_root)        == 64);
    EXPECT("EpochState.uroot.off", offsetof(AIVMEpochState, audit_root)                 == 96);
    EXPECT("EpochState.sroot.off", offsetof(AIVMEpochState, aivm_state_root)            == 128);
    PASS("AIVMEpochState layout");
}

void test_round_descriptor_layout()
{
    EXPECT("Desc.size",  sizeof(AIVMRoundDescriptor) == 96);
    EXPECT("Desc.align", alignof(AIVMRoundDescriptor) == 16);
    EXPECT("Desc.mode.off",  offsetof(AIVMRoundDescriptor, mode)            == 32);
    EXPECT("Desc.parent.off",offsetof(AIVMRoundDescriptor, parent_aivm_root) == 64);
    PASS("AIVMRoundDescriptor layout");
}

void test_op_layouts()
{
    EXPECT("AttestationOp.size",  sizeof(AttestationOp) == 144);
    EXPECT("AttestationOp.align", alignof(AttestationOp) == 16);
    EXPECT("AttOp.kind.off", offsetof(AttestationOp, kind) == 120);

    EXPECT("ModelOp.size",  sizeof(ModelOp) == 160);
    EXPECT("ModelOp.align", alignof(ModelOp) == 16);
    EXPECT("ModelOp.kind.off", offsetof(ModelOp, kind) == 132);

    EXPECT("AnchorOp.size",  sizeof(AnchorOp) == 128);
    EXPECT("AnchorOp.align", alignof(AnchorOp) == 16);
    EXPECT("AnchorOp.height.off", offsetof(AnchorOp, height) == 96);
    PASS("Op layouts");
}

void test_transition_result_layout()
{
    EXPECT("Result.size",  sizeof(AIVMTransitionResult) == 192);
    EXPECT("Result.align", alignof(AIVMTransitionResult) == 16);
    EXPECT("Result.aivm_state_root.off",
           offsetof(AIVMTransitionResult, aivm_state_root) == 160);
    EXPECT("Result.attestation_root.off",
           offsetof(AIVMTransitionResult, attestation_root) == 64);
    PASS("AIVMTransitionResult layout");
}

AIVMRoundDescriptor make_desc(uint64_t round, uint32_t mode = 4u)
{
    AIVMRoundDescriptor d{};
    d.chain_id = 1u;
    d.round = round;
    d.timestamp_ns = 1700000000000000000ULL;
    d.epoch = 0u;
    d.mode = mode;
    d.closing_flag = 1u;
    return d;
}

AttestationOp make_att(uint64_t key_seed, uint8_t kind = 0u, uint64_t expiry = 0u)
{
    AttestationOp op{};
    for (uint32_t i = 0; i < 32; ++i)
        op.tee_quote_digest[i] = uint8_t((key_seed * 31u + i) & 0xFFu);
    for (uint32_t i = 0; i < 32; ++i)
        op.measurement[i] = uint8_t((key_seed * 17u + i) & 0xFFu);
    for (uint32_t i = 0; i < 48; ++i)
        op.attesting_key[i] = uint8_t((key_seed * 41u + i + 1u) & 0xFFu);
    op.expiry_ns = expiry;
    op.kind = kind;
    op.evidence_offset = 0;
    op.evidence_len = 64u;
    return op;
}

ModelOp make_model_register(uint64_t seed, uint64_t param_count = 1000u)
{
    ModelOp op{};
    for (uint32_t i = 0; i < 32; ++i) op.model_root[i]   = uint8_t((seed * 11u + i + 1u) & 0xFFu);
    for (uint32_t i = 0; i < 32; ++i) op.weight_hash[i]  = uint8_t((seed * 23u + i + 7u) & 0xFFu);
    for (uint32_t i = 0; i < 32; ++i) op.license_root[i] = uint8_t((seed *  7u + i + 13u) & 0xFFu);
    for (uint32_t i = 0; i < 20; ++i) op.owner_addr[i]   = uint8_t((seed *  3u + i + 5u) & 0xFFu);
    op.parameter_count = param_count;
    op.modality = 0u; // text
    op.kind = static_cast<uint32_t>(ModelOpKind::Register);
    return op;
}

AnchorOp make_anchor(uint64_t height, const uint8_t parent[32])
{
    AnchorOp op{};
    for (uint32_t i = 0; i < 32; ++i) op.commit_root[i] = uint8_t((height * 13u + i + 1u) & 0xFFu);
    std::memcpy(op.parent_root, parent, 32);
    for (uint32_t i = 0; i < 32; ++i) op.validator_set_root_at_commit[i]
        = uint8_t((height * 19u + i + 3u) & 0xFFu);
    op.height = height;
    op.timestamp_ns = 1700000000000000000ULL + height * 1000000000ULL;
    return op;
}

void test_cpu_reference_basic()
{
    std::vector<AttestationOp> a_ops;
    for (uint64_t i = 1; i <= 8; ++i)
        a_ops.push_back(make_att(i, uint8_t(i % 4u)));

    std::vector<ModelOp> m_ops;
    for (uint64_t i = 1; i <= 4; ++i) m_ops.push_back(make_model_register(i));

    std::vector<AnchorOp> n_ops;
    uint8_t parent[32] = {};
    for (uint64_t h = 1; h <= 3; ++h) {
        auto op = make_anchor(h, parent);
        n_ops.push_back(op);
        std::memcpy(parent, op.commit_root, 32);
    }

    auto desc = make_desc(1u);

    auto state1 = ref::AIVMReferenceState::empty();
    auto state2 = ref::AIVMReferenceState::empty();
    auto r1 = ref::run_reference(state1, desc, a_ops, m_ops, n_ops);
    auto r2 = ref::run_reference(state2, desc, a_ops, m_ops, n_ops);

    EXPECT("ref.status",  r1.status == 1u);
    EXPECT("ref.aapply",  r1.attestation_apply_count == 8u);
    EXPECT("ref.mapply",  r1.model_apply_count == 4u);
    EXPECT("ref.napply",  r1.anchor_apply_count == 3u);

    EXPECT("ref.aroot",   std::memcmp(r1.attestation_root, r2.attestation_root, 32) == 0);
    EXPECT("ref.mroot",   std::memcmp(r1.model_registry_root, r2.model_registry_root, 32) == 0);
    EXPECT("ref.nroot",   std::memcmp(r1.audit_root, r2.audit_root, 32) == 0);
    EXPECT("ref.sroot",   std::memcmp(r1.aivm_state_root, r2.aivm_state_root, 32) == 0);

    EXPECT("ref.epoch",   r1.epoch == 1u);

    bool any_nonzero = false;
    for (auto b : r1.aivm_state_root) if (b != 0) { any_nonzero = true; break; }
    EXPECT("ref.sroot.nz", any_nonzero);

    EXPECT("ref.active",   r1.active_attestations == 8u);
    EXPECT("ref.models",   r1.model_count == 4u);
    EXPECT("ref.anchors",  r1.anchor_count == 3u);
    PASS("CPU reference basic determinism");
}

void test_cpu_reference_expiry_excludes()
{
    std::vector<AttestationOp> a_ops;
    auto a1 = make_att(1u, 0u, 1u); // expired
    auto a2 = make_att(2u, 0u, 0u); // never expires
    a_ops.push_back(a1);
    a_ops.push_back(a2);

    auto desc = make_desc(1u);
    desc.timestamp_ns = 100u;

    auto state = ref::AIVMReferenceState::empty();
    auto r = ref::run_reference(state, desc, a_ops, {}, {});
    EXPECT("expiry.applied", r.attestation_apply_count == 2u);
    EXPECT("expiry.expired", r.expired_attestations == 1u);
    EXPECT("expiry.active",  r.active_attestations == 1u);
    PASS("Expired attestation excluded");
}

void test_cpu_reference_wrong_attkey_rejected()
{
    std::vector<AttestationOp> a_ops;
    auto bad = make_att(1u);
    for (auto& b : bad.attesting_key) b = 0u; // zero key = verification failed
    a_ops.push_back(bad);

    auto good = make_att(2u);
    a_ops.push_back(good);

    auto desc = make_desc(1u);
    auto state = ref::AIVMReferenceState::empty();
    auto r = ref::run_reference(state, desc, a_ops, {}, {});
    EXPECT("attkey.applied", r.attestation_apply_count == 1u);
    EXPECT("attkey.active",  r.active_attestations == 1u);
    PASS("Wrong attesting_key rejected");
}

void test_cpu_reference_anchor_chain()
{
    std::vector<AnchorOp> ops;
    uint8_t parent[32] = {};
    for (uint64_t h = 1; h <= 5; ++h) {
        auto op = make_anchor(h, parent);
        ops.push_back(op);
        std::memcpy(parent, op.commit_root, 32);
    }

    auto desc = make_desc(1u);
    auto state = ref::AIVMReferenceState::empty();
    auto r = ref::run_reference(state, desc, {}, {}, ops);
    EXPECT("chain.applied", r.anchor_apply_count == 5u);
    EXPECT("chain.count",   r.anchor_count == 5u);

    // Now attempt to insert a bad-parent anchor — must be rejected.
    AnchorOp bad{};
    for (uint32_t i = 0; i < 32; ++i) bad.commit_root[i] = uint8_t(i + 1u);
    for (uint32_t i = 0; i < 32; ++i) bad.parent_root[i] = uint8_t(0xFFu); // wrong parent
    bad.height = 6u;
    bad.timestamp_ns = desc.timestamp_ns + 100u;

    auto desc2 = make_desc(2u);
    desc2.epoch = 1u;
    auto r2 = ref::run_reference(state, desc2, {}, {}, std::span<const AnchorOp>(&bad, 1));
    EXPECT("chain.bad.rejected", r2.anchor_apply_count == 0u);
    PASS("Anchor chain integrity");
}

void test_cpu_reference_empty_round()
{
    auto desc = make_desc(1u);
    auto state1 = ref::AIVMReferenceState::empty();
    auto state2 = ref::AIVMReferenceState::empty();
    auto r1 = ref::run_reference(state1, desc, {}, {}, {});
    auto r2 = ref::run_reference(state2, desc, {}, {}, {});

    EXPECT("empty.match", std::memcmp(r1.aivm_state_root, r2.aivm_state_root, 32) == 0);
    bool any_nonzero = false;
    for (auto b : r1.aivm_state_root) if (b != 0) { any_nonzero = true; break; }
    EXPECT("empty.nonzero", any_nonzero);
    EXPECT("empty.epoch", r1.epoch == 1u);
    PASS("Empty round deterministic non-zero");
}

void test_cpu_reference_model_update()
{
    std::vector<ModelOp> ops;

    auto reg = make_model_register(1u, 1000u);
    ops.push_back(reg);

    // Update weights with a different weight_hash — should bump version to 2.
    ModelOp upd = reg;
    upd.kind = static_cast<uint32_t>(ModelOpKind::UpdateWeights);
    for (uint32_t i = 0; i < 32; ++i) upd.weight_hash[i] = uint8_t(0xAB ^ i);
    ops.push_back(upd);

    // License update.
    ModelOp lic = reg;
    lic.kind = static_cast<uint32_t>(ModelOpKind::UpdateLicense);
    for (uint32_t i = 0; i < 32; ++i) lic.license_root[i] = uint8_t(0xCD ^ i);
    ops.push_back(lic);

    auto desc = make_desc(1u);
    auto state = ref::AIVMReferenceState::empty();
    auto r = ref::run_reference(state, desc, {}, ops, {});
    EXPECT("upd.applied", r.model_apply_count == 3u);
    EXPECT("upd.count",   r.model_count == 1u);

    // Find the entry — verify version == 2 (register=1, +1 for UpdateWeights).
    bool found = false;
    for (const auto& m : state.models) {
        if (m.occupied == 0u) continue;
        EXPECT("upd.version", m.version == 2u);
        found = true;
        break;
    }
    EXPECT("upd.found", found);
    PASS("Model register + update weights + update license");
}

}  // namespace

int main(int /*argc*/, char** /*argv*/)
{
    setvbuf(stdout, nullptr, _IOLBF, 0);
    std::printf("[aivm_layout_test] starting\n");

    test_attestation_layout();
    test_model_layout();
    test_anchor_layout();
    test_epoch_state_layout();
    test_round_descriptor_layout();
    test_op_layouts();
    test_transition_result_layout();

    test_cpu_reference_basic();
    test_cpu_reference_expiry_excludes();
    test_cpu_reference_wrong_attkey_rejected();
    test_cpu_reference_anchor_chain();
    test_cpu_reference_empty_round();
    test_cpu_reference_model_update();

    std::printf("[aivm_layout_test] passed=%d failed=%d\n", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
