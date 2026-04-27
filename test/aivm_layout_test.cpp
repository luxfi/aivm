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

void test_cpu_reference_model_transfer()
{
    // Exercise ModelOpKind::Transfer, which neither of the existing tests
    // hits. Register a model, then transfer ownership; final owner_addr
    // must be the new address and the entry must remain occupied.
    std::vector<ModelOp> ops;
    auto reg = make_model_register(7u, 4242u);
    ops.push_back(reg);

    ModelOp xfer = reg;
    xfer.kind = static_cast<uint32_t>(ModelOpKind::Transfer);
    for (uint32_t i = 0; i < 20; ++i) xfer.owner_addr[i] = uint8_t(0xE0 + i);
    ops.push_back(xfer);

    auto desc = make_desc(1u);
    auto state = ref::AIVMReferenceState::empty();
    auto r = ref::run_reference(state, desc, {}, ops, {});
    EXPECT("xfer.applied", r.model_apply_count == 2u);
    EXPECT("xfer.count",   r.model_count == 1u);

    bool checked = false;
    for (const auto& m : state.models) {
        if (m.occupied == 0u) continue;
        bool match_owner = true;
        for (uint32_t i = 0; i < 20; ++i)
            if (m.owner_addr[i] != uint8_t(0xE0 + i)) { match_owner = false; break; }
        EXPECT("xfer.owner", match_owner);
        // Transfer does not mutate weight_hash/license_root.
        EXPECT("xfer.weight_unchanged",
               std::memcmp(m.weight_hash, reg.weight_hash, 32) == 0);
        checked = true;
        break;
    }
    EXPECT("xfer.found", checked);

    // Transfer of an unknown model_root must be a no-op (model_locate
    // miss without insert path).
    ModelOp xfer_miss = xfer;
    for (uint32_t i = 0; i < 32; ++i) xfer_miss.model_root[i] = uint8_t(0xA0 ^ i);
    auto desc2 = make_desc(2u);
    desc2.epoch = 1u;
    auto r2 = ref::run_reference(state, desc2, {}, std::span<const ModelOp>(&xfer_miss, 1), {});
    EXPECT("xfer.miss", r2.model_apply_count == 0u);
    PASS("Model transfer + transfer-of-missing rejected");
}

void test_cpu_reference_update_missing_model()
{
    // UpdateWeights and UpdateLicense against an unregistered model_root —
    // both must be no-ops (this exercises model_locate's
    // "missing without insert" branch).
    auto desc = make_desc(1u);
    auto state = ref::AIVMReferenceState::empty();

    ModelOp upd_w{};
    for (uint32_t i = 0; i < 32; ++i) upd_w.model_root[i]  = uint8_t(0x11 + i);
    for (uint32_t i = 0; i < 32; ++i) upd_w.weight_hash[i] = uint8_t(0x22 + i);
    upd_w.kind = static_cast<uint32_t>(ModelOpKind::UpdateWeights);

    ModelOp upd_l{};
    for (uint32_t i = 0; i < 32; ++i) upd_l.model_root[i]  = uint8_t(0x33 + i);
    for (uint32_t i = 0; i < 32; ++i) upd_l.weight_hash[i] = uint8_t(0x44 + i);
    for (uint32_t i = 0; i < 32; ++i) upd_l.license_root[i]= uint8_t(0x55 + i);
    upd_l.kind = static_cast<uint32_t>(ModelOpKind::UpdateLicense);

    std::vector<ModelOp> ops{upd_w, upd_l};
    auto r = ref::run_reference(state, desc, {}, ops, {});
    EXPECT("missupd.zero", r.model_apply_count == 0u);
    EXPECT("missupd.count", r.model_count == 0u);
    PASS("UpdateWeights/UpdateLicense on missing model is a no-op");
}

void test_cpu_reference_zero_input_skipped()
{
    // Drives the digest_zero==true branches in apply_attestation_ops and
    // apply_model_ops which the existing tests do not exercise.
    auto desc = make_desc(1u);
    auto state = ref::AIVMReferenceState::empty();

    AttestationOp a_zero{};        // all zero — must be rejected
    a_zero.kind = 0u;
    a_zero.evidence_len = 64u;     // metadata set, but digest+key are zero

    AttestationOp a_zero_digest{}; // non-zero key, zero digest
    for (uint32_t i = 0; i < 48; ++i) a_zero_digest.attesting_key[i] = uint8_t(i + 1u);

    ModelOp m_zero_root{};         // zero model_root
    for (uint32_t i = 0; i < 32; ++i) m_zero_root.weight_hash[i] = uint8_t(i + 1u);
    m_zero_root.kind = static_cast<uint32_t>(ModelOpKind::Register);

    ModelOp m_zero_weight{};       // zero weight_hash
    for (uint32_t i = 0; i < 32; ++i) m_zero_weight.model_root[i] = uint8_t(i + 1u);
    m_zero_weight.kind = static_cast<uint32_t>(ModelOpKind::Register);

    AnchorOp n_zero{};             // zero commit_root
    n_zero.height = 1u;

    std::vector<AttestationOp> a_ops{a_zero, a_zero_digest};
    std::vector<ModelOp>       m_ops{m_zero_root, m_zero_weight};
    std::vector<AnchorOp>      n_ops{n_zero};

    auto r = ref::run_reference(state, desc, a_ops, m_ops, n_ops);
    EXPECT("zero.aapply", r.attestation_apply_count == 0u);
    EXPECT("zero.mapply", r.model_apply_count == 0u);
    EXPECT("zero.napply", r.anchor_apply_count == 0u);
    EXPECT("zero.active", r.active_attestations == 0u);
    EXPECT("zero.models", r.model_count == 0u);
    EXPECT("zero.anchors", r.anchor_count == 0u);
    PASS("Zero-digest / zero-root / zero-key inputs are filtered");
}

void test_cpu_reference_uninitialised_state()
{
    // run_reference must lazily populate empty arenas — exercises the
    // `if (state.attestations.empty()) ...` branches.
    ref::AIVMReferenceState s; // raw default-constructed, no arenas
    auto desc = make_desc(1u);
    auto r = ref::run_reference(s, desc, {}, {}, {});
    EXPECT("uninit.status", r.status == 1u);
    EXPECT("uninit.atts.size",   s.attestations.size() == kDefaultAttestationSlots);
    EXPECT("uninit.models.size", s.models.size()       == kDefaultModelSlots);
    EXPECT("uninit.anchors.size", s.anchors.size()     == kDefaultAnchorSlots);

    // Same thing again, this time with one of each pre-populated and the
    // other two empty — checks that the lazy init only fires for the
    // empty arenas (no double-init).
    ref::AIVMReferenceState s2;
    s2.attestations.assign(kDefaultAttestationSlots, Attestation{});
    auto r2 = ref::run_reference(s2, desc, {}, {}, {});
    EXPECT("uninit.partial.status", r2.status == 1u);
    EXPECT("uninit.partial.models", s2.models.size()  == kDefaultModelSlots);
    EXPECT("uninit.partial.anchors", s2.anchors.size() == kDefaultAnchorSlots);
    PASS("Lazy state initialisation fires for empty arenas");
}

void test_cpu_reference_anchor_height_monotonic()
{
    // The anchor kernel rejects anchors whose height is <= previous height,
    // even if the parent_root chains correctly. Exercises the
    // `op.height <= prev.height` branch in apply_anchor_ops.
    std::vector<AnchorOp> ops;
    uint8_t parent[32] = {};

    auto a1 = make_anchor(10u, parent);
    ops.push_back(a1);
    std::memcpy(parent, a1.commit_root, 32);

    // Height regression: still chains (parent matches a1.commit_root) but
    // height is lower than previous. Must be rejected.
    auto a_bad = make_anchor(5u, parent);
    ops.push_back(a_bad);

    // Then a valid one again to confirm the kernel keeps going after a
    // rejection.
    auto a_good = make_anchor(11u, parent);
    ops.push_back(a_good);

    auto desc = make_desc(1u);
    auto state = ref::AIVMReferenceState::empty();
    auto r = ref::run_reference(state, desc, {}, {}, ops);
    EXPECT("mono.applied", r.anchor_apply_count == 2u);
    EXPECT("mono.count",   r.anchor_count == 2u);
    PASS("Anchor height monotonicity enforced");
}

void test_cpu_reference_hash_probe_collision()
{
    // Force the open-addressing probe-advance branch by constructing two
    // model_roots whose first 8 bytes collide on the FNV-style index
    // (mask = kDefaultModelSlots - 1 = 511). After registering both,
    // model_locate on the second has to probe past the first slot —
    // exercising the `idx = (idx + 1u) & mask` path that the brief
    // workloads do not touch.
    auto fnv_index = [](uint64_t k) -> uint32_t {
        uint64_t h = 0xcbf29ce484222325ULL;
        h = (h ^ k) * 0x100000001b3ULL;
        return uint32_t(h) & (kDefaultModelSlots - 1u);
    };
    auto first8 = [](uint8_t out[8], uint64_t v) {
        for (uint32_t i = 0; i < 8u; ++i) out[i] = uint8_t(v >> (i * 8u));
    };

    uint64_t k1 = 1u;
    uint32_t target_bucket = fnv_index(k1);
    uint64_t k2 = 0;
    for (uint64_t cand = 2; cand < (1u << 22); ++cand) {
        if (fnv_index(cand) == target_bucket) { k2 = cand; break; }
    }
    EXPECT("probe.found_collision", k2 != 0);

    ModelOp m1{};
    first8(m1.model_root,  k1); m1.model_root[31]  = 0x11; // distinguish full digest
    for (uint32_t i = 0; i < 32; ++i) m1.weight_hash[i]  = uint8_t(0xA0 + i);
    for (uint32_t i = 0; i < 32; ++i) m1.license_root[i] = uint8_t(0xB0 + i);
    m1.kind = static_cast<uint32_t>(ModelOpKind::Register);

    ModelOp m2{};
    first8(m2.model_root,  k2); m2.model_root[31]  = 0x22;
    for (uint32_t i = 0; i < 32; ++i) m2.weight_hash[i]  = uint8_t(0xC0 + i);
    for (uint32_t i = 0; i < 32; ++i) m2.license_root[i] = uint8_t(0xD0 + i);
    m2.kind = static_cast<uint32_t>(ModelOpKind::Register);

    // Sanity — the brief slice hashes really do match.
    uint64_t k1r = 0, k2r = 0;
    for (uint32_t i = 0; i < 8u; ++i) {
        k1r |= uint64_t(m1.model_root[i]) << (i * 8u);
        k2r |= uint64_t(m2.model_root[i]) << (i * 8u);
    }
    EXPECT("probe.collide", fnv_index(k1r) == fnv_index(k2r));

    auto desc = make_desc(1u);
    auto state = ref::AIVMReferenceState::empty();
    std::vector<ModelOp> ops{m1, m2};
    auto r = ref::run_reference(state, desc, {}, ops, {});
    EXPECT("probe.applied", r.model_apply_count == 2u);
    EXPECT("probe.count",   r.model_count == 2u);

    // Now drive a second-round UpdateLicense for both models — the locate
    // for m2 must traverse past m1's slot, exercising the probe-advance.
    ModelOp upd1 = m1;
    upd1.kind = static_cast<uint32_t>(ModelOpKind::UpdateLicense);
    for (uint32_t i = 0; i < 32; ++i) upd1.license_root[i] = uint8_t(0x10 ^ i);

    ModelOp upd2 = m2;
    upd2.kind = static_cast<uint32_t>(ModelOpKind::UpdateLicense);
    for (uint32_t i = 0; i < 32; ++i) upd2.license_root[i] = uint8_t(0x20 ^ i);

    auto desc2 = make_desc(2u);
    desc2.epoch = 1u;
    auto r2 = ref::run_reference(state, desc2, {}, std::vector<ModelOp>{upd1, upd2}, {});
    EXPECT("probe.upd.applied", r2.model_apply_count == 2u);
    PASS("Open-addressing probe-advance after hash collision");
}

void test_cpu_reference_anchor_arena_full()
{
    // kDefaultAnchorSlots = 4096. Drive `cursor >= state.anchors.size()`
    // by pushing more anchors across rounds than the arena can hold.
    auto state = ref::AIVMReferenceState::empty();
    uint8_t parent[32] = {};
    uint64_t height = 1;

    auto run_chunk = [&](uint32_t round, uint32_t count) {
        std::vector<AnchorOp> ops;
        for (uint32_t i = 0; i < count; ++i) {
            auto op = make_anchor(height++, parent);
            ops.push_back(op);
            std::memcpy(parent, op.commit_root, 32);
        }
        auto desc = make_desc(round);
        desc.epoch = round - 1u;
        return ref::run_reference(state, desc, {}, {}, ops);
    };

    // Fill 4 rounds of 1024 = 4096 (full arena).
    for (uint32_t r = 1; r <= 4; ++r) {
        auto rr = run_chunk(r, 1024u);
        EXPECT("arena.applied", rr.anchor_apply_count == 1024u);
    }

    // Now push 1 more — the arena is full so cursor >= size and the
    // op must be dropped without applying.
    auto over = run_chunk(5u, 1u);
    EXPECT("arena.overflow", over.anchor_apply_count == 0u);
    EXPECT("arena.full_count", over.anchor_count == 4096u);
    PASS("Anchor arena gracefully refuses when full");
}

void test_cpu_reference_version_saturates()
{
    // Force the saturating_add overflow branch by registering a model
    // and then directly priming its `version` field at UINT64_MAX,
    // followed by an UpdateWeights op. The reference must clamp the
    // version to UINT64_MAX rather than wrap.
    auto desc = make_desc(1u);
    auto state = ref::AIVMReferenceState::empty();

    auto reg = make_model_register(11u, 999u);
    auto r = ref::run_reference(state, desc, {},
                                std::span<const ModelOp>(&reg, 1), {});
    EXPECT("sat.reg", r.model_apply_count == 1u);

    // Find the model and saturate its version manually.
    bool primed = false;
    for (auto& m : state.models) {
        if (m.occupied == 0u) continue;
        m.version = UINT64_MAX;
        primed = true;
        break;
    }
    EXPECT("sat.primed", primed);

    ModelOp upd = reg;
    upd.kind = static_cast<uint32_t>(ModelOpKind::UpdateWeights);
    for (uint32_t i = 0; i < 32; ++i) upd.weight_hash[i] = uint8_t(0x77 + i);

    auto desc2 = make_desc(2u);
    desc2.epoch = 1u;
    auto r2 = ref::run_reference(state, desc2, {},
                                 std::span<const ModelOp>(&upd, 1), {});
    EXPECT("sat.applied", r2.model_apply_count == 1u);
    for (const auto& m : state.models) {
        if (m.occupied == 0u) continue;
        EXPECT("sat.clamped", m.version == UINT64_MAX);
        break;
    }
    PASS("version saturates at UINT64_MAX");
}

void test_cpu_reference_already_expired_flag()
{
    // Drive the `(a.status & kAttStatusExpired) != 0u` branch in
    // compute_attestation_root: an attestation can already carry the
    // expired bit even if expiry_ns == 0 (e.g. set by an earlier round).
    // Brief tests only set expired via expiry_ns timestamps.
    auto desc = make_desc(1u);
    auto state = ref::AIVMReferenceState::empty();
    auto a = make_att(99u, 0u, 0u);
    auto r = ref::run_reference(state, desc,
                                std::span<const AttestationOp>(&a, 1), {}, {});
    EXPECT("flagexp.applied", r.attestation_apply_count == 1u);

    // Manually flip the already-expired bit on the entry, then close
    // another epoch. The kernel must recognise it as expired without
    // any expiry_ns trigger.
    bool flipped = false;
    for (auto& slot : state.attestations) {
        if (slot.occupied == 0u) continue;
        slot.status |= 0x4u;     // kAttStatusExpired
        flipped = true;
        break;
    }
    EXPECT("flagexp.flipped", flipped);

    auto desc2 = make_desc(2u);
    desc2.epoch = 1u;
    auto r2 = ref::run_reference(state, desc2, {}, {}, {});
    EXPECT("flagexp.expired", r2.expired_attestations == 1u);
    EXPECT("flagexp.active",  r2.active_attestations == 0u);
    PASS("Pre-set kAttStatusExpired is honored at root computation");
}

void test_cpu_reference_modes()
{
    // Exercises every AIVMTransitionMode dispatch path — only FullRound is
    // covered by the brief tests above.
    auto state = ref::AIVMReferenceState::empty();

    auto a = make_att(1u);
    auto m = make_model_register(1u);
    uint8_t parent[32] = {};
    auto n = make_anchor(1u, parent);

    auto desc_a = make_desc(1u);
    desc_a.mode = static_cast<uint32_t>(AIVMTransitionMode::AttestationApply);
    desc_a.closing_flag = 0u;
    auto ra = ref::run_reference(state, desc_a,
                                 std::span<const AttestationOp>(&a, 1), {}, {});
    EXPECT("mode.a.applied", ra.attestation_apply_count == 1u);
    EXPECT("mode.a.no_models", ra.model_apply_count == 0u);
    EXPECT("mode.a.no_anchors", ra.anchor_apply_count == 0u);

    auto desc_m = make_desc(2u);
    desc_m.mode = static_cast<uint32_t>(AIVMTransitionMode::ProvenanceApply);
    desc_m.closing_flag = 0u;
    auto rm = ref::run_reference(state, desc_m, {},
                                 std::span<const ModelOp>(&m, 1), {});
    EXPECT("mode.m.applied", rm.model_apply_count == 1u);
    EXPECT("mode.m.no_atts", rm.attestation_apply_count == 0u);

    auto desc_n = make_desc(3u);
    desc_n.mode = static_cast<uint32_t>(AIVMTransitionMode::AnchorApply);
    desc_n.closing_flag = 0u;
    auto rn = ref::run_reference(state, desc_n, {}, {},
                                 std::span<const AnchorOp>(&n, 1));
    EXPECT("mode.n.applied", rn.anchor_apply_count == 1u);
    EXPECT("mode.n.no_models", rn.model_apply_count == 0u);

    auto desc_e = make_desc(4u);
    desc_e.mode = static_cast<uint32_t>(AIVMTransitionMode::EpochTransition);
    desc_e.closing_flag = 1u;
    auto re = ref::run_reference(state, desc_e, {}, {}, {});
    EXPECT("mode.e.no_apply",
           re.attestation_apply_count == 0u && re.model_apply_count == 0u
        && re.anchor_apply_count == 0u);
    // EpochTransition still computes roots over carried-forward state.
    bool nz = false;
    for (auto b : re.aivm_state_root) if (b != 0) { nz = true; break; }
    EXPECT("mode.e.root_nz", nz);
    PASS("Each AIVMTransitionMode dispatch path");
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
    test_cpu_reference_model_transfer();
    test_cpu_reference_update_missing_model();
    test_cpu_reference_zero_input_skipped();
    test_cpu_reference_uninitialised_state();
    test_cpu_reference_anchor_height_monotonic();
    test_cpu_reference_hash_probe_collision();
    test_cpu_reference_anchor_arena_full();
    test_cpu_reference_version_saturates();
    test_cpu_reference_already_expired_flag();
    test_cpu_reference_modes();

    std::printf("[aivm_layout_test] passed=%d failed=%d\n", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
