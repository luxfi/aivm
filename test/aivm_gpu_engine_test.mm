// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// aivm_gpu_engine_test.mm — Metal-side correctness for AIVMGPUEngine.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "lux/aivm/aivm_gpu_engine.hpp"
#include "lux/aivm/aivm_cpu_reference.hpp"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <system_error>
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

AIVMRoundDescriptor make_desc(uint64_t round, uint64_t epoch = 0)
{
    AIVMRoundDescriptor d{};
    d.chain_id = 1u;
    d.round = round;
    d.timestamp_ns = 1700000000000000000ULL;
    d.epoch = epoch;
    d.mode = static_cast<uint32_t>(AIVMTransitionMode::FullRound);
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
    op.modality = 0u;
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

void test_engine_creates()
{
    auto engine = AIVMGPUEngine::create();
    EXPECT("engine.create", engine != nullptr);
    std::printf("  engine.device: %s\n", engine->device_name());
    PASS("engine creates");
}

void test_engine_full_round_matches_cpu()
{
    auto engine = AIVMGPUEngine::create();
    EXPECT("matches.engine", engine != nullptr);

    std::vector<AttestationOp> a_ops;
    for (uint64_t i = 1; i <= 100; ++i)
        a_ops.push_back(make_att(i, uint8_t(i % 4u)));

    std::vector<ModelOp> m_ops;
    for (uint64_t i = 1; i <= 25; ++i)
        m_ops.push_back(make_model_register(i, 1000u + i*100u));

    std::vector<AnchorOp> n_ops;
    uint8_t parent[32] = {};
    for (uint64_t h = 1; h <= 10; ++h) {
        auto op = make_anchor(h, parent);
        n_ops.push_back(op);
        std::memcpy(parent, op.commit_root, 32);
    }

    auto desc = make_desc(1u);

    auto h = engine->begin_round(desc);
    EXPECT("matches.handle", h.valid());
    engine->push_attestation_ops(h, a_ops);
    engine->push_model_ops(h, m_ops);
    engine->push_anchor_ops(h, n_ops);
    auto gpu_r = engine->run_until_done(h);
    engine->end_round(h);

    auto state = ref::AIVMReferenceState::empty();
    auto cpu_r = ref::run_reference(state, desc, a_ops, m_ops, n_ops);

    EXPECT("matches.status",     gpu_r.status == cpu_r.status);
    EXPECT("matches.aapply",     gpu_r.attestation_apply_count == cpu_r.attestation_apply_count);
    EXPECT("matches.mapply",     gpu_r.model_apply_count       == cpu_r.model_apply_count);
    EXPECT("matches.napply",     gpu_r.anchor_apply_count      == cpu_r.anchor_apply_count);
    EXPECT("matches.active",     gpu_r.active_attestations     == cpu_r.active_attestations);
    EXPECT("matches.expired",    gpu_r.expired_attestations    == cpu_r.expired_attestations);
    EXPECT("matches.models",     gpu_r.model_count             == cpu_r.model_count);
    EXPECT("matches.anchors",    gpu_r.anchor_count            == cpu_r.anchor_count);
    EXPECT("matches.epoch",      gpu_r.epoch                   == cpu_r.epoch);

    EXPECT("matches.aroot", std::memcmp(gpu_r.attestation_root,    cpu_r.attestation_root,    32) == 0);
    EXPECT("matches.mroot", std::memcmp(gpu_r.model_registry_root, cpu_r.model_registry_root, 32) == 0);
    EXPECT("matches.nroot", std::memcmp(gpu_r.audit_root,          cpu_r.audit_root,          32) == 0);
    EXPECT("matches.sroot", std::memcmp(gpu_r.aivm_state_root,     cpu_r.aivm_state_root,     32) == 0);

    std::printf("  gpu  aapply=%u mapply=%u napply=%u active=%u models=%u anchors=%u\n",
                gpu_r.attestation_apply_count, gpu_r.model_apply_count,
                gpu_r.anchor_apply_count, gpu_r.active_attestations,
                gpu_r.model_count, gpu_r.anchor_count);
    PASS("Full round matches CPU reference");
}

void test_engine_empty_round_deterministic()
{
    auto engine = AIVMGPUEngine::create();
    EXPECT("empty.engine", engine != nullptr);

    auto desc = make_desc(1u);
    auto h1 = engine->begin_round(desc);
    auto r1 = engine->run_until_done(h1);
    engine->end_round(h1);
    auto h2 = engine->begin_round(desc);
    auto r2 = engine->run_until_done(h2);
    engine->end_round(h2);

    EXPECT("empty.same", std::memcmp(r1.aivm_state_root, r2.aivm_state_root, 32) == 0);

    bool any_nonzero = false;
    for (auto b : r1.aivm_state_root) if (b != 0) { any_nonzero = true; break; }
    EXPECT("empty.nonzero", any_nonzero);

    auto state = ref::AIVMReferenceState::empty();
    auto cpu_r = ref::run_reference(state, desc, {}, {}, {});
    EXPECT("empty.cpu",  std::memcmp(r1.aivm_state_root, cpu_r.aivm_state_root, 32) == 0);
    PASS("Empty round deterministic and matches CPU");
}

void test_engine_expiry_excludes()
{
    auto engine = AIVMGPUEngine::create();
    EXPECT("expiry.engine", engine != nullptr);

    std::vector<AttestationOp> a_ops;
    a_ops.push_back(make_att(1u, 0u, /*expired*/ 1u));
    a_ops.push_back(make_att(2u, 0u, /*never*/   0u));

    auto desc = make_desc(1u);
    desc.timestamp_ns = 100u;

    auto h = engine->begin_round(desc);
    engine->push_attestation_ops(h, a_ops);
    auto gpu_r = engine->run_until_done(h);
    engine->end_round(h);

    auto state = ref::AIVMReferenceState::empty();
    auto cpu_r = ref::run_reference(state, desc, a_ops, {}, {});
    EXPECT("expiry.match.expired", gpu_r.expired_attestations == cpu_r.expired_attestations);
    EXPECT("expiry.match.active",  gpu_r.active_attestations  == cpu_r.active_attestations);
    EXPECT("expiry.match.sroot",   std::memcmp(gpu_r.aivm_state_root, cpu_r.aivm_state_root, 32) == 0);
    PASS("Expired attestation excluded");
}

void test_engine_anchor_chain_integrity()
{
    auto engine = AIVMGPUEngine::create();
    EXPECT("chain.engine", engine != nullptr);

    std::vector<AnchorOp> good;
    uint8_t parent[32] = {};
    for (uint64_t h = 1; h <= 100; ++h) {
        auto op = make_anchor(h, parent);
        good.push_back(op);
        std::memcpy(parent, op.commit_root, 32);
    }

    auto desc = make_desc(1u);
    auto h = engine->begin_round(desc);
    engine->push_anchor_ops(h, good);
    auto gpu_r = engine->run_until_done(h);
    engine->end_round(h);

    auto state = ref::AIVMReferenceState::empty();
    auto cpu_r = ref::run_reference(state, desc, {}, {}, good);
    EXPECT("chain.match.napply", gpu_r.anchor_apply_count == cpu_r.anchor_apply_count);
    EXPECT("chain.match.count",  gpu_r.anchor_count       == cpu_r.anchor_count);
    EXPECT("chain.match.uroot",  std::memcmp(gpu_r.audit_root, cpu_r.audit_root, 32) == 0);
    PASS("Anchor chain integrity (100 anchors)");
}

void test_engine_api_surface()
{
    // Drive the rest of the AIVMGPUEngine API surface that the brief
    // determinism tests do not hit:
    //   * round_active() before/during/after a round
    //   * run_epoch() (alternative to run_until_done)
    //   * poll_round_result() returns the same struct as run_*
    //   * double begin_round() must reject (returns invalid handle)
    //   * push_*_ops on an invalid handle is a no-op
    //   * end_round on an invalid handle is a no-op
    auto engine = AIVMGPUEngine::create();
    EXPECT("api.engine", engine != nullptr);
    EXPECT("api.idle.before", engine->round_active() == false);

    std::vector<AttestationOp> a_ops;
    for (uint64_t i = 1; i <= 4; ++i) a_ops.push_back(make_att(i, 0u));

    auto desc = make_desc(1u);
    auto h = engine->begin_round(desc);
    EXPECT("api.handle", h.valid());
    EXPECT("api.idle.during", engine->round_active() == true);

    // Double begin_round must reject without disturbing the active round.
    auto h2 = engine->begin_round(desc);
    EXPECT("api.double_begin", h2.valid() == false);

    // Push ops with an invalid handle — must be a silent no-op.
    AIVMRoundHandle bogus{0xDEADBEEF};
    engine->push_attestation_ops(bogus, a_ops);
    engine->push_model_ops(bogus, std::span<const ModelOp>{});
    engine->push_anchor_ops(bogus, std::span<const AnchorOp>{});

    engine->push_attestation_ops(h, a_ops);
    auto re = engine->run_epoch(h);
    EXPECT("api.run_epoch.applied", re.attestation_apply_count == 4u);
    EXPECT("api.run_epoch.status",  re.status == 1u);
    bool nz = false;
    for (auto b : re.aivm_state_root) if (b != 0) { nz = true; break; }
    EXPECT("api.run_epoch.root_nz", nz);

    // poll_round_result on the active handle must report the same root.
    auto rp = engine->poll_round_result(h);
    EXPECT("api.poll.match", std::memcmp(rp.aivm_state_root,
                                         re.aivm_state_root, 32) == 0);

    // poll_round_result with a bogus handle returns a zero result.
    auto rp_bogus = engine->poll_round_result(bogus);
    bool all_zero = true;
    for (auto b : rp_bogus.aivm_state_root) if (b != 0) { all_zero = false; break; }
    EXPECT("api.poll.bogus_zero", all_zero);

    // end_round with bogus handle is a no-op (round still active).
    engine->end_round(bogus);
    EXPECT("api.idle.bogus_endround", engine->round_active() == true);

    engine->end_round(h);
    EXPECT("api.idle.after", engine->round_active() == false);
    PASS("Engine API surface: round_active / run_epoch / poll / handle hygiene");
}

void test_engine_destructor_with_active_round()
{
    // The Metal engine's destructor calls end_round() if a round is
    // still active. Drives that branch.
    {
        auto engine = AIVMGPUEngine::create();
        EXPECT("dtor.engine", engine != nullptr);
        auto h = engine->begin_round(make_desc(7u));
        EXPECT("dtor.handle", h.valid());
        // Intentionally no end_round — let the destructor close it.
    }
    PASS("Destructor cleans up an active round");
}

void test_engine_source_compile_fallback()
{
    // Force the metallib-load path to fail by renaming all candidate
    // metallib files aside, then create an engine. The engine must fall
    // back to compile_aivm_library() which compiles the .metal sources
    // online. Restore the files at the end so subsequent tests still hit
    // the fast metallib path.
    namespace fs = std::filesystem;
    std::vector<fs::path> moved;
    auto try_move = [&](const fs::path& p) {
        std::error_code ec;
        if (!fs::exists(p, ec)) return;
        fs::path away = p;
        away += ".hidden";
        fs::rename(p, away, ec);
        if (!ec) moved.push_back(p);
    };

    fs::path here = fs::path(__FILE__).parent_path();
    try_move(here / "aivm.metallib");
    try_move(fs::current_path() / "aivm.metallib");
    try_move(fs::current_path() / "src" / "aivm.metallib");
    try_move(fs::current_path() / "aivm" / "src" / "aivm.metallib");
    try_move(fs::current_path().parent_path() / "src" / "aivm.metallib");

    auto engine = AIVMGPUEngine::create();
    bool got_engine = (engine != nullptr);

    // Restore everything before asserting so a failure does not leave
    // the build dir in a broken state for the next ctest run.
    for (const auto& p : moved) {
        fs::path away = p;
        away += ".hidden";
        std::error_code ec;
        fs::rename(away, p, ec);
    }

    EXPECT("fallback.engine", got_engine);
    if (engine) {
        // Sanity: a tiny round through the source-compiled library still
        // matches the CPU oracle byte-for-byte.
        std::vector<AttestationOp> a_ops;
        for (uint64_t i = 1; i <= 4; ++i) a_ops.push_back(make_att(i, 0u));
        auto desc = make_desc(1u);
        auto h = engine->begin_round(desc);
        engine->push_attestation_ops(h, a_ops);
        auto gpu_r = engine->run_until_done(h);
        engine->end_round(h);

        auto state = ref::AIVMReferenceState::empty();
        auto cpu_r = ref::run_reference(state, desc, a_ops, {}, {});
        EXPECT("fallback.match",
               std::memcmp(gpu_r.aivm_state_root, cpu_r.aivm_state_root, 32) == 0);
    }
    PASS("Metal source-compile fallback works when metallib is missing");
}

}  // namespace

int main(int /*argc*/, char** /*argv*/)
{
    setvbuf(stdout, nullptr, _IOLBF, 0);
    @autoreleasepool {
        std::printf("[aivm_gpu_engine_test] starting\n");

        test_engine_creates();
        test_engine_full_round_matches_cpu();
        test_engine_empty_round_deterministic();
        test_engine_expiry_excludes();
        test_engine_anchor_chain_integrity();
        test_engine_api_surface();
        test_engine_destructor_with_active_round();
        test_engine_source_compile_fallback();

        std::printf("[aivm_gpu_engine_test] passed=%d failed=%d\n",
                    g_passed, g_failed);
        return g_failed == 0 ? 0 : 1;
    }
}
