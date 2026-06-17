// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// aivm_determinism_test.cpp — cross-backend determinism harness.
//
// Compares CPU reference vs GPU engine (Metal on Apple, CUDA on Linux+CUDA)
// across canonical workloads. WGSL byte-equivalence is verified at the
// kernel-source level (the encoding matches CPU/Metal/CUDA by construction;
// see aivm_kernels_common.wgsl). When LUX_HAS_WEBGPU is wired, the WGPU
// backend factory becomes available and this harness will exercise the
// fourth path automatically.
//
// Workloads (the brief):
//   1. 1000 attestations across 4 quote families (BLS / Corona /
//      Groth16 / SLH-DSA) → byte-equivalent across backends
//   2. 50 model registrations + license-update path
//   3. 100 audit anchors with parent-root chain integrity
//   4. Attestation expiry properly excludes expired entries
//   5. Wrong attesting_key (zero) → rejected
//   6. Empty round → deterministic non-zero root

#include "lux/aivm/aivm_gpu_engine.hpp"
#include "lux/aivm/aivm_cpu_reference.hpp"

#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

using namespace aivm::gpu;

#if defined(LUX_AIVM_TEST_WGPU)
namespace aivm::gpu {
std::unique_ptr<AIVMGPUEngine> create_aivm_gpu_engine_wgpu();
}
#endif

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
    op.modality = uint32_t(seed % 4u);
    op.kind = static_cast<uint32_t>(ModelOpKind::Register);
    return op;
}

ModelOp make_license_update(const ModelOp& reg, uint8_t fill)
{
    ModelOp upd = reg;
    upd.kind = static_cast<uint32_t>(ModelOpKind::UpdateLicense);
    for (uint32_t i = 0; i < 32; ++i) upd.license_root[i] = uint8_t(fill ^ i);
    return upd;
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

struct WorkloadResult {
    uint8_t  attestation_root[32];
    uint8_t  model_registry_root[32];
    uint8_t  audit_root[32];
    uint8_t  aivm_state_root[32];
    uint32_t active;
    uint32_t expired;
    uint32_t models;
    uint32_t anchors;

    static WorkloadResult from(const AIVMTransitionResult& r) {
        WorkloadResult w{};
        std::memcpy(w.attestation_root,    r.attestation_root,    32);
        std::memcpy(w.model_registry_root, r.model_registry_root, 32);
        std::memcpy(w.audit_root,          r.audit_root,          32);
        std::memcpy(w.aivm_state_root,     r.aivm_state_root,     32);
        w.active  = r.active_attestations;
        w.expired = r.expired_attestations;
        w.models  = r.model_count;
        w.anchors = r.anchor_count;
        return w;
    }

    bool equals(const WorkloadResult& o) const {
        return std::memcmp(attestation_root,    o.attestation_root,    32) == 0
            && std::memcmp(model_registry_root, o.model_registry_root, 32) == 0
            && std::memcmp(audit_root,          o.audit_root,          32) == 0
            && std::memcmp(aivm_state_root,     o.aivm_state_root,     32) == 0
            && active  == o.active
            && expired == o.expired
            && models  == o.models
            && anchors == o.anchors;
    }
};

WorkloadResult run_cpu(const AIVMRoundDescriptor& desc,
                       const std::vector<AttestationOp>& a,
                       const std::vector<ModelOp>& m,
                       const std::vector<AnchorOp>& n)
{
    auto state = ref::AIVMReferenceState::empty();
    auto r = ref::run_reference(state, desc, a, m, n);
    return WorkloadResult::from(r);
}

WorkloadResult run_gpu(AIVMGPUEngine* engine,
                       const AIVMRoundDescriptor& desc,
                       const std::vector<AttestationOp>& a,
                       const std::vector<ModelOp>& m,
                       const std::vector<AnchorOp>& n)
{
    auto h = engine->begin_round(desc);
    if (!a.empty()) engine->push_attestation_ops(h, a);
    if (!m.empty()) engine->push_model_ops(h, m);
    if (!n.empty()) engine->push_anchor_ops(h, n);
    auto r = engine->run_until_done(h);
    engine->end_round(h);
    return WorkloadResult::from(r);
}

void test_brief_workload(AIVMGPUEngine* engine)
{
    std::vector<AttestationOp> a_ops;
    // 1000 attestations across 4 kinds (BLS=0/SGX, Corona=1/SEV,
    // Groth16=2/TDX, SLH-DSA=3/NvTrust). The kernel does not branch on
    // kind; the kind only enters the leaf hash so different kinds produce
    // different roots. Mix them to exercise all four.
    for (uint64_t i = 1; i <= 1000; ++i)
        a_ops.push_back(make_att(i, uint8_t(i % 4u)));

    std::vector<ModelOp> m_ops;
    for (uint64_t i = 1; i <= 50; ++i) {
        auto reg = make_model_register(i, 1000u + i*100u);
        m_ops.push_back(reg);
        if ((i % 5u) == 0u) m_ops.push_back(make_license_update(reg, uint8_t(0x99u + i)));
    }

    std::vector<AnchorOp> n_ops;
    uint8_t parent[32] = {};
    for (uint64_t h = 1; h <= 100; ++h) {
        auto op = make_anchor(h, parent);
        n_ops.push_back(op);
        std::memcpy(parent, op.commit_root, 32);
    }

    auto desc = make_desc(1u);
    auto cpu = run_cpu(desc, a_ops, m_ops, n_ops);

    if (engine == nullptr) {
        auto cpu2 = run_cpu(desc, a_ops, m_ops, n_ops);
        EXPECT("brief.cpu.det", cpu.equals(cpu2));
        std::printf("  brief: CPU-only path; 1000a/50m/100n; root match across runs\n");
        PASS("brief workload (1000a / 50m / 100n) — CPU determinism");
        return;
    }
    auto gpu = run_gpu(engine, desc, a_ops, m_ops, n_ops);
    EXPECT("brief.match", cpu.equals(gpu));
    std::printf("  brief: roots match CPU<->GPU; active=%u models=%u anchors=%u\n",
                gpu.active, gpu.models, gpu.anchors);
    PASS("brief workload (1000a / 50m / 100n) — CPU<->GPU byte match");
}

// v0.59 extension — sweep size families to catch any size-dependent kernel
// drift (XVM caught a small/medium-only race at large/xlarge in its parallel
// rewrite; AIVM's locate-then-writeback is byte-identical at all sizes by
// construction, but we exercise it explicitly).
void test_size_sweep(AIVMGPUEngine* engine)
{
    if (engine == nullptr) {
        PASS("size sweep (skipped — no GPU)");
        return;
    }
    struct Size { const char* name; uint32_t a, m, n; };
    const Size sizes[] = {
        {"small",   100,    5,    10},
        {"medium", 1000,   50,   100},
        // Capped to arena sizes so we exercise saturation but stay
        // determinism-test fast (xlarge takes minutes through Metal apply).
        {"large",  1024,  512,  1024},
    };
    for (const auto& s : sizes) {
        std::vector<AttestationOp> a_ops;
        for (uint64_t i = 1; i <= s.a; ++i)
            a_ops.push_back(make_att(i, uint8_t(i % 4u)));
        std::vector<ModelOp> m_ops;
        for (uint64_t i = 1; i <= s.m; ++i)
            m_ops.push_back(make_model_register(i, 1000u + i*100u));
        std::vector<AnchorOp> n_ops;
        uint8_t parent[32] = {};
        for (uint64_t h = 1; h <= s.n; ++h) {
            auto op = make_anchor(h, parent);
            n_ops.push_back(op);
            std::memcpy(parent, op.commit_root, 32);
        }

        auto desc = make_desc(uint64_t(s.a + s.m + s.n));
        auto cpu = run_cpu(desc, a_ops, m_ops, n_ops);
        auto gpu = run_gpu(engine, desc, a_ops, m_ops, n_ops);
        if (!cpu.equals(gpu)) {
            std::printf("  FAIL[size.%s]: CPU != GPU\n", s.name);
            std::fflush(stdout);
            ++g_failed;
            return;
        }
        std::printf("  size %-7s: %u/%u/%u → active=%u models=%u anchors=%u\n",
                    s.name, s.a, s.m, s.n, gpu.active, gpu.models, gpu.anchors);
    }
    PASS("size sweep small/medium/large — CPU<->GPU byte match");
}

void test_empty_round(AIVMGPUEngine* engine)
{
    auto desc = make_desc(1u);
    auto cpu = run_cpu(desc, {}, {}, {});
    bool nonzero = false;
    for (auto b : cpu.aivm_state_root) if (b != 0) { nonzero = true; break; }
    EXPECT("empty.cpu.nz", nonzero);

    if (engine != nullptr) {
        auto gpu = run_gpu(engine, desc, {}, {}, {});
        EXPECT("empty.match", cpu.equals(gpu));
    }
    PASS("empty round deterministic non-zero");
}

void test_expiry_excludes(AIVMGPUEngine* engine)
{
    std::vector<AttestationOp> a_ops;
    a_ops.push_back(make_att(1u, 0u, 1u));   // expired
    a_ops.push_back(make_att(2u, 0u, 0u));   // never expires
    a_ops.push_back(make_att(3u, 1u, uint64_t(1) << 60)); // far future

    auto desc = make_desc(1u);
    desc.timestamp_ns = 100u;

    auto cpu = run_cpu(desc, a_ops, {}, {});
    EXPECT("expiry.cpu.expired", cpu.expired == 1u);
    EXPECT("expiry.cpu.active",  cpu.active == 2u);

    if (engine != nullptr) {
        auto gpu = run_gpu(engine, desc, a_ops, {}, {});
        EXPECT("expiry.match", cpu.equals(gpu));
    }
    PASS("attestation expiry excludes expired");
}

void test_wrong_attkey_rejected(AIVMGPUEngine* engine)
{
    std::vector<AttestationOp> a_ops;
    auto bad = make_att(1u);
    for (auto& b : bad.attesting_key) b = 0u;
    a_ops.push_back(bad);
    a_ops.push_back(make_att(2u));

    auto desc = make_desc(1u);
    auto cpu = run_cpu(desc, a_ops, {}, {});
    EXPECT("attkey.cpu.applied", cpu.active == 1u);

    if (engine != nullptr) {
        auto gpu = run_gpu(engine, desc, a_ops, {}, {});
        EXPECT("attkey.match", cpu.equals(gpu));
    }
    PASS("wrong attesting_key rejected");
}

void test_anchor_chain(AIVMGPUEngine* engine)
{
    std::vector<AnchorOp> n_ops;
    uint8_t parent[32] = {};
    for (uint64_t h = 1; h <= 100; ++h) {
        auto op = make_anchor(h, parent);
        n_ops.push_back(op);
        std::memcpy(parent, op.commit_root, 32);
    }

    auto desc = make_desc(1u);
    auto cpu = run_cpu(desc, {}, {}, n_ops);
    EXPECT("chain.cpu.count", cpu.anchors == 100u);

    if (engine != nullptr) {
        auto gpu = run_gpu(engine, desc, {}, {}, n_ops);
        EXPECT("chain.match", cpu.equals(gpu));
    }
    PASS("100 audit anchors with parent-root chain");
}

void test_two_engines_match(AIVMGPUEngine* engine)
{
    if (engine == nullptr) {
        PASS("two-engines (skipped — no GPU)");
        return;
    }
    auto a = AIVMGPUEngine::create();
    auto b = AIVMGPUEngine::create();
    EXPECT("twoeng.a", a != nullptr);
    EXPECT("twoeng.b", b != nullptr);

    std::vector<AttestationOp> a_ops;
    for (uint64_t i = 1; i <= 32; ++i)
        a_ops.push_back(make_att(i, uint8_t(i % 4u)));

    std::vector<ModelOp> m_ops;
    for (uint64_t i = 1; i <= 16; ++i) m_ops.push_back(make_model_register(i));

    auto desc = make_desc(1u);
    auto ra = run_gpu(a.get(), desc, a_ops, m_ops, {});
    auto rb = run_gpu(b.get(), desc, a_ops, m_ops, {});
    EXPECT("twoeng.match", ra.equals(rb));
    PASS("two engines bytewise identical");
}

void test_engine_api_surface(AIVMGPUEngine* engine)
{
    // Drives run_epoch / poll_round_result / round_active / handle hygiene
    // on whatever GPU backend the harness is currently running.
    if (engine == nullptr) {
        PASS("api-surface (skipped — no GPU)");
        return;
    }
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

    // Bogus handle ops are silent no-ops.
    AIVMRoundHandle bogus{0xDEADBEEFu};
    engine->push_attestation_ops(bogus, a_ops);
    engine->push_model_ops(bogus, std::span<const ModelOp>{});
    engine->push_anchor_ops(bogus, std::span<const AnchorOp>{});

    engine->push_attestation_ops(h, a_ops);
    auto re = engine->run_epoch(h);
    EXPECT("api.run_epoch.applied", re.attestation_apply_count == 4u);
    EXPECT("api.run_epoch.status",  re.status == 1u);

    auto rp = engine->poll_round_result(h);
    EXPECT("api.poll.match", std::memcmp(rp.aivm_state_root,
                                         re.aivm_state_root, 32) == 0);

    auto rp_bogus = engine->poll_round_result(bogus);
    bool all_zero = true;
    for (auto b : rp_bogus.aivm_state_root) if (b != 0) { all_zero = false; break; }
    EXPECT("api.poll.bogus_zero", all_zero);

    engine->end_round(bogus);
    EXPECT("api.idle.bogus_endround", engine->round_active() == true);

    engine->end_round(h);
    EXPECT("api.idle.after", engine->round_active() == false);
    PASS("API surface: round_active / run_epoch / poll / handle hygiene");
}

}  // namespace

void run_all_against(const char* label, AIVMGPUEngine* engine)
{
    std::printf("[%s] %s\n", label,
                engine ? engine->device_name() : "(CPU-only)");
    test_brief_workload(engine);
    test_empty_round(engine);
    test_expiry_excludes(engine);
    test_wrong_attkey_rejected(engine);
    test_anchor_chain(engine);
    test_two_engines_match(engine);
    test_engine_api_surface(engine);
    test_size_sweep(engine);
}

int main(int /*argc*/, char** /*argv*/)
{
    setvbuf(stdout, nullptr, _IOLBF, 0);
    std::printf("[aivm_determinism_test] starting\n");

    // -- Default backend (Metal on Apple, CUDA on Linux+CUDA, else nullptr) --
    auto engine = AIVMGPUEngine::create();
    run_all_against("default", engine.get());

#if defined(LUX_AIVM_TEST_WGPU)
    // -- WGSL/Dawn backend — runs the same workloads through the WebGPU
    //    runtime and compares to the CPU oracle for byte equivalence. --
    auto wgpu_engine = create_aivm_gpu_engine_wgpu();
    if (wgpu_engine == nullptr) {
        std::printf("[wgpu] runtime unavailable — driver compiled but no Dawn/wgpu-native at link\n");
    } else {
        run_all_against("wgpu", wgpu_engine.get());
    }
#endif

    std::printf("[aivm_determinism_test] passed=%d failed=%d\n", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
