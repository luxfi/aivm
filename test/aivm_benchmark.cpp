// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// aivm_benchmark.cpp — wall-clock benchmark harness across CPU reference and
// every GPU backend the build links in (Metal on Apple, CUDA on Linux+CUDA,
// WGSL via wgpu-native when LUX_AIVM_BENCH_WGPU is defined).
//
// Workloads are AIVM-shaped (attestation-heavy):
//
//   small  : 100   attestations,    5  model regs,   10 audit anchors
//   medium : 1000  attestations,   50  model regs,  100 audit anchors
//   large  : 10000 attestations,  500  model regs, 1000 audit anchors
//   xlarge : 100000 attestations, 5000 model regs, 10000 audit anchors
//
// Each (size × backend) combination runs 3 warm-up + 10 measured iterations
// and reports mean / p50 / p95 / p99 / throughput / speedup-vs-CPU.
//
// Per-mode breakdown (AttestationApply / ProvenanceApply / AnchorApply /
// EpochTransition) is captured by issuing focused single-mode rounds with
// only the relevant op stream — the kernel still walks all four phases but
// the empty phases collapse to a no-op dispatch, leaving the measured
// kernel as the dominant cost.

#include "lux/aivm/aivm_gpu_engine.hpp"
#include "lux/aivm/aivm_cpu_reference.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <vector>

using namespace aivm::gpu;

#if defined(LUX_AIVM_BENCH_WGPU)
namespace aivm::gpu {
std::unique_ptr<AIVMGPUEngine> create_aivm_gpu_engine_wgpu();
}
#endif

namespace {

constexpr int    kWarmups        = 3;
constexpr int    kIters          = 10;
constexpr double kMaxCellSeconds = 60.0;   // hard budget per (backend × size × mode)

struct Workload {
    const char* name;
    uint32_t    n_att;
    uint32_t    n_model;
    uint32_t    n_anchor;
};

const Workload kWorkloads[] = {
    {"small",     100,    5,    10},
    {"medium",   1000,   50,   100},
    {"large",   10000,  500,  1000},
    {"xlarge", 100000, 5000, 10000},
};

AIVMRoundDescriptor make_desc(uint64_t round, uint64_t epoch, uint32_t mode)
{
    AIVMRoundDescriptor d{};
    d.chain_id = 1u;
    d.round = round;
    d.timestamp_ns = 1700000000000000000ULL;
    d.epoch = epoch;
    d.mode = mode;
    d.closing_flag = (mode == static_cast<uint32_t>(AIVMTransitionMode::FullRound)
                   || mode == static_cast<uint32_t>(AIVMTransitionMode::EpochTransition))
                       ? 1u : 0u;
    return d;
}

AttestationOp make_att(uint64_t seed, uint8_t kind = 0u, uint64_t expiry = 0u)
{
    AttestationOp op{};
    for (uint32_t i = 0; i < 32; ++i)
        op.tee_quote_digest[i] = uint8_t((seed * 31u + i) & 0xFFu);
    for (uint32_t i = 0; i < 32; ++i)
        op.measurement[i] = uint8_t((seed * 17u + i) & 0xFFu);
    for (uint32_t i = 0; i < 48; ++i)
        op.attesting_key[i] = uint8_t((seed * 41u + i + 1u) & 0xFFu);
    op.expiry_ns = expiry;
    op.kind = kind;
    op.evidence_offset = 0;
    op.evidence_len = 64u;
    return op;
}

ModelOp make_model(uint64_t seed)
{
    ModelOp op{};
    for (uint32_t i = 0; i < 32; ++i) op.model_root[i]   = uint8_t((seed * 11u + i + 1u) & 0xFFu);
    for (uint32_t i = 0; i < 32; ++i) op.weight_hash[i]  = uint8_t((seed * 23u + i + 7u) & 0xFFu);
    for (uint32_t i = 0; i < 32; ++i) op.license_root[i] = uint8_t((seed *  7u + i + 13u) & 0xFFu);
    for (uint32_t i = 0; i < 20; ++i) op.owner_addr[i]   = uint8_t((seed *  3u + i + 5u) & 0xFFu);
    op.parameter_count = 1000u + seed;
    op.modality = uint32_t(seed % 4u);
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

struct Inputs {
    std::vector<AttestationOp> att;
    std::vector<ModelOp>       model;
    std::vector<AnchorOp>      anchor;
};

Inputs build_inputs(const Workload& w)
{
    Inputs in;
    in.att.reserve(w.n_att);
    in.model.reserve(w.n_model);
    in.anchor.reserve(w.n_anchor);

    for (uint64_t i = 1; i <= w.n_att; ++i)
        in.att.push_back(make_att(i, uint8_t(i % 4u)));
    for (uint64_t i = 1; i <= w.n_model; ++i)
        in.model.push_back(make_model(i));
    uint8_t parent[32] = {};
    for (uint64_t h = 1; h <= w.n_anchor; ++h) {
        auto op = make_anchor(h, parent);
        in.anchor.push_back(op);
        std::memcpy(parent, op.commit_root, 32);
    }
    return in;
}

struct Stats {
    double mean_ms = 0.0;
    double p50_ms  = 0.0;
    double p95_ms  = 0.0;
    double p99_ms  = 0.0;
};

Stats summarize(std::vector<double>& samples_ms)
{
    Stats s;
    if (samples_ms.empty()) return s;
    std::sort(samples_ms.begin(), samples_ms.end());
    double sum = 0.0;
    for (double v : samples_ms) sum += v;
    s.mean_ms = sum / samples_ms.size();
    auto pick = [&](double p) -> double {
        if (samples_ms.empty()) return 0.0;
        const std::size_t n = samples_ms.size();
        const double idx_d = p * (n - 1);
        const std::size_t idx = static_cast<std::size_t>(idx_d);
        const std::size_t hi  = std::min(idx + 1, n - 1);
        const double frac = idx_d - static_cast<double>(idx);
        return samples_ms[idx] + frac * (samples_ms[hi] - samples_ms[idx]);
    };
    s.p50_ms = pick(0.50);
    s.p95_ms = pick(0.95);
    s.p99_ms = pick(0.99);
    return s;
}

double now_ms_since(std::chrono::steady_clock::time_point t0)
{
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// -- runners ---------------------------------------------------------------

double run_cpu_once(const AIVMRoundDescriptor& desc, const Inputs& in)
{
    auto state = ref::AIVMReferenceState::empty();
    auto t0 = std::chrono::steady_clock::now();
    (void)ref::run_reference(state, desc, in.att, in.model, in.anchor);
    return now_ms_since(t0);
}

double run_gpu_once(AIVMGPUEngine* engine,
                    const AIVMRoundDescriptor& desc,
                    const Inputs& in)
{
    auto t0 = std::chrono::steady_clock::now();
    auto h = engine->begin_round(desc);
    if (!in.att.empty())    engine->push_attestation_ops(h, in.att);
    if (!in.model.empty())  engine->push_model_ops(h, in.model);
    if (!in.anchor.empty()) engine->push_anchor_ops(h, in.anchor);
    (void)engine->run_until_done(h);
    engine->end_round(h);
    return now_ms_since(t0);
}

struct BenchResult {
    Stats stats;
    int   iters_run;
    bool  budget_capped;
};

BenchResult bench_cpu(const AIVMRoundDescriptor& desc, const Inputs& in)
{
    for (int i = 0; i < kWarmups; ++i) (void)run_cpu_once(desc, in);
    std::vector<double> samples;
    samples.reserve(kIters);
    auto t_start = std::chrono::steady_clock::now();
    bool capped = false;
    for (int i = 0; i < kIters; ++i) {
        if (now_ms_since(t_start) / 1000.0 > kMaxCellSeconds) { capped = true; break; }
        samples.push_back(run_cpu_once(desc, in));
    }
    return {summarize(samples), int(samples.size()), capped};
}

BenchResult bench_gpu(AIVMGPUEngine* engine,
                      const AIVMRoundDescriptor& desc,
                      const Inputs& in)
{
    // Warmups also subject to the budget — a single warmup may exceed it.
    auto t_start = std::chrono::steady_clock::now();
    bool capped = false;
    for (int i = 0; i < kWarmups; ++i) {
        if (now_ms_since(t_start) / 1000.0 > kMaxCellSeconds) { capped = true; break; }
        (void)run_gpu_once(engine, desc, in);
    }
    std::vector<double> samples;
    samples.reserve(kIters);
    if (!capped) {
        for (int i = 0; i < kIters; ++i) {
            if (now_ms_since(t_start) / 1000.0 > kMaxCellSeconds) { capped = true; break; }
            samples.push_back(run_gpu_once(engine, desc, in));
        }
    }
    return {summarize(samples), int(samples.size()), capped};
}

// -- reporting -------------------------------------------------------------

struct Row {
    std::string backend;
    std::string size;
    Stats       stats;
    double      throughput;       // attestations/sec, the headline workload
    double      speedup_vs_cpu;   // 1.0 for CPU itself
    int         iters;
    bool        capped;
};

void print_header(const char* section)
{
    std::printf("\n=== %s ===\n", section);
    std::printf("%-10s %-8s %10s %10s %10s %10s %14s %10s %6s\n",
                "backend", "size",
                "mean(ms)", "p50(ms)", "p95(ms)", "p99(ms)",
                "ops/sec", "vs CPU", "iters");
}

void print_row(const Row& r)
{
    if (r.iters == 0) {
        std::printf("%-10s %-8s %10s %10s %10s %10s %14s %10s %6s\n",
                    r.backend.c_str(), r.size.c_str(),
                    "skip", "skip", "skip", "skip", "—", "—", "0");
        std::fflush(stdout);
        return;
    }
    std::printf("%-10s %-8s %10.3f %10.3f %10.3f %10.3f %14.0f %9.2fx %5d%s\n",
                r.backend.c_str(), r.size.c_str(),
                r.stats.mean_ms, r.stats.p50_ms, r.stats.p95_ms, r.stats.p99_ms,
                r.throughput, r.speedup_vs_cpu,
                r.iters, r.capped ? "*" : " ");
    std::fflush(stdout);
}

bool skip_for_size(const std::string& backend, const std::string& size,
                   bool per_kernel)
{
    // wgsl runs every kernel under wgpu-native's serial-dispatch path with
    // ~ms-scale Device::maintain poll latency per dispatch. Single iters
    // beyond medium take minutes; xlarge can't be reasonably timed in any
    // budget we're prepared to spend. We capture wgsl correctness at small
    // and medium for FullRound, and only at small for per-kernel breakdown.
    if (backend == "wgsl") {
        if (size == "xlarge") return true;
        if (size == "large")  return true;
        if (per_kernel && size == "medium") return true;
    }
    return false;
}

}  // namespace

// Backend kinds — at most one GPU backend is live at any moment because
// holding both Metal and wgpu-native sessions concurrently causes GPU
// command-buffer stalls on macOS (wgpu-native's queue interferes with the
// Metal command buffer the AIVM driver waits on).
enum class BackendKind { CPU, Default, WGPU };

struct BackendSpec {
    std::string  label;
    BackendKind  kind;
};

std::unique_ptr<AIVMGPUEngine> instantiate(BackendKind k)
{
    switch (k) {
        case BackendKind::CPU:     return nullptr;
        case BackendKind::Default: return AIVMGPUEngine::create();
#if defined(LUX_AIVM_BENCH_WGPU)
        case BackendKind::WGPU:    return create_aivm_gpu_engine_wgpu();
#else
        case BackendKind::WGPU:    return nullptr;
#endif
    }
    return nullptr;
}

// Section runners — outer loop over backends so each GPU engine is created
// once and re-used across every workload size. Recreating the engine per
// (size × mode) cell forces re-compilation of the pipeline state objects
// and saturates the Metal driver under stress.

struct SectionPlan {
    AIVMTransitionMode mode;
    const char*        label;     // "FullRound" / "AttestationApply" / ...
    bool               use_att;
    bool               use_model;
    bool               use_anchor;
    bool               per_kernel;
};

void run_section_for_backend(const SectionPlan& plan,
                             const BackendSpec& spec,
                             AIVMGPUEngine* engine,    // nullptr for CPU
                             Stats* out_cpu_stats_for_size /*[len(kWorkloads)]*/)
{
    for (std::size_t wi = 0; wi < (sizeof(kWorkloads)/sizeof(kWorkloads[0])); ++wi) {
        const auto& w = kWorkloads[wi];
        if (skip_for_size(spec.label, w.name, plan.per_kernel)) {
            Row row;
            row.backend = spec.label; row.size = w.name;
            row.iters = 0; row.capped = true;
            print_row(row);
            continue;
        }
        Inputs in;
        if (plan.use_att)    in.att = build_inputs(w).att;
        if (plan.use_model)  in.model = build_inputs(w).model;
        if (plan.use_anchor) in.anchor = build_inputs(w).anchor;

        auto desc = make_desc(1u, 0u, static_cast<uint32_t>(plan.mode));
        BenchResult br;
        if (spec.kind == BackendKind::CPU) {
            br = bench_cpu(desc, in);
            if (out_cpu_stats_for_size) out_cpu_stats_for_size[wi] = br.stats;
        } else {
            br = bench_gpu(engine, desc, in);
        }

        uint32_t primary;
        if (plan.mode == AIVMTransitionMode::FullRound) primary = w.n_att;
        else if (plan.use_att)    primary = w.n_att;
        else if (plan.use_model)  primary = w.n_model;
        else if (plan.use_anchor) primary = w.n_anchor;
        else                      primary = 1u;

        double cpu_mean = (out_cpu_stats_for_size
                           && out_cpu_stats_for_size[wi].mean_ms > 0.0)
                              ? out_cpu_stats_for_size[wi].mean_ms : 0.0;
        Row row;
        row.backend = spec.label;
        row.size = w.name;
        row.stats = br.stats;
        row.throughput = (br.stats.mean_ms > 0.0)
            ? (double(primary) / (br.stats.mean_ms / 1000.0)) : 0.0;
        row.speedup_vs_cpu = (cpu_mean > 0.0 && br.stats.mean_ms > 0.0)
            ? (cpu_mean / br.stats.mean_ms) : 1.0;
        row.iters = br.iters_run;
        row.capped = br.budget_capped;
        print_row(row);
    }
}

void run_section(const SectionPlan& plan,
                 const std::vector<BackendSpec>& specs)
{
    char header[128];
    if (plan.mode == AIVMTransitionMode::FullRound) {
        std::snprintf(header, sizeof(header),
                      "FullRound (Attestation + Provenance + Anchor + EpochTransition)");
    } else {
        std::snprintf(header, sizeof(header),
                      "%s - single-mode kernel breakdown", plan.label);
    }
    print_header(header);

    Stats cpu_stats[sizeof(kWorkloads)/sizeof(kWorkloads[0])] = {};

    for (const auto& spec : specs) {
        if (spec.kind == BackendKind::CPU) {
            run_section_for_backend(plan, spec, nullptr, cpu_stats);
        } else {
            auto eng = instantiate(spec.kind);
            if (!eng) {
                for (const auto& w : kWorkloads) {
                    Row row;
                    row.backend = spec.label; row.size = w.name;
                    row.iters = 0; row.capped = true;
                    print_row(row);
                }
                continue;
            }
            run_section_for_backend(plan, spec, eng.get(), cpu_stats);
            // Engine destroyed here — next backend gets a clean slate.
        }
    }
}

int main(int /*argc*/, char** /*argv*/)
{
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::printf("[aivm_benchmark] starting (warmups=%d, iters=%d, cap=%.0fs)\n",
                kWarmups, kIters, kMaxCellSeconds);

    // Detect available backends by instantiating once (and immediately
    // destroying) so we never have two GPU contexts live simultaneously.
    std::vector<BackendSpec> specs;
    specs.push_back({"cpu", BackendKind::CPU});
    {
        auto def = AIVMGPUEngine::create();
        if (def) {
            const char* dn = def->device_name();
            std::string label = (dn && std::string(dn).find("NVIDIA") != std::string::npos)
                                    ? "cuda" : "metal";
            std::printf("backend: %-6s -> %s\n", label.c_str(), dn ? dn : "");
            specs.push_back({label, BackendKind::Default});
        } else {
            std::printf("backend: default -> (none)\n");
        }
    }
#if defined(LUX_AIVM_BENCH_WGPU)
    {
        auto w = create_aivm_gpu_engine_wgpu();
        if (w) {
            std::printf("backend: wgsl   -> %s\n", w->device_name());
            specs.push_back({"wgsl", BackendKind::WGPU});
        } else {
            std::printf("backend: wgsl   -> (unavailable)\n");
        }
    }
#endif
    std::printf("backend: cpu    -> reference oracle\n");

    SectionPlan sections[] = {
        {AIVMTransitionMode::FullRound,        "FullRound",        true,  true,  true,  false},
        {AIVMTransitionMode::AttestationApply, "AttestationApply", true,  false, false, true },
        {AIVMTransitionMode::ProvenanceApply,  "ProvenanceApply",  false, true,  false, true },
        {AIVMTransitionMode::AnchorApply,      "AnchorApply",      false, false, true,  true },
        {AIVMTransitionMode::EpochTransition,  "EpochTransition",  true,  true,  true,  true },
    };
    for (const auto& s : sections) run_section(s, specs);

    std::printf("\n[aivm_benchmark] done\n");
    return 0;
}
