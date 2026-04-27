// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// aivm_gpu_engine.mm — Metal-backed driver for AIVMGPUEngine.
//
// One round = a chain of kernel dispatches that preserves byte-for-byte
// determinism with the CPU reference while exploiting per-record GPU
// parallelism for the heavy paths:
//
//   1. aivm_attestation_locate   (1 thread)   — canonical slot allocation
//   2. aivm_attestation_writeback (parallel)  — per-slot field write
//   3. aivm_provenance_locate    (1 thread)   — canonical slot allocation,
//                                              version-counter semantics
//                                              require in-order replay
//   4. aivm_anchor_apply         (1 thread)   — chain integrity is
//                                              fundamentally sequential
//   5. aivm_epoch_mark_expired   (parallel)   — per-attestation expiry bit
//   6. aivm_epoch_attestation_leaves (parallel) — per-slot keccak leaf
//   7. aivm_epoch_model_leaves   (parallel)   — per-slot keccak leaf
//   8. aivm_epoch_anchor_leaves  (parallel)   — per-slot keccak leaf
//   9. aivm_epoch_finalize       (1 thread)   — serial fold + state root
//
// Parallel kernels dispatch one thread per slot/op with threadgroup width
// = 256 (clean multiple of Apple SIMD width 32).

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "lux/aivm/aivm_gpu_engine.hpp"

#include <atomic>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

namespace aivm::gpu {

namespace {

id<MTLLibrary> load_aivm_metallib(id<MTLDevice> device)
{
    NSError* error = nil;
    std::filesystem::path here = std::filesystem::path(__FILE__).parent_path();
    std::filesystem::path candidates[] = {
        here / "aivm.metallib",
        std::filesystem::current_path() / "aivm.metallib",
        std::filesystem::current_path() / "src" / "aivm.metallib",
        std::filesystem::current_path() / "aivm" / "src" / "aivm.metallib",
        std::filesystem::current_path().parent_path() / "src" / "aivm.metallib",
    };
    for (const auto& p : candidates) {
        if (!std::filesystem::exists(p)) continue;
        NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:p.c_str()]];
        id<MTLLibrary> lib = [device newLibraryWithURL:url error:&error];
        if (lib) return lib;
        if (error)
            std::fprintf(stderr, "aivm metallib load error %s: %s\n",
                         p.c_str(), [[error localizedDescription] UTF8String]);
    }
    return nil;
}

id<MTLLibrary> compile_aivm_library(id<MTLDevice> device)
{
    NSError* error = nil;
    std::filesystem::path here = std::filesystem::path(__FILE__).parent_path();
    std::filesystem::path candidates_dir[] = {
        here,
        std::filesystem::current_path(),
        std::filesystem::current_path() / "src",
        std::filesystem::current_path() / "aivm" / "src",
        std::filesystem::current_path().parent_path() / "src",
    };
    auto load_file = [&](const std::filesystem::path& p) -> NSString* {
        if (!std::filesystem::exists(p)) return nil;
        NSString* path = [NSString stringWithUTF8String:p.c_str()];
        return [NSString stringWithContentsOfFile:path
                                         encoding:NSUTF8StringEncoding
                                            error:&error];
    };
    NSString* common = nil;
    NSString* k_a = nil;
    NSString* k_p = nil;
    NSString* k_an = nil;
    NSString* k_e = nil;
    for (const auto& dir : candidates_dir) {
        common = load_file(dir / "aivm_kernels_common.h.metal");
        k_a    = load_file(dir / "aivm_attestation.metal");
        k_p    = load_file(dir / "aivm_provenance.metal");
        k_an   = load_file(dir / "aivm_anchor.metal");
        k_e    = load_file(dir / "aivm_transition.metal");
        if (common && k_a && k_p && k_an && k_e) break;
    }
    if (!common || !k_a || !k_p || !k_an || !k_e) {
        std::fprintf(stderr, "AIVM Metal sources not found near %s\n",
                     here.c_str());
        return nil;
    }
    auto strip_include = [](NSString* src) -> NSString* {
        NSMutableString* out = [NSMutableString string];
        NSArray<NSString*>* lines = [src componentsSeparatedByString:@"\n"];
        for (NSString* line in lines) {
            if ([line containsString:@"aivm_kernels_common.h.metal"]) continue;
            [out appendString:line];
            [out appendString:@"\n"];
        }
        return out;
    };
    NSMutableString* combined = [NSMutableString string];
    [combined appendString:common];
    [combined appendString:@"\n"];
    [combined appendString:strip_include(k_a)];
    [combined appendString:@"\n"];
    [combined appendString:strip_include(k_p)];
    [combined appendString:@"\n"];
    [combined appendString:strip_include(k_an)];
    [combined appendString:@"\n"];
    [combined appendString:strip_include(k_e)];

    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    opts.languageVersion = MTLLanguageVersion3_0;
    id<MTLLibrary> lib = [device newLibraryWithSource:combined
                                              options:opts
                                                error:&error];
    if (!lib && error)
        std::fprintf(stderr, "AIVM Metal compile error: %s\n",
                     [[error localizedDescription] UTF8String]);
    return lib;
}

constexpr uint32_t kAttSlots       = kDefaultAttestationSlots;
constexpr uint32_t kModelSlots     = kDefaultModelSlots;
constexpr uint32_t kAnchorSlots    = kDefaultAnchorSlots;
constexpr uint32_t kMaxOpsPerRound = 4096u;

struct Round {
    AIVMRoundHandle handle{};
    AIVMRoundDescriptor desc{};

    id<MTLBuffer> desc_buf            = nil;
    id<MTLBuffer> att_ops_buf         = nil;
    id<MTLBuffer> model_ops_buf       = nil;
    id<MTLBuffer> anchor_ops_buf      = nil;
    id<MTLBuffer> attestations_buf    = nil;
    id<MTLBuffer> models_buf          = nil;
    id<MTLBuffer> anchors_buf         = nil;
    id<MTLBuffer> epoch_buf           = nil;
    id<MTLBuffer> result_buf          = nil;
    id<MTLBuffer> att_applied_buf     = nil;
    id<MTLBuffer> model_applied_buf   = nil;
    id<MTLBuffer> anchor_applied_buf  = nil;
    // v0.59 parallel-dispatch auxiliaries: track per-op slot assignment so
    // the writeback phase can run one thread per slot.
    id<MTLBuffer> att_op_slot_buf     = nil;  // u32[kMaxOpsPerRound] op->slot
    id<MTLBuffer> att_winner_buf      = nil;  // u32[kAttSlots] slot->winner_op
    id<MTLBuffer> model_op_slot_buf   = nil;  // u32[kMaxOpsPerRound]
};

class AIVMGPUEngineMetal final : public AIVMGPUEngine {
public:
    struct Pipelines {
        id<MTLComputePipelineState> att_locate;
        id<MTLComputePipelineState> att_writeback;
        id<MTLComputePipelineState> prov_locate;
        id<MTLComputePipelineState> anchor_apply;
        id<MTLComputePipelineState> epoch_transition;
    };

    AIVMGPUEngineMetal(id<MTLDevice> device,
                       id<MTLCommandQueue> queue,
                       Pipelines psos,
                       NSString* device_name)
        : device_(device)
        , queue_(queue)
        , psos_(psos)
        , device_name_str_([device_name UTF8String]) {}

    ~AIVMGPUEngineMetal() override {
        if (round_active()) end_round(round_.handle);
    }

    const char* device_name() const override { return device_name_str_.c_str(); }
    bool round_active() const override { return round_.handle.valid(); }

    AIVMRoundHandle begin_round(const AIVMRoundDescriptor& desc) override {
        std::lock_guard<std::mutex> g(mu_);
        if (round_.handle.valid()) return AIVMRoundHandle{0};

        round_ = Round{};
        round_.desc = desc;

        round_.desc_buf            = [device_ newBufferWithLength:sizeof(AIVMRoundDescriptor) options:MTLResourceStorageModeShared];
        round_.att_ops_buf         = [device_ newBufferWithLength:sizeof(AttestationOp) * kMaxOpsPerRound options:MTLResourceStorageModeShared];
        round_.model_ops_buf       = [device_ newBufferWithLength:sizeof(ModelOp) * kMaxOpsPerRound options:MTLResourceStorageModeShared];
        round_.anchor_ops_buf      = [device_ newBufferWithLength:sizeof(AnchorOp) * kMaxOpsPerRound options:MTLResourceStorageModeShared];
        round_.attestations_buf    = [device_ newBufferWithLength:sizeof(Attestation) * kAttSlots options:MTLResourceStorageModeShared];
        round_.models_buf          = [device_ newBufferWithLength:sizeof(ModelRegistryEntry) * kModelSlots options:MTLResourceStorageModeShared];
        round_.anchors_buf         = [device_ newBufferWithLength:sizeof(AuditAnchor) * kAnchorSlots options:MTLResourceStorageModeShared];
        round_.epoch_buf           = [device_ newBufferWithLength:sizeof(AIVMEpochState) options:MTLResourceStorageModeShared];
        round_.result_buf          = [device_ newBufferWithLength:sizeof(AIVMTransitionResult) options:MTLResourceStorageModeShared];
        round_.att_applied_buf     = [device_ newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        round_.model_applied_buf   = [device_ newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        round_.anchor_applied_buf  = [device_ newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        round_.att_op_slot_buf     = [device_ newBufferWithLength:sizeof(uint32_t) * kMaxOpsPerRound options:MTLResourceStorageModeShared];
        round_.att_winner_buf      = [device_ newBufferWithLength:sizeof(uint32_t) * kAttSlots options:MTLResourceStorageModeShared];
        round_.model_op_slot_buf   = [device_ newBufferWithLength:sizeof(uint32_t) * kMaxOpsPerRound options:MTLResourceStorageModeShared];

        if (!round_.desc_buf || !round_.att_ops_buf || !round_.model_ops_buf
            || !round_.anchor_ops_buf || !round_.attestations_buf || !round_.models_buf
            || !round_.anchors_buf || !round_.epoch_buf || !round_.result_buf
            || !round_.att_applied_buf || !round_.model_applied_buf
            || !round_.anchor_applied_buf
            || !round_.att_op_slot_buf || !round_.att_winner_buf
            || !round_.model_op_slot_buf)
            return AIVMRoundHandle{0};

        std::memset([round_.attestations_buf contents], 0, sizeof(Attestation) * kAttSlots);
        std::memset([round_.models_buf contents], 0, sizeof(ModelRegistryEntry) * kModelSlots);
        std::memset([round_.anchors_buf contents], 0, sizeof(AuditAnchor) * kAnchorSlots);
        std::memset([round_.epoch_buf contents], 0, sizeof(AIVMEpochState));
        std::memset([round_.result_buf contents], 0, sizeof(AIVMTransitionResult));
        *static_cast<uint32_t*>([round_.att_applied_buf contents]) = 0;
        *static_cast<uint32_t*>([round_.model_applied_buf contents]) = 0;
        *static_cast<uint32_t*>([round_.anchor_applied_buf contents]) = 0;
        // Sentinel-init slot/winner buffers to 0xFFFFFFFFu.
        std::memset([round_.att_op_slot_buf contents], 0xFF, sizeof(uint32_t) * kMaxOpsPerRound);
        std::memset([round_.att_winner_buf contents],  0xFF, sizeof(uint32_t) * kAttSlots);
        std::memset([round_.model_op_slot_buf contents], 0xFF, sizeof(uint32_t) * kMaxOpsPerRound);

        round_.desc.attestation_op_count = 0;
        round_.desc.model_op_count = 0;
        round_.desc.anchor_op_count = 0;

        round_.handle = AIVMRoundHandle{++next_handle_};
        return round_.handle;
    }

    void push_attestation_ops(AIVMRoundHandle h, std::span<const AttestationOp> ops) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h) || ops.empty()) return;
        auto* dst = static_cast<AttestationOp*>([round_.att_ops_buf contents]);
        uint32_t cap_left = kMaxOpsPerRound - round_.desc.attestation_op_count;
        uint32_t take = std::min<uint32_t>(uint32_t(ops.size()), cap_left);
        std::memcpy(dst + round_.desc.attestation_op_count,
                    ops.data(), take * sizeof(AttestationOp));
        round_.desc.attestation_op_count += take;
    }

    void push_model_ops(AIVMRoundHandle h, std::span<const ModelOp> ops) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h) || ops.empty()) return;
        auto* dst = static_cast<ModelOp*>([round_.model_ops_buf contents]);
        uint32_t cap_left = kMaxOpsPerRound - round_.desc.model_op_count;
        uint32_t take = std::min<uint32_t>(uint32_t(ops.size()), cap_left);
        std::memcpy(dst + round_.desc.model_op_count,
                    ops.data(), take * sizeof(ModelOp));
        round_.desc.model_op_count += take;
    }

    void push_anchor_ops(AIVMRoundHandle h, std::span<const AnchorOp> ops) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h) || ops.empty()) return;
        auto* dst = static_cast<AnchorOp*>([round_.anchor_ops_buf contents]);
        uint32_t cap_left = kMaxOpsPerRound - round_.desc.anchor_op_count;
        uint32_t take = std::min<uint32_t>(uint32_t(ops.size()), cap_left);
        std::memcpy(dst + round_.desc.anchor_op_count,
                    ops.data(), take * sizeof(AnchorOp));
        round_.desc.anchor_op_count += take;
    }

    AIVMTransitionResult run_epoch(AIVMRoundHandle h) override {
        return run_until_done(h, 1);
    }

    AIVMTransitionResult run_until_done(AIVMRoundHandle h, std::size_t /*max_epochs*/) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h)) return AIVMTransitionResult{};

        std::memcpy([round_.desc_buf contents], &round_.desc, sizeof(AIVMRoundDescriptor));

        id<MTLCommandBuffer> cmd = [queue_ commandBuffer];

        auto dispatch_n = [&](id<MTLComputePipelineState> pso,
                              uint32_t n,
                              void(^bind)(id<MTLComputeCommandEncoder>)) {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso];
            bind(enc);
            // Threadgroup width 256 (clean multiple of Apple SIMD width 32).
            // Cap at the pipeline's max threads / N.
            NSUInteger tg = std::min<NSUInteger>(256u,
                std::min<NSUInteger>(pso.maxTotalThreadsPerThreadgroup, n));
            if (tg == 0) tg = 1;
            [enc dispatchThreads:MTLSizeMake(n, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
            [enc endEncoding];
        };
        auto dispatch_1 = [&](id<MTLComputePipelineState> pso,
                              void(^bind)(id<MTLComputeCommandEncoder>)) {
            dispatch_n(pso, 1u, bind);
        };

        uint32_t att_count_v    = kAttSlots;
        uint32_t model_count_v  = kModelSlots;
        uint32_t anchor_count_v = kAnchorSlots;

        auto mode = static_cast<AIVMTransitionMode>(round_.desc.mode);

        if (mode == AIVMTransitionMode::AttestationApply ||
            mode == AIVMTransitionMode::FullRound) {
            // Reset auxiliary buffers (slot/winner) for this run. The arena
            // is already zero from begin_round; intermediate state stays
            // sentinel-initialised between begin_round and run_until_done.
            std::memset([round_.att_op_slot_buf contents], 0xFF,
                        sizeof(uint32_t) * kMaxOpsPerRound);
            std::memset([round_.att_winner_buf contents], 0xFF,
                        sizeof(uint32_t) * kAttSlots);
            // Phase 1 — single thread, canonical-order locate.
            dispatch_1(psos_.att_locate, ^(id<MTLComputeCommandEncoder> enc) {
                [enc setBuffer:round_.desc_buf         offset:0 atIndex:0];
                [enc setBuffer:round_.att_ops_buf      offset:0 atIndex:1];
                [enc setBuffer:round_.attestations_buf offset:0 atIndex:2];
                [enc setBuffer:round_.att_applied_buf  offset:0 atIndex:3];
                [enc setBytes:&att_count_v length:sizeof(att_count_v) atIndex:4];
                [enc setBuffer:round_.att_op_slot_buf  offset:0 atIndex:5];
                [enc setBuffer:round_.att_winner_buf   offset:0 atIndex:6];
            });
            // Phase 2 — parallel writeback, one thread per slot.
            dispatch_n(psos_.att_writeback, kAttSlots, ^(id<MTLComputeCommandEncoder> enc) {
                [enc setBuffer:round_.desc_buf         offset:0 atIndex:0];
                [enc setBuffer:round_.att_ops_buf      offset:0 atIndex:1];
                [enc setBuffer:round_.attestations_buf offset:0 atIndex:2];
                [enc setBytes:&att_count_v length:sizeof(att_count_v) atIndex:4];
                [enc setBuffer:round_.att_winner_buf   offset:0 atIndex:6];
            });
        }
        if (mode == AIVMTransitionMode::ProvenanceApply ||
            mode == AIVMTransitionMode::FullRound) {
            std::memset([round_.model_op_slot_buf contents], 0xFF,
                        sizeof(uint32_t) * kMaxOpsPerRound);
            // ProvenanceApply has order-dependent semantics (UpdateWeights
            // increments a version counter); Phase 1 does the full work
            // single-threaded. No Phase 2 writeback.
            dispatch_1(psos_.prov_locate, ^(id<MTLComputeCommandEncoder> enc) {
                [enc setBuffer:round_.desc_buf          offset:0 atIndex:0];
                [enc setBuffer:round_.model_ops_buf     offset:0 atIndex:1];
                [enc setBuffer:round_.models_buf        offset:0 atIndex:2];
                [enc setBuffer:round_.model_applied_buf offset:0 atIndex:3];
                [enc setBytes:&model_count_v length:sizeof(model_count_v) atIndex:4];
                [enc setBuffer:round_.model_op_slot_buf offset:0 atIndex:5];
            });
        }
        if (mode == AIVMTransitionMode::AnchorApply ||
            mode == AIVMTransitionMode::FullRound) {
            // Anchor application stays single-threaded — chain integrity
            // (parent_root == previous.commit_root) is fundamentally
            // sequential. Throughput is still helped by EpochTransition
            // parallelism downstream.
            dispatch_1(psos_.anchor_apply, ^(id<MTLComputeCommandEncoder> enc) {
                [enc setBuffer:round_.desc_buf           offset:0 atIndex:0];
                [enc setBuffer:round_.anchor_ops_buf     offset:0 atIndex:1];
                [enc setBuffer:round_.anchors_buf        offset:0 atIndex:2];
                [enc setBuffer:round_.anchor_applied_buf offset:0 atIndex:3];
                [enc setBytes:&anchor_count_v length:sizeof(anchor_count_v) atIndex:4];
            });
        }

        // EpochTransition — same single-thread kernel as v0.58.3. Trying to
        // parallelise across multiple kernel dispatches added per-dispatch
        // latency that exceeded the parallel work savings on the M1 Max
        // integrated GPU. The leaf-hash work is small enough that a single
        // thread completes in ~2 ms, dominated by dispatch overhead anyway.
        dispatch_1(psos_.epoch_transition, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:round_.desc_buf         offset:0 atIndex:0];
            [enc setBuffer:round_.attestations_buf offset:0 atIndex:1];
            [enc setBuffer:round_.models_buf       offset:0 atIndex:2];
            [enc setBuffer:round_.anchors_buf      offset:0 atIndex:3];
            [enc setBuffer:round_.epoch_buf        offset:0 atIndex:4];
            [enc setBuffer:round_.result_buf       offset:0 atIndex:5];
            [enc setBytes:&att_count_v    length:sizeof(att_count_v)    atIndex:6];
            [enc setBytes:&model_count_v  length:sizeof(model_count_v)  atIndex:7];
            [enc setBytes:&anchor_count_v length:sizeof(anchor_count_v) atIndex:8];
        });

        [cmd commit];
        [cmd waitUntilCompleted];

        auto* result = static_cast<AIVMTransitionResult*>([round_.result_buf contents]);
        uint32_t a_app = *static_cast<uint32_t*>([round_.att_applied_buf contents]);
        uint32_t m_app = *static_cast<uint32_t*>([round_.model_applied_buf contents]);
        uint32_t n_app = *static_cast<uint32_t*>([round_.anchor_applied_buf contents]);
        result->attestation_apply_count = a_app;
        result->model_apply_count       = m_app;
        result->anchor_apply_count      = n_app;
        return *result;
    }

    AIVMTransitionResult poll_round_result(AIVMRoundHandle h) const override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle_const(h)) return AIVMTransitionResult{};
        return *static_cast<const AIVMTransitionResult*>([round_.result_buf contents]);
    }

    void end_round(AIVMRoundHandle h) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h)) return;
        round_ = Round{};
    }

private:
    bool check_handle(AIVMRoundHandle h) const {
        return h.valid() && h.opaque == round_.handle.opaque;
    }
    bool check_handle_const(AIVMRoundHandle h) const { return check_handle(h); }

    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;
    Pipelines psos_;
    std::string device_name_str_;
    Round round_;
    uint64_t next_handle_ = 0;
    mutable std::mutex mu_;
};

}  // namespace

std::unique_ptr<AIVMGPUEngine> AIVMGPUEngine::create() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return nullptr;
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) return nullptr;

        id<MTLLibrary> lib = load_aivm_metallib(device);
        if (!lib) lib = compile_aivm_library(device);
        if (!lib) return nullptr;

        NSError* err = nil;
        auto fn = [&](NSString* name) -> id<MTLComputePipelineState> {
            id<MTLFunction> f = [lib newFunctionWithName:name];
            if (!f) {
                std::fprintf(stderr, "AIVM kernel %s not found in library\n",
                             [name UTF8String]);
                return nil;
            }
            id<MTLComputePipelineState> p = [device newComputePipelineStateWithFunction:f error:&err];
            if (!p && err)
                std::fprintf(stderr, "PSO compile error for %s: %s\n",
                             [name UTF8String], [[err localizedDescription] UTF8String]);
            return p;
        };
        AIVMGPUEngineMetal::Pipelines psos{};
        psos.att_locate       = fn(@"aivm_attestation_locate");
        psos.att_writeback    = fn(@"aivm_attestation_writeback");
        psos.prov_locate      = fn(@"aivm_provenance_locate");
        psos.anchor_apply     = fn(@"aivm_anchor_apply");
        psos.epoch_transition = fn(@"aivm_epoch_transition");
        if (!psos.att_locate || !psos.att_writeback
            || !psos.prov_locate || !psos.anchor_apply
            || !psos.epoch_transition)
            return nullptr;

        return std::unique_ptr<AIVMGPUEngine>(
            new AIVMGPUEngineMetal(device, queue, psos, [device name]));
    }
}

}  // namespace aivm::gpu
