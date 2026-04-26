// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// aivm_gpu_engine_cuda.cpp — CUDA-backed driver for AIVMGPUEngine.

#include "lux/aivm/aivm_gpu_engine.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

namespace aivm::gpu {

extern void launch_aivm_attestation_apply(
    const AIVMRoundDescriptor*, const AttestationOp*,
    Attestation*, uint32_t*, uint32_t);
extern void launch_aivm_provenance_apply(
    const AIVMRoundDescriptor*, const ModelOp*,
    ModelRegistryEntry*, uint32_t*, uint32_t);
extern void launch_aivm_anchor_apply(
    const AIVMRoundDescriptor*, const AnchorOp*,
    AuditAnchor*, uint32_t*, uint32_t);
extern void launch_aivm_epoch_transition(
    const AIVMRoundDescriptor*, Attestation*, ModelRegistryEntry*,
    AuditAnchor*, AIVMEpochState*, AIVMTransitionResult*,
    uint32_t, uint32_t, uint32_t);

namespace {

constexpr uint32_t kAttSlots       = kDefaultAttestationSlots;
constexpr uint32_t kModelSlots     = kDefaultModelSlots;
constexpr uint32_t kAnchorSlots    = kDefaultAnchorSlots;
constexpr uint32_t kMaxOpsPerRound = 4096u;

struct Round {
    AIVMRoundHandle handle{};
    AIVMRoundDescriptor desc{};

    AIVMRoundDescriptor* d_desc        = nullptr;
    AttestationOp*       d_att_ops     = nullptr;
    ModelOp*             d_model_ops   = nullptr;
    AnchorOp*            d_anchor_ops  = nullptr;
    Attestation*         d_attestations = nullptr;
    ModelRegistryEntry*  d_models      = nullptr;
    AuditAnchor*         d_anchors     = nullptr;
    AIVMEpochState*      d_epoch       = nullptr;
    AIVMTransitionResult* d_result     = nullptr;
    uint32_t*            d_att_applied = nullptr;
    uint32_t*            d_model_applied = nullptr;
    uint32_t*            d_anchor_applied = nullptr;
};

class AIVMGPUEngineCuda final : public AIVMGPUEngine {
public:
    AIVMGPUEngineCuda() {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
            device_name_str_ = prop.name;
        } else {
            device_name_str_ = "cuda";
        }
    }
    ~AIVMGPUEngineCuda() override {
        if (round_active()) end_round(round_.handle);
    }

    const char* device_name() const override { return device_name_str_.c_str(); }
    bool round_active() const override { return round_.handle.valid(); }

    AIVMRoundHandle begin_round(const AIVMRoundDescriptor& desc) override {
        std::lock_guard<std::mutex> g(mu_);
        if (round_.handle.valid()) return AIVMRoundHandle{0};
        round_ = Round{};
        round_.desc = desc;
        round_.desc.attestation_op_count = 0;
        round_.desc.model_op_count = 0;
        round_.desc.anchor_op_count = 0;

        auto alloc = [](void** p, size_t n) -> bool {
            if (cudaMalloc(p, n) != cudaSuccess) return false;
            return cudaMemset(*p, 0, n) == cudaSuccess;
        };

        if (!alloc((void**)&round_.d_desc, sizeof(AIVMRoundDescriptor))
            || !alloc((void**)&round_.d_att_ops, sizeof(AttestationOp) * kMaxOpsPerRound)
            || !alloc((void**)&round_.d_model_ops, sizeof(ModelOp) * kMaxOpsPerRound)
            || !alloc((void**)&round_.d_anchor_ops, sizeof(AnchorOp) * kMaxOpsPerRound)
            || !alloc((void**)&round_.d_attestations, sizeof(Attestation) * kAttSlots)
            || !alloc((void**)&round_.d_models, sizeof(ModelRegistryEntry) * kModelSlots)
            || !alloc((void**)&round_.d_anchors, sizeof(AuditAnchor) * kAnchorSlots)
            || !alloc((void**)&round_.d_epoch, sizeof(AIVMEpochState))
            || !alloc((void**)&round_.d_result, sizeof(AIVMTransitionResult))
            || !alloc((void**)&round_.d_att_applied, sizeof(uint32_t))
            || !alloc((void**)&round_.d_model_applied, sizeof(uint32_t))
            || !alloc((void**)&round_.d_anchor_applied, sizeof(uint32_t))) {
            return AIVMRoundHandle{0};
        }

        round_.handle = AIVMRoundHandle{++next_handle_};
        return round_.handle;
    }

    void push_attestation_ops(AIVMRoundHandle h, std::span<const AttestationOp> ops) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h) || ops.empty()) return;
        uint32_t cap_left = kMaxOpsPerRound - round_.desc.attestation_op_count;
        uint32_t take = std::min<uint32_t>(uint32_t(ops.size()), cap_left);
        cudaMemcpy(round_.d_att_ops + round_.desc.attestation_op_count,
                   ops.data(), take * sizeof(AttestationOp), cudaMemcpyHostToDevice);
        round_.desc.attestation_op_count += take;
    }

    void push_model_ops(AIVMRoundHandle h, std::span<const ModelOp> ops) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h) || ops.empty()) return;
        uint32_t cap_left = kMaxOpsPerRound - round_.desc.model_op_count;
        uint32_t take = std::min<uint32_t>(uint32_t(ops.size()), cap_left);
        cudaMemcpy(round_.d_model_ops + round_.desc.model_op_count,
                   ops.data(), take * sizeof(ModelOp), cudaMemcpyHostToDevice);
        round_.desc.model_op_count += take;
    }

    void push_anchor_ops(AIVMRoundHandle h, std::span<const AnchorOp> ops) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h) || ops.empty()) return;
        uint32_t cap_left = kMaxOpsPerRound - round_.desc.anchor_op_count;
        uint32_t take = std::min<uint32_t>(uint32_t(ops.size()), cap_left);
        cudaMemcpy(round_.d_anchor_ops + round_.desc.anchor_op_count,
                   ops.data(), take * sizeof(AnchorOp), cudaMemcpyHostToDevice);
        round_.desc.anchor_op_count += take;
    }

    AIVMTransitionResult run_epoch(AIVMRoundHandle h) override {
        return run_until_done(h, 1);
    }

    AIVMTransitionResult run_until_done(AIVMRoundHandle h, std::size_t /*max_epochs*/) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h)) return AIVMTransitionResult{};

        cudaMemcpy(round_.d_desc, &round_.desc, sizeof(AIVMRoundDescriptor),
                   cudaMemcpyHostToDevice);

        auto mode = static_cast<AIVMTransitionMode>(round_.desc.mode);
        if (mode == AIVMTransitionMode::AttestationApply ||
            mode == AIVMTransitionMode::FullRound) {
            launch_aivm_attestation_apply(
                round_.d_desc, round_.d_att_ops, round_.d_attestations,
                round_.d_att_applied, kAttSlots);
        }
        if (mode == AIVMTransitionMode::ProvenanceApply ||
            mode == AIVMTransitionMode::FullRound) {
            launch_aivm_provenance_apply(
                round_.d_desc, round_.d_model_ops, round_.d_models,
                round_.d_model_applied, kModelSlots);
        }
        if (mode == AIVMTransitionMode::AnchorApply ||
            mode == AIVMTransitionMode::FullRound) {
            launch_aivm_anchor_apply(
                round_.d_desc, round_.d_anchor_ops, round_.d_anchors,
                round_.d_anchor_applied, kAnchorSlots);
        }
        launch_aivm_epoch_transition(
            round_.d_desc, round_.d_attestations, round_.d_models,
            round_.d_anchors, round_.d_epoch, round_.d_result,
            kAttSlots, kModelSlots, kAnchorSlots);

        cudaDeviceSynchronize();

        AIVMTransitionResult result{};
        cudaMemcpy(&result, round_.d_result, sizeof(AIVMTransitionResult),
                   cudaMemcpyDeviceToHost);
        uint32_t a_app = 0, m_app = 0, n_app = 0;
        cudaMemcpy(&a_app, round_.d_att_applied,    sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&m_app, round_.d_model_applied,  sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&n_app, round_.d_anchor_applied, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        result.attestation_apply_count = a_app;
        result.model_apply_count       = m_app;
        result.anchor_apply_count      = n_app;
        return result;
    }

    AIVMTransitionResult poll_round_result(AIVMRoundHandle h) const override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle_const(h)) return AIVMTransitionResult{};
        AIVMTransitionResult result{};
        cudaMemcpy(&result, round_.d_result, sizeof(AIVMTransitionResult),
                   cudaMemcpyDeviceToHost);
        return result;
    }

    void end_round(AIVMRoundHandle h) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h)) return;
        cudaFree(round_.d_desc);
        cudaFree(round_.d_att_ops);
        cudaFree(round_.d_model_ops);
        cudaFree(round_.d_anchor_ops);
        cudaFree(round_.d_attestations);
        cudaFree(round_.d_models);
        cudaFree(round_.d_anchors);
        cudaFree(round_.d_epoch);
        cudaFree(round_.d_result);
        cudaFree(round_.d_att_applied);
        cudaFree(round_.d_model_applied);
        cudaFree(round_.d_anchor_applied);
        round_ = Round{};
    }

private:
    bool check_handle(AIVMRoundHandle h) const {
        return h.valid() && h.opaque == round_.handle.opaque;
    }
    bool check_handle_const(AIVMRoundHandle h) const { return check_handle(h); }

    std::string device_name_str_;
    Round round_;
    uint64_t next_handle_ = 0;
    mutable std::mutex mu_;
};

}  // namespace

std::unique_ptr<AIVMGPUEngine> create_aivm_gpu_engine_cuda()
{
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n <= 0) return nullptr;
    return std::unique_ptr<AIVMGPUEngine>(new AIVMGPUEngineCuda());
}

#if !defined(__APPLE__)
std::unique_ptr<AIVMGPUEngine> AIVMGPUEngine::create() {
    return create_aivm_gpu_engine_cuda();
}
#endif

}  // namespace aivm::gpu
