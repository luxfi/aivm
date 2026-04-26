// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file aivm_gpu_engine.hpp
/// AIVMGPUEngine — GPU-native A-Chain transition substrate.
///
/// Lifecycle (mirrors PVMGPUEngine and QuasarGPUEngine):
///   begin_round(AIVMRoundDescriptor)
///   push_attestation_ops / push_model_ops / push_anchor_ops
///   run_epoch / run_until_done
///   poll_round_result  -> AIVMTransitionResult (with aivm_state_root)
///   end_round
///
/// AIVMTransitionResult.aivm_state_root binds attestation_root,
/// model_registry_root, and audit_root for a given epoch — this is the
/// linkage that makes the A-Chain GPU-native under LP-137.

#pragma once

#include "aivm_gpu_layout.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace aivm::gpu {

struct AIVMRoundHandle {
    uint64_t opaque = 0;
    bool valid() const { return opaque != 0; }
};

class AIVMGPUEngine {
public:
    virtual ~AIVMGPUEngine() = default;

    static std::unique_ptr<AIVMGPUEngine> create();

    virtual AIVMRoundHandle begin_round(const AIVMRoundDescriptor& desc) = 0;

    virtual void push_attestation_ops(AIVMRoundHandle h,
                                      std::span<const AttestationOp> ops) = 0;
    virtual void push_model_ops(AIVMRoundHandle h,
                                std::span<const ModelOp> ops) = 0;
    virtual void push_anchor_ops(AIVMRoundHandle h,
                                 std::span<const AnchorOp> ops) = 0;

    virtual AIVMTransitionResult run_epoch(AIVMRoundHandle h) = 0;
    virtual AIVMTransitionResult run_until_done(AIVMRoundHandle h,
                                                std::size_t max_epochs = 64) = 0;
    virtual AIVMTransitionResult poll_round_result(AIVMRoundHandle h) const = 0;

    virtual void end_round(AIVMRoundHandle h) = 0;

    virtual bool round_active() const = 0;
    virtual const char* device_name() const = 0;

protected:
    AIVMGPUEngine() = default;
    AIVMGPUEngine(const AIVMGPUEngine&) = delete;
    AIVMGPUEngine& operator=(const AIVMGPUEngine&) = delete;
};

}  // namespace aivm::gpu
