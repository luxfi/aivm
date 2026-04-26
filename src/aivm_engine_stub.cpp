// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// aivm_engine_stub.cpp — CPU-only fallback for AIVMGPUEngine::create().
// Linked when neither Metal nor CUDA is enabled, so tests still build.
// The stub returns nullptr, exercising the CPU-only test paths.

#include "lux/aivm/aivm_gpu_engine.hpp"

namespace aivm::gpu {

#if !defined(__APPLE__)
std::unique_ptr<AIVMGPUEngine> AIVMGPUEngine::create() {
    return nullptr;
}
#else
__attribute__((weak)) std::unique_ptr<AIVMGPUEngine> AIVMGPUEngine::create() {
    return nullptr;
}
#endif

}  // namespace aivm::gpu
