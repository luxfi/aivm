// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// aivm_gpu_engine_wgpu.cpp — WebGPU/wgpu-native driver entry point.
//
// The full WGPU host-side driver requires a real wgpu/Dawn runtime which is
// gated behind LUX_HAS_WEBGPU in the build system (see ~/work/luxcpp/lux-webgpu).
// At v0.58 we ship the four WGSL kernels and a determinism contract that
// matches CPU/Metal/CUDA byte-for-byte; the host driver here exposes a
// factory hook that the platform driver can call when WGPU is available.
//
// When LUX_HAS_WEBGPU is not defined, this file is still compiled into the
// determinism test binary so the linker has a no-op factory. The Apple
// Metal driver (or the CUDA driver on Linux) is the canonical engine the
// determinism harness compares against; WGSL byte-equivalence is verified
// at the kernel-source level (the WGSL leaf encoding matches CPU/Metal/CUDA
// by construction; see aivm_kernels_common.wgsl).

#include "lux/aivm/aivm_gpu_engine.hpp"

namespace aivm::gpu {

std::unique_ptr<AIVMGPUEngine> create_aivm_gpu_engine_wgpu()
{
    // No wgpu runtime linked: return nullptr so the platform engine factory
    // falls through to Metal/CUDA. When LUX_HAS_WEBGPU is wired, this becomes
    // the wgpu host driver constructor.
    return nullptr;
}

}  // namespace aivm::gpu
