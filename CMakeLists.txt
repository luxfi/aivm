# Copyright (C) 2026, Lux Partners Limited. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# AIVM (AI-VM, A-Chain) GPU substrate.
# v0.58: A-Chain transition kernels — CUDA + Metal + WGSL with 4-way
# determinism (CPU reference oracle + Metal + CUDA + WGSL byte-equivalence).

cmake_minimum_required(VERSION 3.20)

project(lux-aivm
    VERSION 0.58.0
    DESCRIPTION "GPU-native A-Chain transition substrate (attestation, AI provenance, audit anchors)"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(LUX_AIVM_BUILD_TESTS "Build AIVM tests" ON)
option(LUX_AIVM_ENABLE_METAL "Build Metal backend (Apple)" ${APPLE})
option(LUX_AIVM_ENABLE_CUDA  "Build CUDA backend" OFF)
option(LUX_AIVM_ENABLE_WGPU  "Build WebGPU/WGSL backend" OFF)

# ---- core library (host code: layout + CPU reference) ---------------------

add_library(lux_aivm_core STATIC
    src/aivm_cpu_reference.cpp
)

target_include_directories(lux_aivm_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(lux_aivm_core PUBLIC cxx_std_20)

set_target_properties(lux_aivm_core PROPERTIES
    OUTPUT_NAME "lux_aivm_core"
    POSITION_INDEPENDENT_CODE ON
)

# ---- Metal driver (Apple) -------------------------------------------------

if(LUX_AIVM_ENABLE_METAL AND APPLE)
    add_library(lux_aivm_metal STATIC
        src/aivm_gpu_engine.mm
    )
    target_include_directories(lux_aivm_metal PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    )
    target_compile_features(lux_aivm_metal PUBLIC cxx_std_20)
    target_link_libraries(lux_aivm_metal PUBLIC lux_aivm_core
        "-framework Metal" "-framework Foundation")
    set_target_properties(lux_aivm_metal PROPERTIES
        OUTPUT_NAME "lux_aivm_metal"
        POSITION_INDEPENDENT_CODE ON
    )

    # Stage the .metal sources for runtime fallback.
    foreach(src
        aivm_kernels_common.h.metal
        aivm_attestation.metal
        aivm_provenance.metal
        aivm_anchor.metal
        aivm_transition.metal)
        configure_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/src/${src}
            ${CMAKE_CURRENT_BINARY_DIR}/src/${src}
            COPYONLY)
    endforeach()

    # Pre-compile a unified .metallib so the runtime can skip the online
    # MSL compiler entirely (more reliable than newLibraryWithSource for
    # large kernels). The combined source is the same one the runtime
    # would build by concatenation: common header + four kernels with
    # the include-line stripped.
    set(AIVM_METAL_COMBINED ${CMAKE_CURRENT_BINARY_DIR}/src/aivm_combined.metal)
    add_custom_command(
        OUTPUT ${AIVM_METAL_COMBINED}
        COMMAND bash -c "cat ${CMAKE_CURRENT_SOURCE_DIR}/src/aivm_kernels_common.h.metal > ${AIVM_METAL_COMBINED} && for f in aivm_attestation.metal aivm_provenance.metal aivm_anchor.metal aivm_transition.metal; do grep -v 'aivm_kernels_common.h.metal' ${CMAKE_CURRENT_SOURCE_DIR}/src/$f >> ${AIVM_METAL_COMBINED}; done"
        DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/src/aivm_kernels_common.h.metal
            ${CMAKE_CURRENT_SOURCE_DIR}/src/aivm_attestation.metal
            ${CMAKE_CURRENT_SOURCE_DIR}/src/aivm_provenance.metal
            ${CMAKE_CURRENT_SOURCE_DIR}/src/aivm_anchor.metal
            ${CMAKE_CURRENT_SOURCE_DIR}/src/aivm_transition.metal
        VERBATIM)

    set(AIVM_METAL_AIR ${CMAKE_CURRENT_BINARY_DIR}/src/aivm_combined.air)
    set(AIVM_METAL_LIB ${CMAKE_CURRENT_BINARY_DIR}/src/aivm.metallib)

    add_custom_command(
        OUTPUT ${AIVM_METAL_AIR}
        COMMAND xcrun -sdk macosx metal -std=metal3.0
            -Wno-pragma-once-outside-header
            -Wno-unused-const-variable
            -c ${AIVM_METAL_COMBINED} -o ${AIVM_METAL_AIR}
        DEPENDS ${AIVM_METAL_COMBINED}
        VERBATIM)

    add_custom_command(
        OUTPUT ${AIVM_METAL_LIB}
        COMMAND xcrun -sdk macosx metallib ${AIVM_METAL_AIR} -o ${AIVM_METAL_LIB}
        DEPENDS ${AIVM_METAL_AIR}
        VERBATIM)

    add_custom_target(aivm_metallib ALL DEPENDS ${AIVM_METAL_LIB})
    add_dependencies(lux_aivm_metal aivm_metallib)
endif()

# ---- CUDA driver ----------------------------------------------------------

if(LUX_AIVM_ENABLE_CUDA)
    enable_language(CUDA)
    add_library(lux_aivm_cuda STATIC
        src/aivm_gpu_engine_cuda.cpp
        src/aivm_cuda_launchers.cu
        src/aivm_attestation.cu
        src/aivm_provenance.cu
        src/aivm_anchor.cu
        src/aivm_transition.cu
    )
    target_include_directories(lux_aivm_cuda PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    )
    set_target_properties(lux_aivm_cuda PROPERTIES
        CUDA_STANDARD 17
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        OUTPUT_NAME "lux_aivm_cuda"
    )
    target_link_libraries(lux_aivm_cuda PUBLIC lux_aivm_core)
endif()

# ---- WGPU driver ---------------------------------------------------------

if(LUX_AIVM_ENABLE_WGPU)
    add_library(lux_aivm_wgpu STATIC
        src/aivm_gpu_engine_wgpu.cpp
    )
    target_include_directories(lux_aivm_wgpu PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    )
    target_compile_features(lux_aivm_wgpu PUBLIC cxx_std_20)
    target_link_libraries(lux_aivm_wgpu PUBLIC lux_aivm_core)
    set_target_properties(lux_aivm_wgpu PROPERTIES
        OUTPUT_NAME "lux_aivm_wgpu"
        POSITION_INDEPENDENT_CODE ON
    )

    # Stage the WGSL sources for runtime use.
    foreach(src
        aivm_kernels_common.wgsl
        aivm_attestation.wgsl
        aivm_provenance.wgsl
        aivm_anchor.wgsl
        aivm_transition.wgsl)
        configure_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/src/${src}
            ${CMAKE_CURRENT_BINARY_DIR}/src/${src}
            COPYONLY)
    endforeach()
endif()

# ---- tests ----------------------------------------------------------------

if(LUX_AIVM_BUILD_TESTS)
    enable_testing()

    # Layout test — host-only.
    add_executable(aivm-layout-test test/aivm_layout_test.cpp)
    target_link_libraries(aivm-layout-test PRIVATE lux_aivm_core)
    target_compile_features(aivm-layout-test PRIVATE cxx_std_20)
    add_test(NAME aivm-layout-test COMMAND aivm-layout-test)

    # GPU engine test — Apple-only (.mm).
    if(LUX_AIVM_ENABLE_METAL AND APPLE)
        add_executable(aivm-gpu-engine-test test/aivm_gpu_engine_test.mm)
        target_link_libraries(aivm-gpu-engine-test PRIVATE lux_aivm_metal)
        target_compile_features(aivm-gpu-engine-test PRIVATE cxx_std_20)
        add_test(NAME aivm-gpu-engine-test COMMAND aivm-gpu-engine-test)
    endif()

    # Cross-backend determinism — links to whichever GPU backend is enabled.
    add_executable(aivm-determinism-test test/aivm_determinism_test.cpp)
    target_compile_features(aivm-determinism-test PRIVATE cxx_std_20)
    if(LUX_AIVM_ENABLE_METAL AND APPLE)
        target_link_libraries(aivm-determinism-test PRIVATE lux_aivm_metal)
    elseif(LUX_AIVM_ENABLE_CUDA)
        target_link_libraries(aivm-determinism-test PRIVATE lux_aivm_cuda)
    else()
        target_sources(aivm-determinism-test PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src/aivm_engine_stub.cpp)
        target_link_libraries(aivm-determinism-test PRIVATE lux_aivm_core)
    endif()
    add_test(NAME aivm-determinism-test COMMAND aivm-determinism-test)
endif()

message(STATUS "")
message(STATUS "=== Lux AIVM Configuration ===")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build Tests: ${LUX_AIVM_BUILD_TESTS}")
message(STATUS "  Metal:       ${LUX_AIVM_ENABLE_METAL}")
message(STATUS "  CUDA:        ${LUX_AIVM_ENABLE_CUDA}")
message(STATUS "  WGPU:        ${LUX_AIVM_ENABLE_WGPU}")
message(STATUS "")
