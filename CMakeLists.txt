# Copyright (C) 2026, Lux Partners Limited. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# AIVM (AI-VM, A-Chain) GPU substrate.
# v0.58: A-Chain transition kernels — CUDA + Metal + WGSL with 4-way
# determinism (CPU reference oracle + Metal + CUDA + WGSL byte-equivalence).

cmake_minimum_required(VERSION 3.20)

project(lux-aivm
    VERSION 0.59.0
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
    src/ai_precompile.cpp
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
        src/ai_precompile_metal.mm
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

    # Embed the WGSL kernel sources as raw-string literals so the driver has
    # no runtime filesystem dependency. Generates aivm_wgsl_sources.h.
    set(_AIVM_WGSL_HEADER ${CMAKE_CURRENT_BINARY_DIR}/src/aivm_wgsl_sources.h)
    set(_AIVM_WGSL_FILES
        aivm_kernels_common.wgsl
        aivm_attestation.wgsl
        aivm_provenance.wgsl
        aivm_anchor.wgsl
        aivm_transition.wgsl)
    set(_AIVM_WGSL_NAMES
        Common Attestation Provenance Anchor Transition)
    file(WRITE ${_AIVM_WGSL_HEADER}
        "// Auto-generated by CMakeLists.txt — do not edit.\n"
        "// Source: src/aivm_*.wgsl\n"
        "#pragma once\n\n")
    list(LENGTH _AIVM_WGSL_FILES _N)
    math(EXPR _LAST "${_N} - 1")
    foreach(i RANGE 0 ${_LAST})
        list(GET _AIVM_WGSL_FILES ${i} _SRC)
        list(GET _AIVM_WGSL_NAMES ${i} _NAME)
        file(READ ${CMAKE_CURRENT_SOURCE_DIR}/src/${_SRC} _CONTENT)
        file(APPEND ${_AIVM_WGSL_HEADER}
            "// ---- ${_SRC} ----\n"
            "static constexpr char kAIVMWGSL_${_NAME}[] = R\"AIVMWGSL(\n${_CONTENT}\n)AIVMWGSL\";\n\n")
    endforeach()

    target_include_directories(lux_aivm_wgpu PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}/src)

    # Locate WebGPU runtime — Dawn first, then wgpu-native (Homebrew flat layout).
    set(_aivm_webgpu_found FALSE)
    find_package(Dawn QUIET)
    if(Dawn_FOUND)
        target_link_libraries(lux_aivm_wgpu PRIVATE Dawn::dawn)
        target_compile_definitions(lux_aivm_wgpu PRIVATE
            LUX_AIVM_HAS_WEBGPU=1)
        set(_aivm_webgpu_found TRUE)
        message(STATUS "lux_aivm_wgpu: linking against Dawn")
    endif()
    if(NOT _aivm_webgpu_found)
        find_path(_AIVM_WGPU_INCLUDE webgpu/webgpu.h
            HINTS /opt/homebrew/include /usr/local/include /usr/include)
        if(NOT _AIVM_WGPU_INCLUDE)
            find_path(_AIVM_WGPU_FLAT_INCLUDE webgpu.h
                HINTS /opt/homebrew/include /usr/local/include /usr/include)
        endif()
        find_library(_AIVM_WGPU_LIB NAMES wgpu_native wgpu
            HINTS /opt/homebrew/lib /usr/local/lib /usr/lib)
        if((_AIVM_WGPU_INCLUDE OR _AIVM_WGPU_FLAT_INCLUDE) AND _AIVM_WGPU_LIB)
            if(_AIVM_WGPU_INCLUDE)
                target_include_directories(lux_aivm_wgpu PRIVATE ${_AIVM_WGPU_INCLUDE})
            else()
                target_include_directories(lux_aivm_wgpu PRIVATE ${_AIVM_WGPU_FLAT_INCLUDE})
            endif()
            target_link_libraries(lux_aivm_wgpu PRIVATE ${_AIVM_WGPU_LIB})
            target_compile_definitions(lux_aivm_wgpu PRIVATE
                LUX_AIVM_HAS_WEBGPU=1
                LUX_AIVM_HAS_WGPU_NATIVE=1)
            set(_aivm_webgpu_found TRUE)
            message(STATUS "lux_aivm_wgpu: linking against wgpu-native (${_AIVM_WGPU_LIB})")
        endif()
    endif()
    if(NOT _aivm_webgpu_found)
        message(STATUS "lux_aivm_wgpu: no WebGPU runtime found — driver compiles to stub factory")
    endif()
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
    if(LUX_AIVM_ENABLE_WGPU)
        target_link_libraries(aivm-determinism-test PRIVATE lux_aivm_wgpu)
        target_compile_definitions(aivm-determinism-test PRIVATE
            LUX_AIVM_TEST_WGPU=1)
    endif()
    add_test(NAME aivm-determinism-test COMMAND aivm-determinism-test)

    # Wall-clock benchmark — built at -O3 -DNDEBUG regardless of outer
    # CMAKE_BUILD_TYPE so the numbers reflect optimised kernels. Links to
    # whichever backends are enabled, exactly like the determinism harness.
    add_executable(aivm-benchmark test/aivm_benchmark.cpp)
    target_compile_features(aivm-benchmark PRIVATE cxx_std_20)
    target_compile_options(aivm-benchmark PRIVATE -O3 -DNDEBUG)
    if(LUX_AIVM_ENABLE_METAL AND APPLE)
        target_link_libraries(aivm-benchmark PRIVATE lux_aivm_metal)
    elseif(LUX_AIVM_ENABLE_CUDA)
        target_link_libraries(aivm-benchmark PRIVATE lux_aivm_cuda)
    else()
        target_sources(aivm-benchmark PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src/aivm_engine_stub.cpp)
        target_link_libraries(aivm-benchmark PRIVATE lux_aivm_core)
    endif()
    if(LUX_AIVM_ENABLE_WGPU)
        target_link_libraries(aivm-benchmark PRIVATE lux_aivm_wgpu)
        target_compile_definitions(aivm-benchmark PRIVATE
            LUX_AIVM_BENCH_WGPU=1)
    endif()

    # ---- AI/ML v0.1 — AI precompile (Apple-only .mm test) ------------------

    if(LUX_AIVM_ENABLE_METAL AND APPLE)
        # Tiny classifier fixture generator. Produces tiny_classifier.bin
        # alongside the test binary so the test can load it from CWD.
        add_executable(aivm-gen-tiny-classifier
            test/gen_tiny_classifier_fixture.cpp)
        target_link_libraries(aivm-gen-tiny-classifier PRIVATE lux_aivm_core)
        target_compile_features(aivm-gen-tiny-classifier PRIVATE cxx_std_20)

        set(_AIVM_FIXTURE ${CMAKE_CURRENT_BINARY_DIR}/tiny_classifier.bin)
        add_custom_command(
            OUTPUT ${_AIVM_FIXTURE}
            COMMAND aivm-gen-tiny-classifier ${_AIVM_FIXTURE}
            DEPENDS aivm-gen-tiny-classifier
            COMMENT "Writing canonical tiny_classifier.bin fixture")
        add_custom_target(aivm-tiny-classifier-fixture ALL
            DEPENDS ${_AIVM_FIXTURE})

        # Stage the metal sources next to the build dir so the runtime loader
        # finds them via __FILE__-relative search.
        configure_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/gpu/metal/ai_inference_int8.metal
            ${CMAKE_CURRENT_BINARY_DIR}/gpu/metal/ai_inference_int8.metal
            COPYONLY)
        configure_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/gpu/metal/ai_inference_attest.metal
            ${CMAKE_CURRENT_BINARY_DIR}/gpu/metal/ai_inference_attest.metal
            COPYONLY)

        add_executable(ai-precompile-test test/ai_precompile_test.mm)
        target_link_libraries(ai-precompile-test PRIVATE lux_aivm_metal)
        target_compile_features(ai-precompile-test PRIVATE cxx_std_20)
        add_dependencies(ai-precompile-test aivm-tiny-classifier-fixture)
        add_test(NAME ai-precompile-test
            COMMAND ai-precompile-test
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    endif()
endif()

message(STATUS "")
message(STATUS "=== Lux AIVM Configuration ===")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build Tests: ${LUX_AIVM_BUILD_TESTS}")
message(STATUS "  Metal:       ${LUX_AIVM_ENABLE_METAL}")
message(STATUS "  CUDA:        ${LUX_AIVM_ENABLE_CUDA}")
message(STATUS "  WGPU:        ${LUX_AIVM_ENABLE_WGPU}")
message(STATUS "")
