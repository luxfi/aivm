// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// ai_precompile_metal.mm — Metal driver for the AI precompile.
//
// Goal: run the same int8 GEMM as ai_precompile.cpp on the GPU and produce
// byte-identical output bytes. Reduction order, accumulator width, shift
// semantics, and saturation match the CPU reference exactly.
//
// This driver is intentionally narrow — only the tiny classifier path is
// wired today. It exists to prove byte-equality across CPU↔Metal for the
// AI/ML v0.1 ship; broader model support lands in v0.2 alongside the model
// loader expansion.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "lux/aivm/ai_precompile.hpp"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <vector>

namespace aivm::ai::metal {

struct AIMetalParams {
    uint32_t n;
    int32_t  shift1;
    int32_t  shift2;
};

class AIMetalEngine {
public:
    static AIMetalEngine* shared() {
        static AIMetalEngine* g = create();
        return g;
    }

    bool ok() const { return device_ != nil; }
    const char* device_name() const { return device_name_.c_str(); }

    // Run the tiny classifier on `inputs` (size = n*32 bytes int8). Writes
    // exactly n*1 output bytes. Mode-aware: when seal_output is true, the
    // raw output stays in a private buffer; the caller provides a separate
    // commitment-only path. We always also return the raw bytes (the engine
    // is a primitive — sealing is the caller's contract).
    bool run_tiny_classifier(const TinyClassifier& m,
                             const int8_t* inputs, uint32_t n,
                             int8_t* outputs)
    {
        if (!ok()) return false;
        @autoreleasepool {
            uint32_t in_bytes = n * 32u;
            uint32_t hid_bytes = n * 16u;
            uint32_t out_bytes = n * 1u;

            id<MTLBuffer> bW1 = [device_ newBufferWithBytes:m.w1.data()
                                                     length:m.w1.size()
                                                    options:MTLResourceStorageModeShared];
            id<MTLBuffer> bB1 = [device_ newBufferWithBytes:m.b1.data()
                                                     length:m.b1.size() * sizeof(int32_t)
                                                    options:MTLResourceStorageModeShared];
            id<MTLBuffer> bW2 = [device_ newBufferWithBytes:m.w2.data()
                                                     length:m.w2.size()
                                                    options:MTLResourceStorageModeShared];
            id<MTLBuffer> bB2 = [device_ newBufferWithBytes:m.b2.data()
                                                     length:m.b2.size() * sizeof(int32_t)
                                                    options:MTLResourceStorageModeShared];
            id<MTLBuffer> bX  = [device_ newBufferWithBytes:inputs
                                                     length:in_bytes
                                                    options:MTLResourceStorageModeShared];
            id<MTLBuffer> bH  = [device_ newBufferWithLength:hid_bytes
                                                    options:MTLResourceStorageModeShared];
            id<MTLBuffer> bY  = [device_ newBufferWithLength:out_bytes
                                                    options:MTLResourceStorageModeShared];

            AIMetalParams p{n, m.shift1, m.shift2};
            id<MTLBuffer> bP  = [device_ newBufferWithBytes:&p
                                                     length:sizeof(p)
                                                    options:MTLResourceStorageModeShared];

            id<MTLCommandBuffer> cb = [queue_ commandBuffer];

            // Layer 1.
            {
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:layer1_];
                [enc setBuffer:bW1 offset:0 atIndex:0];
                [enc setBuffer:bB1 offset:0 atIndex:1];
                [enc setBuffer:bX  offset:0 atIndex:4];
                [enc setBuffer:bH  offset:0 atIndex:5];
                [enc setBuffer:bP  offset:0 atIndex:7];
                MTLSize grid = MTLSizeMake(16, n, 1);
                MTLSize tg   = MTLSizeMake(16, 1, 1);
                [enc dispatchThreads:grid threadsPerThreadgroup:tg];
                [enc endEncoding];
            }
            // Layer 2.
            {
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:layer2_];
                [enc setBuffer:bW2 offset:0 atIndex:2];
                [enc setBuffer:bB2 offset:0 atIndex:3];
                [enc setBuffer:bH  offset:0 atIndex:5];
                [enc setBuffer:bY  offset:0 atIndex:6];
                [enc setBuffer:bP  offset:0 atIndex:7];
                MTLSize grid = MTLSizeMake(1, n, 1);
                MTLSize tg   = MTLSizeMake(1, 1, 1);
                [enc dispatchThreads:grid threadsPerThreadgroup:tg];
                [enc endEncoding];
            }

            [cb commit];
            [cb waitUntilCompleted];

            std::memcpy(outputs, bY.contents, out_bytes);
            return true;
        }
    }

private:
    id<MTLDevice>               device_  = nil;
    id<MTLCommandQueue>         queue_   = nil;
    id<MTLComputePipelineState> layer1_  = nil;
    id<MTLComputePipelineState> layer2_  = nil;
    std::string                 device_name_;

    static AIMetalEngine* create() {
        auto* e = new AIMetalEngine();
        @autoreleasepool {
            e->device_ = MTLCreateSystemDefaultDevice();
            if (!e->device_) return e;
            e->queue_ = [e->device_ newCommandQueue];
            if (!e->queue_) { e->device_ = nil; return e; }
            e->device_name_ = std::string([[e->device_ name] UTF8String]);

            NSString* src = load_source();
            if (!src) { e->device_ = nil; return e; }

            MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
            opts.languageVersion = MTLLanguageVersion3_0;
            NSError* err = nil;
            id<MTLLibrary> lib = [e->device_ newLibraryWithSource:src
                                                          options:opts
                                                            error:&err];
            if (!lib) {
                if (err) std::fprintf(stderr, "AI Metal compile: %s\n",
                                      [[err localizedDescription] UTF8String]);
                e->device_ = nil;
                return e;
            }

            id<MTLFunction> f1 = [lib newFunctionWithName:@"ai_int8_layer1"];
            id<MTLFunction> f2 = [lib newFunctionWithName:@"ai_int8_layer2"];
            if (!f1 || !f2) { e->device_ = nil; return e; }

            e->layer1_ = [e->device_ newComputePipelineStateWithFunction:f1 error:&err];
            if (!e->layer1_) { e->device_ = nil; return e; }
            e->layer2_ = [e->device_ newComputePipelineStateWithFunction:f2 error:&err];
            if (!e->layer2_) { e->device_ = nil; return e; }
        }
        return e;
    }

    static NSString* load_source() {
        std::filesystem::path here = std::filesystem::path(__FILE__).parent_path();
        std::filesystem::path candidates[] = {
            here.parent_path() / "gpu" / "metal" / "ai_inference_int8.metal",
            std::filesystem::current_path() / "gpu" / "metal" / "ai_inference_int8.metal",
            std::filesystem::current_path() / "aivm" / "gpu" / "metal" / "ai_inference_int8.metal",
            std::filesystem::current_path().parent_path() / "gpu" / "metal" / "ai_inference_int8.metal",
        };
        for (const auto& p : candidates) {
            if (!std::filesystem::exists(p)) continue;
            NSString* path = [NSString stringWithUTF8String:p.c_str()];
            NSError* e = nil;
            NSString* s = [NSString stringWithContentsOfFile:path
                                                    encoding:NSUTF8StringEncoding
                                                       error:&e];
            if (s) return s;
        }
        return nil;
    }
};

bool metal_available() { return AIMetalEngine::shared()->ok(); }

const char* metal_device_name() { return AIMetalEngine::shared()->device_name(); }

bool metal_run_tiny_classifier(const TinyClassifier& m,
                               const int8_t* inputs, uint32_t n,
                               int8_t* outputs)
{
    auto* e = AIMetalEngine::shared();
    if (!e->ok()) return false;
    return e->run_tiny_classifier(m, inputs, n, outputs);
}

}  // namespace aivm::ai::metal
