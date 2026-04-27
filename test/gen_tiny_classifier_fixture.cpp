// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// gen_tiny_classifier_fixture.cpp — emits the canonical tiny_classifier.bin
// fixture. CMake invokes this once at build time so the file lands in the
// build/test/fixtures directory; the test reads it back.

#include "lux/aivm/ai_precompile.hpp"

#include <cstdio>
#include <fstream>
#include <string>

int main(int argc, char** argv) {
    std::string out_path = "tiny_classifier.bin";
    if (argc >= 2) out_path = argv[1];

    auto m     = aivm::ai::make_tiny_classifier_fixture();
    auto bytes = aivm::ai::tiny_classifier_to_bytes(m);

    std::ofstream f(out_path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "gen_tiny_classifier_fixture: cannot open %s\n",
                     out_path.c_str());
        return 1;
    }
    f.write(reinterpret_cast<const char*>(bytes.data()),
            std::streamsize(bytes.size()));
    if (!f) {
        std::fprintf(stderr, "gen_tiny_classifier_fixture: write failed %s\n",
                     out_path.c_str());
        return 1;
    }
    std::printf("wrote %s (%zu bytes, model_hash[0..4]=%02x%02x%02x%02x)\n",
                out_path.c_str(), bytes.size(),
                m.model_hash[0], m.model_hash[1], m.model_hash[2], m.model_hash[3]);
    return 0;
}
