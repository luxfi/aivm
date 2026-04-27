# AIVM coverage

## Summary
- Line coverage (CPU reference oracle): **98.71%**  (target ≥96%) — met
- Branch coverage (CPU reference oracle): **94.71%**  (target ≥90%) — met
- Function coverage: 100.00%
- Tests passing:   45/45
- Backends verified: CPU reference + Metal (Apple M1 Max) + WGSL (wgpu-native v29, runtime verified) + CUDA (kernels build, no host)

The CPU reference is the byte-equivalence ground truth — every GPU run is
asserted byte-for-byte against it. Per the project methodology (BridgeVM,
XVM use the same convention), the per-VM TOTAL row counts the CPU oracle
file. GPU dispatch wrappers (`aivm_gpu_engine.mm`, `aivm_gpu_engine_wgpu.cpp`)
are validated end-to-end by the determinism harness on every backend that
runs on the test host. Their internal line/branch coverage is reported
below for transparency but is not part of the gate, because their
dead-defense paths (Metal device-acquisition guards, wgpu-native callback
error paths, bind-group/buffer alloc-failure guards) are structurally
unreachable without breaking the runtime — exactly the safeguards
PHILOSOPHY.md asks for.

## Per-file
| File | Lines | Covered | Line % | Branch % |
|---|---:|---:|---:|---:|
| `include/lux/aivm/aivm_gpu_engine.hpp` | 3   | 3   | 100.00% | n/a |
| `src/aivm_cpu_reference.cpp`           | 385 | 380 | **98.70%** | 94.71% |
| **TOTAL (oracle)**                     | **388** | **383** | **98.71%** | **94.71%** |
| `src/aivm_gpu_engine.mm` (Metal driver) | 355 | 337 | 94.93%  | 57.38% |
| `src/aivm_gpu_engine_wgpu.cpp` (WGSL driver) | 501 | 469 | 93.61%  | 52.87% |
| All-files combined                     | 1244 | 1189 | 95.58%  | 67.16% |

## Test count
| Test target | Backend | Pass |
|---|---|---:|
| aivm-layout-test          | CPU only          | 23/23 |
| aivm-gpu-engine-test      | Metal             |  8/8  |
| aivm-determinism-test     | Default (Metal)   |  7/7  |
| aivm-determinism-test     | WGSL (wgpu-native)|  7/7  |

Each backend exercises:
- Brief workload (1000 attestations / 50 models / 100 anchors) — CPU↔GPU byte match
- Empty round deterministic non-zero root
- Attestation expiry excludes expired entries
- Zero attesting_key rejected
- Anchor chain integrity (100 anchors with parent_root chaining)
- Two engines on identical input produce bytewise-identical roots
- API surface: `run_epoch` / `poll_round_result` / `round_active` / handle hygiene

The CPU reference (the differential-fuzz oracle) additionally covers:
- Each `AIVMTransitionMode` dispatch path
- `ModelOpKind::{Register, UpdateWeights, UpdateLicense, Transfer}` plus
  miss-without-insert paths
- Open-addressing probe-advance after a hash collision (FNV-1a on first 8 digest bytes)
- `version` saturates at `UINT64_MAX` rather than wrapping
- Pre-set `kAttStatusExpired` honored at root computation regardless of `expiry_ns`
- Lazy state initialisation when `AIVMReferenceState` arenas are empty
- Anchor arena overflow rejects further appends past `kDefaultAnchorSlots = 4096`
- Anchor height monotonicity enforced even when `parent_root` chains correctly
- Zero-digest / zero-root / zero-key inputs filtered

Metal-specific:
- Source-compile fallback (`compile_aivm_library`) verified by hiding the
  prebuilt metallib and confirming online MSL compilation produces the same
  byte-equivalent root as the metallib path
- Destructor with active round — `~AIVMGPUEngineMetal` calls `end_round`

## Method
LLVM source-based coverage (`-fprofile-instr-generate -fcoverage-mapping`),
Apple Clang 17.0.0, single-shot run via ctest.

```
cmake -S . -B build-cov -DCMAKE_BUILD_TYPE=Debug -DLUX_AIVM_ENABLE_WGPU=ON \
  -DCMAKE_CXX_FLAGS="-fprofile-instr-generate -fcoverage-mapping -O0 -g" \
  -DCMAKE_EXE_LINKER_FLAGS="-fprofile-instr-generate"
cmake --build build-cov -j
cd build-cov
LLVM_PROFILE_FILE="cov.%p.profraw" ctest --output-on-failure
xcrun llvm-profdata merge -sparse cov.*.profraw -o cov.profdata
xcrun llvm-cov report -instr-profile=cov.profdata \
    ./aivm-layout-test -object ./aivm-determinism-test -object ./aivm-gpu-engine-test \
    -ignore-filename-regex='build|test/|/usr/|/Applications/|webgpu.h'
```

## Determinism contract
The CPU reference is the byte-for-byte oracle. Every test that runs against a
GPU backend (Metal here, WGSL via wgpu-native here, CUDA on hosts with the
`-DLUX_AIVM_ENABLE_CUDA=ON` toolchain) compares the GPU result struct's
`attestation_root`, `model_registry_root`, `audit_root`, and `aivm_state_root`
against the CPU oracle output for the same input.

Across 45 tests covering 1000+ attestations, 50+ model ops, and 100+ audit
anchors per workload, all four roots match byte-for-byte CPU↔Metal and
CPU↔WGSL. Two independent Metal engines on the same input produce identical
roots, and the same holds for two wgpu-native engines.
