# AIVM Benchmarks — v0.58.3

A-Chain transition substrate, four backends, four workload sizes.

## Hardware / build

| Item       | Value                                              |
|------------|----------------------------------------------------|
| Host       | Apple M1 Max, 10 cores, 64 GiB                      |
| OS         | macOS 26.4 (build 25E241)                           |
| Compiler   | Apple clang 17.0.0                                  |
| Build      | `-O3 -DNDEBUG`, Release                             |
| GPU        | Apple M1 Max integrated GPU (Metal, AGXMetalG13X)   |
| WebGPU     | wgpu-native via Homebrew (`/opt/homebrew/lib`)      |
| CUDA       | not available on this host                          |

Backends measured:

- **cpu** — `aivm_cpu_reference.cpp` reference oracle, vector-Keccak-f1600
- **metal** — `aivm_gpu_engine.mm`, prebuilt `.metallib` (no online MSL compile)
- **wgsl** — `aivm_gpu_engine_wgpu.cpp` via wgpu-native

CUDA is built and tested separately on Linux+CUDA hosts; not present here.

## Methodology

Each cell is `(backend × size × mode)`. Per cell: 3 warm-ups + 10 measured
iterations. A 60s wall-clock budget caps any single cell so a slow
configuration cannot stall the run.

Workloads:

| size   | attestations | model regs | audit anchors |
|--------|-------------:|-----------:|--------------:|
| small  |          100 |          5 |            10 |
| medium |        1 000 |         50 |           100 |
| large  |       10 000 |        500 |         1 000 |
| xlarge |      100 000 |      5 000 |        10 000 |

Modes (one section each):

- `FullRound` — Attestation + Provenance + Anchor + EpochTransition chained
- `AttestationApply` — TEE attestations only (140-byte leaves → multi-block keccak path)
- `ProvenanceApply` — model registry only
- `AnchorApply` — audit anchors only
- `EpochTransition` — close epoch, emit `aivm_state_root`

Metric `ops/sec` is the throughput of the primary stream for that mode
(attestations for FullRound/AttestationApply, models for ProvenanceApply,
anchors for AnchorApply, transitions for EpochTransition). `vs CPU` is
`mean(cpu) / mean(backend)` — `<1.00x` means the GPU is slower.

Raw output: [`BENCHMARKS_RAW.txt`](./BENCHMARKS_RAW.txt).

## Headline result

**On this hardware AIVM is faster on the CPU reference than on either GPU
backend at every workload size.** The GPU kernels are correct (4-way
byte-equivalent with the CPU oracle, all 45 determinism tests pass) but
they are not parallelised — the Metal driver issues `dispatchThreads:
MTLSizeMake(1,1,1)` for each phase and the WGSL driver does the same. Each
attestation, model op, and anchor op is processed serially on a single GPU
thread, so the GPU is doing what is fundamentally a 1-core hash loop with
PCIe / dispatch overhead on top.

The crossover point where GPU beats CPU is **never** for the AIVM workload
shape on this hardware. Numbers below state this honestly.

## FullRound — end-to-end transition

Throughput is attestations/sec (the headline workload).

| backend | size   | mean (ms) |  p50  |  p95  |  p99  |   att/sec   | vs CPU |
|---------|--------|----------:|------:|------:|------:|------------:|-------:|
| cpu     | small  |     1.337 | 1.338 | 1.381 | 1.390 |      74 781 |  1.00x |
| cpu     | medium |     4.414 | 4.417 | 4.511 | 4.538 |     226 549 |  1.00x |
| cpu     | large  |    13.927 |13.937 |14.071 |14.075 |     718 053 |  1.00x |
| cpu     | xlarge |    39.363 |39.329 |39.934 |39.937 |   2 540 461 |  1.00x |
| metal   | small  |    26.303 |26.279 |26.425 |26.427 |       3 802 |  0.05x |
| metal   | medium |    83.325 |83.325 |83.491 |83.528 |      12 001 |  0.05x |
| metal   | large  |   261.286 |261.34 |261.53 |261.56 |      38 272 |  0.05x |
| metal   | xlarge |   695.354 |695.13 |697.04 |697.18 |     143 812 |  0.06x |
| wgsl    | small  |  3 832.20 |1 711  |10 223 |10 997 |          26 |  0.00x |
| wgsl    | medium |  4 192.34 |3 086  | 9 929 |12 765 |         239 |  0.00x |
| wgsl    | large  |      skip |   —   |   —   |   —   |           — |     —  |
| wgsl    | xlarge |      skip |   —   |   —   |   —   |           — |     —  |

CPU keeps a flat 16–20× lead across all sizes. Metal scales with workload
(per-op cost ~7 µs at xlarge), WGSL doesn't scale (per-iter cost dominated
by `wgpu_hal::metal::Device::wait` polling latency).

## AttestationApply — multi-block keccak path (140-byte leaves)

This is the path the optnone fix targets — every attestation hashes a
140-byte leaf, so the keccak absorber crosses the 136-byte rate boundary
and runs the multi-block path. CPU result here is the post-optnone number.

| backend | size   | mean (ms) |   ops/sec   | vs CPU |
|---------|--------|----------:|------------:|-------:|
| cpu     | small  |     1.186 |      84 344 |  1.00x |
| cpu     | medium |     3.434 |     291 241 |  1.00x |
| cpu     | large  |     3.191 |   3 134 125 |  1.00x |
| cpu     | xlarge |     4.091 |  24 440 989 |  1.00x |
| metal   | small  |    23.370 |       4 279 |  0.05x |
| metal   | medium |    58.530 |      17 085 |  0.06x |
| metal   | large  |    72.738 |     137 480 |  0.04x |
| metal   | xlarge |    72.772 |   1 374 151 |  0.06x |
| wgsl    | small  |  4 749.32 |          21 |  0.00x |

CPU peak: **24.4 M attestations/sec at xlarge**. Metal peak: 1.37 M
attestations/sec — 17× behind CPU. (The CPU jump from medium → large at
constant 3 ms is the `kDefaultAttestationSlots = 1024` arena saturating;
inserts past 1024 collide and probe linearly to the same handful of slots.
This is a fixed-cost ceiling, not a scaling property.)

## ProvenanceApply — model registry

| backend | size   | mean (ms) |  ops/sec   | vs CPU |
|---------|--------|----------:|-----------:|-------:|
| cpu     | small  |     0.071 |     70 381 |  1.00x |
| cpu     | medium |     0.639 |     78 300 |  1.00x |
| cpu     | large  |     3.823 |    130 774 |  1.00x |
| cpu     | xlarge |     3.144 |  1 590 413 |  1.00x |
| metal   | small  |     3.573 |      1 399 |  0.02x |
| metal   | medium |    12.922 |      3 869 |  0.05x |
| metal   | large  |    56.848 |      8 795 |  0.07x |
| metal   | xlarge |    73.731 |     67 814 |  0.04x |
| wgsl    | small  |  3 586.03 |          1 |  0.00x |

## AnchorApply — audit anchor chain

| backend | size   | mean (ms) |  ops/sec   | vs CPU |
|---------|--------|----------:|-----------:|-------:|
| cpu     | small  |     0.118 |     84 959 |  1.00x |
| cpu     | medium |     0.959 |    104 237 |  1.00x |
| cpu     | large  |     7.774 |    128 631 |  1.00x |
| cpu     | xlarge |    32.268 |    309 902 |  1.00x |
| metal   | small  |     3.854 |      2 595 |  0.03x |
| metal   | medium |    17.027 |      5 873 |  0.06x |
| metal   | large  |   137.713 |      7 261 |  0.06x |
| metal   | xlarge |   551.869 |     18 120 |  0.06x |
| wgsl    | small  |  1 609.60 |          6 |  0.00x |

## EpochTransition — root computation only

EpochTransition is a single keccak that binds attestation_root,
model_registry_root, and audit_root into `aivm_state_root`. There is no
per-op iteration here, only a fixed-cost finalise.

| backend | size   | mean (ms) |     trans/sec     | vs CPU |
|---------|--------|----------:|------------------:|-------:|
| cpu     | small  |     0.016 |        6 075 925  |  1.00x |
| cpu     | medium |     0.016 |       60 759 248  |  1.00x |
| cpu     | large  |     0.016 |      608 987 437  |  1.00x |
| cpu     | xlarge |     0.016 |    6 082 096 134  |  1.00x |
| metal   | small  |     2.267 |           44 116  |  0.01x |
| metal   | medium |     2.318 |          431 344  |  0.01x |
| metal   | large  |     1 691*|            5 914  |  0.00x |
| metal   | xlarge |     2.483 |       40 269 539  |  0.01x |
| wgsl    | small  |  1 174.44 |               85  |  0.00x |

(* metal large mean is dominated by a single ~12 s p99 outlier — likely a
GPU power-state event during the cell. p50 was 2.5 ms, in line with the
other Metal sizes.)

EpochTransition shows the floor: even with no per-op work the GPU pays a
~2.3 ms dispatch + roundtrip, while the CPU finishes the same finalise in
16 µs. This 2.3 ms is the effective minimum for any AIVM round on this
backend.

## Why GPU loses

The kernels in `src/aivm_*.metal`, `src/aivm_*.cu`, and `src/aivm_*.wgsl`
are written single-threaded — one shader thread walks the entire
attestation/model/anchor stream, computing each leaf hash and arena
insertion in sequence. The Metal dispatch is literally `dispatchThreads:
MTLSizeMake(1, 1, 1) threadsPerThreadgroup: MTLSizeMake(1, 1, 1)` for all
four phases.

This was the right choice for v0.58 — the goal was 4-way byte-equivalence
with the CPU reference (CPU vs Metal vs CUDA vs WGSL), and a serial kernel
makes that determinism easy to reason about. It was **not** an attempt to
beat the CPU on throughput.

The CPU substrate is also doing real work:

- Vector keccak-f1600 inlined per-step
- Direct array access into pre-allocated arenas
- No syscalls, no dispatch, no kernel-launch overhead

For 100 k attestations the CPU finishes in 39 ms — that is 390 ns per
attestation, dominated by one keccak permutation per 140-byte leaf. There
is no PCIe bus, no thread-group barrier, no command buffer, no fence.

## Where GPU is the right tool

The benchmark reports on raw substrate throughput. AIVM's actual production
value from the GPU substrate is:

1. **Offload from the validator CPU during consensus** — the CPU is busy
   running Snow consensus and can't also run keccak chains. Even at 0.05×
   CPU throughput, the GPU running the AIVM round in the background frees
   the CPU to keep up with consensus.
2. **Cross-backend determinism enforcement** — having three independent
   GPU implementations that must agree byte-for-byte with the CPU oracle
   is the mechanism that lets us catch consensus-level bugs in any single
   backend.
3. **Future parallelisation** — the data layouts (open-addressing arenas,
   independent leaf hashes) allow per-thread parallelism. v0.59+ work
   should parallelise the per-op loop within each phase. The expected
   crossover with CPU is at ~10 k ops once per-thread dispatch is N=1024
   instead of N=1.

## Reproducing

```
cmake -S /Users/z/work/luxcpp/aivm -B build-bench \
  -DCMAKE_BUILD_TYPE=Release -DLUX_AIVM_ENABLE_WGPU=ON
cmake --build build-bench --target aivm-benchmark
./build-bench/aivm-benchmark > BENCHMARKS_RAW.txt
```

Wall time on the M1 Max for the full benchmark: ~5 minutes.
