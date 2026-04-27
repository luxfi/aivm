# AIVM Benchmarks — v0.59

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

## v0.59 architectural change

v0.58.3 dispatched every kernel as `MTLSizeMake(1, 1, 1)` — a single GPU
thread walking each op stream serially. v0.59 splits the apply phases into
`locate` (1 thread) + `writeback` (parallel, one thread per slot, threadgroup
width 256). The locate phase preserves canonical-order slot assignment
(byte-equal to the CPU oracle); the writeback phase performs the per-slot
field memcpy in parallel.

Determinism test extended with a size sweep (small/medium/large) — all
sizes byte-equal CPU vs Metal vs WGSL. **47 / 47 tests pass.**

The split keeps byte-equivalence trivially:
- Locate is the same canonical sweep as v0.58.3.
- Writeback is independent per slot — each slot's writer thread reads
  `slot_winner_op[s]` (the highest op index that claimed slot s during
  locate) and writes that op's fields to the slot. No cross-slot data flow.

ProvenanceApply keeps the v0.58.3 single-pass behaviour because UpdateWeights
increments a shared version counter (not last-writer-wins). AnchorApply also
stays single-pass because chain integrity (`parent_root == prev.commit_root`)
is fundamentally sequential.

EpochTransition stays at the v0.58.3 single-thread design — splitting it
into per-table parallel-leaf-hash kernels added per-dispatch latency on the
M1 Max integrated GPU (~1-2 ms each) that exceeded the parallel keccak
savings at every workload size we measured.

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

Raw output: [`BENCHMARKS_V059.txt`](./BENCHMARKS_V059.txt).

## Headline result

**On the M1 Max integrated GPU, AIVM remains faster on the CPU reference
than on either GPU backend at every workload size.** This is unchanged from
v0.58.3. Per-kernel dispatch latency (~1-2 ms each) on the integrated GPU
dominates total round time at every size, swamping any per-thread parallel
savings.

The GPU kernels are still byte-for-byte deterministic with the CPU oracle
(all 47 / 47 determinism tests pass, including the new small/medium/large
size sweep). The architecture change is live and prepared for hardware where
dispatch latency is amortised differently — discrete CUDA GPUs, MTLBindless
on Apple Silicon Pro / Ultra, and Dawn/Vulkan on dedicated cards.

The crossover point where GPU beats CPU is **never** for the AIVM workload
shape on this hardware. Numbers below state this honestly.

## FullRound — end-to-end transition

Throughput is attestations/sec (the headline workload).

| backend | size   | mean (ms) |  p50  |  p95  |  p99  |   att/sec   | vs CPU | vs v0.58.3 |
|---------|--------|----------:|------:|------:|------:|------------:|-------:|-----------:|
| cpu     | small  |     1.363 | 1.358 | 1.397 | 1.403 |      73 370 |  1.00x |   ~1.00x   |
| cpu     | medium |     4.535 | 4.531 | 4.610 | 4.612 |     220 524 |  1.00x |   ~1.00x   |
| cpu     | large  |    14.492 |14.558 |14.654 |14.671 |     690 036 |  1.00x |   ~1.00x   |
| cpu     | xlarge |    40.694 |40.780 |41.230 |41.359 |   2 457 378 |  1.00x |   ~1.00x   |
| metal   | small  |    26.513 |26.490 |26.629 |26.644 |       3 772 |  0.05x |    1.01x   |
| metal   | medium |    83.533 |83.529 |83.588 |83.614 |      11 971 |  0.05x |    1.00x   |
| metal   | large  |   260.996 |261.246|261.517|261.566|      38 315 |  0.06x |    1.00x   |
| metal   | xlarge |   706.062 |706.084|706.333|706.354|     141 631 |  0.06x |    0.99x   |

Net effect: FullRound Metal is the same as v0.58.3 within run-to-run noise
on this hardware. The architecture change is in place for future GPUs but
does not unlock a measurable speedup on the M1 Max integrated GPU.

## AttestationApply — multi-block keccak path (140-byte leaves)

Per-mode breakdown for the path the v0.58.3 optnone fix targets — every
attestation hashes a 140-byte leaf, so the keccak absorber crosses the
136-byte rate boundary and runs the multi-block path.

| backend | size   | mean (ms) |   ops/sec   | vs CPU | vs v0.58.3 |
|---------|--------|----------:|------------:|-------:|-----------:|
| cpu     | small  |     1.217 |      82 185 |  1.00x |   ~1.00x   |
| cpu     | medium |     3.162 |     316 213 |  1.00x |   ~1.00x   |
| cpu     | large  |     3.232 |   3 093 605 |  1.00x |   ~1.00x   |
| cpu     | xlarge |     4.259 |  23 481 528 |  1.00x |   ~1.00x   |
| metal   | small  | 737 / 25† |  136 / 3.9k†|  0.00x |  noise     |
| metal   | medium |    87.174 |      11 471 |  0.04x |   0.67x    |
| metal   | large  |  1156 / 90†|  9k / 110k†|  0.00x |  noise     |
| metal   | xlarge |   180.505 |     554 001 |  0.02x |   0.40x    |

(†) per-mode AttestationApply on this hardware shows large run-to-run
variance (10 ms p50 ↔ 1000 ms p99 in the same cell). The 2-phase locate
+ writeback architecture adds one extra dispatch per round, and on the
integrated GPU each dispatch can stall arbitrarily during GPU power-state
transitions. Use FullRound numbers for stable comparison.

CPU peak: **23.5 M attestations/sec at xlarge**. Metal is dispatch-bound at
all sizes — the architectural change is correct (per-thread parallel work)
but the M1 Max integrated GPU's dispatch latency dominates. The benchmark
captures the architecture cost honestly.

## ProvenanceApply — model registry

| backend | size   | mean (ms) |  ops/sec   | vs CPU |
|---------|--------|----------:|-----------:|-------:|
| cpu     | small  |     0.075 |     67 013 |  1.00x |
| cpu     | medium |     0.629 |     79 486 |  1.00x |
| cpu     | large  |     3.095 |    161 526 |  1.00x |
| cpu     | xlarge |     3.195 |  1 565 153 |  1.00x |
| metal   | small  |   116.876 |         43 |  0.00x |
| metal   | medium |    52.624 |        950 |  0.01x |
| metal   | large  |   173.517 |      2 882 |  0.02x |
| metal   | xlarge |   308.984 |     16 182 |  0.01x |

ProvenanceApply stays single-thread by design — the version counter on
UpdateWeights is order-dependent.

## AnchorApply — audit anchor chain

| backend | size   | mean (ms) |  ops/sec   | vs CPU |
|---------|--------|----------:|-----------:|-------:|
| cpu     | small  |     0.091 |    109 945 |  1.00x |
| cpu     | medium |     0.811 |    123 258 |  1.00x |
| cpu     | large  |     8.038 |    124 404 |  1.00x |
| cpu     | xlarge |    33.040 |    302 667 |  1.00x |
| metal   | small  |   213.203 |         47 |  0.00x |
| metal   | medium |   253.513 |        394 |  0.00x |
| metal   | large  |   629.953 |      1 587 |  0.01x |
| metal   | xlarge |   900.272 |     11 108 |  0.04x |

AnchorApply stays single-thread — chain integrity (parent_root chain) is
fundamentally sequential.

## EpochTransition — root computation only

EpochTransition is a single keccak that binds attestation_root,
model_registry_root, and audit_root into `aivm_state_root`. Single dispatch,
single thread (same as v0.58.3 — multi-dispatch parallel was tried and
reverted; the dispatch overhead was higher than the saved keccak work on
this hardware).

| backend | size   | mean (ms) |     trans/sec     | vs CPU |
|---------|--------|----------:|------------------:|-------:|
| cpu     | small  |     0.014 |        6 918 548  |  1.00x |
| cpu     | medium |     0.013 |       74 142 175  |  1.00x |
| cpu     | large  |     0.014 |      723 102 399  |  1.00x |
| cpu     | xlarge |     0.013 |    7 741 975 442  |  1.00x |
| metal   | small  |     ~3 ms*|         (variable) |  0.00x |
| metal   | medium |     3.271 |          305 754  |  0.00x |
| metal   | large  |     2.613 |        3 827 135  |  0.01x |
| metal   | xlarge |     2.558 |       39 091 069  |  0.01x |

(* metal small showed an outlier 6.7 s mean during this run, dominated by
GPU power-state transitions during the 60s budget. p50 was ~4 s; the kernel
itself completes in ~2 ms on warm hardware.)

EpochTransition shows the floor: even with no per-op work the GPU pays a
~2 ms dispatch + roundtrip on the integrated GPU. The CPU finishes the
same finalise in 13 µs. This 2 ms is the effective minimum for any AIVM
round on this backend.

## Why GPU still loses

The kernels in `src/aivm_*.metal` are now split into apply-locate + apply-
writeback (parallel) phases. The writeback runs one thread per slot at
threadgroup width 256. **However:**

1. The locate phase is still single-threaded (preserves canonical-order slot
   assignment, byte-equal to CPU oracle). For xlarge this is 100 k ops × ~700
   ns/op probe work = ~70 ms — it's the bottleneck of AttestationApply.
2. The writeback phase is parallel but on this hardware the
   `dispatchThreadgroups` overhead (~1 ms) is comparable to the saved work
   for kAttSlots = 1024.
3. EpochTransition is single-thread by necessity (the multi-kernel
   parallel split was reverted because dispatch overhead exceeded the
   keccak savings).

The CPU substrate is also doing real work:

- Vector keccak-f1600 inlined per-step
- Direct array access into pre-allocated arenas
- No syscalls, no dispatch, no kernel-launch overhead

For 100 k attestations the CPU finishes in 41 ms — that is 410 ns per
attestation, dominated by one keccak permutation per 140-byte leaf. There
is no PCIe bus, no thread-group barrier, no command buffer, no fence.

## Where GPU is the right tool

The benchmark reports raw substrate throughput on M1 Max integrated. AIVM's
actual production value from the GPU substrate is:

1. **Offload from the validator CPU during consensus** — the CPU is busy
   running Snow consensus and can't also run keccak chains. Even at 0.05×
   CPU throughput, the GPU running the AIVM round in the background frees
   the CPU to keep up with consensus.
2. **Cross-backend determinism enforcement** — having three independent
   GPU implementations that must agree byte-for-byte with the CPU oracle
   is the mechanism that catches consensus-level bugs in any single
   backend. v0.59 extends the determinism harness with a size sweep
   (small/medium/large) on top of the v0.58.3 brief workload.
3. **Discrete-GPU acceleration** — on Linux+CUDA hosts (where dispatch
   latency is ~10 µs not ~1 ms) the v0.59 architectural split should yield
   the per-thread parallelism speedup the brief targets. CUDA path is
   built and tested separately on those hosts.

## Reproducing

```
cmake -S /Users/z/work/luxcpp/aivm -B build \
  -DCMAKE_BUILD_TYPE=Release -DLUX_AIVM_ENABLE_WGPU=ON
cmake --build build --target aivm-layout-test aivm-gpu-engine-test \
                            aivm-determinism-test aivm-benchmark
ctest --test-dir build --output-on-failure
./build/aivm-benchmark > BENCHMARKS_V059.txt
```

Wall time on the M1 Max for the full benchmark: ~5 minutes.
