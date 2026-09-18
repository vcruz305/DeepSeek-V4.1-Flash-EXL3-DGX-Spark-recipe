# TP1 measured results, with provenance

Every figure here was measured on one DGX Spark. Nothing in this file is projected, scaled or
carried over from another topology.

## Runtime identity

Recorded per `AGENTS.md` rule 8. This identity applies to every number in this file.

| Field | Value |
|---|---|
| Hardware | 1 × NVIDIA DGX Spark (GB10), 128 GB unified LPDDR5X, ATS addressing mode |
| Engine | native ExLlamaV3 — **not** vLLM, **not** `vllm-exl3` |
| ExLlamaV3 repo | `https://github.com/vcruz305/exllamav3.git` |
| ExLlamaV3 branch | `feat/gb10-ats-load` |
| ExLlamaV3 commit | `954a8ca6e59d` |
| CUDA | 13.0, `TORCH_CUDA_ARCH_LIST=12.1a` |
| Architecture class | `DeepseekV41ForCausalLM` |
| Drafter | `deepseek_v41_mtp.py` (DSpark / MTP block drafting, block size 5) |
| Model | `vcruz305/DSV4.1-Flash-SAGE-EXL3-1.59bpw` **plus the EXL3 attention / MTP overlay** (`exllamav3/` in that repo) |
| Model revision | `5dc954019183ab3d994b60433256001a3f1780e7` (pack + `exllamav3/` overlay) |
| Topology | TP1, one local CUDA device |
| Context | `CTX=6144` |
| Batch | `max_batch_size=1`, single sequence |
| Speculative policy | `EXL3_DSPARK_CONF=0.7` where a drafter is used; otherwise none |
| EXL3 backend | ExLlamaV3 EXL3 kernels, per-expert mixed K (K1–K6 present) |
| MoE dispatch | per-expert path; grouped CUDA-graph modes measured slower and are off |

`runtime.lock.json` in the repository root pins the **vLLM** stack for TP2/TP4. It does not
describe this path and is not modified by it.

## Decode, no drafter

| Loading mode | tok/s | Load time | Resident |
|---|---:|---:|---:|
| Main model in CUDA | **15.13 – 15.22** | 37.5 s | ~107 GiB |

## Decode, DSpark drafter, confidence 0.7

Warm and cold are reported separately and are never averaged together.

| Loading mode | Fresh prompt | Repeat prompt | Acceptance |
|---|---:|---:|---:|
| **Main in CUDA, drafter aliased** (`EXL3_ATS_COPY='^(?!mtp\.)'`) | **17.53 median**, 19.82 mean | **20.11 – 24.67** | 0.889 |
| Interactive chat session | 11.8 cold | 17.4 warm | 0.74 |

That row is the configuration this recipe recommends, and the only one that fits: 107 GiB of main
model plus ~14 GiB of drafter cannot both be resident in 128 GB, so the drafter stays aliased.

## Prefill

| Loading mode | Chunk | tok/s | Context |
|---|---:|---:|---|
| Warm | 4096 | 254 – 261 | 4k – 6k |
| Model resident in CUDA | 2048 | 154 – 229 | 2k – 6k |

Chunk 4096 does not fit once the model is resident; use 2048 in that configuration.

## Context is nearly free

Single prompt of 1964 tokens repeated in one process, 256 new tokens per generation,
`EXL3_ATS_COPY='^(?!mtp\.)'`, DSpark drafter at `EXL3_DSPARK_CONF=0.7`. This is a
best-case prefix-cache hit and is **not** comparable to the 12-varied-prompt figures
above; it isolates the cost of context, not of a realistic workload.

| CTX | warm decode tok/s | `torch_alloc` |
|---:|---:|---:|
| 6,144 | 33.36 | 107.19 GiB |
| 131,072 | 33.37 | 107.92 GiB |
| 262,144 | 32.16 | 108.69 GiB |

A 43x increase in context costs about 1 tok/s and 1.5 GiB. Only 4 of 46 cache layers
scale with context, at **6272 bytes/token** total (main cache 0.029 GiB + 3200 B/tok
across 4 `CacheLayer_dsa`; drafter 3072 B/tok across 3 `CacheLayer_dspark`). The other
42 layers are fixed-shape recurrent state. `PAGE_SIZE` is 256, so `CTX` must be a
multiple of 256. Every context up to the architecture ceiling of 1,048,576 loads in
36-44 s.

## Confidence gate sweep

Same harness and prompt, CTX=131072, warm repeats:

| `EXL3_DSPARK_CONF` | best warm tok/s | acceptance |
|---:|---:|---:|
| 0.3 | 32.94 | 0.920 |
| 0.5 | 32.27 | 0.945 |
| **0.7** | **33.37** | **0.981** |
| 0.85 | 32.46 | 0.967 |
| 0.95 | 29.83 | 0.970 |

0.7 is optimal. Lowering the gate to make more rounds draft does **not** help, and 0.95
costs 10%. A 10-repetition run at 0.7 held 32.7-33.33 tok/s with no decay.

`EXL3_MOE_MIXED_BSZ1=1` produced no measurable gain (33.27 vs 33.33) and remains off by
default; it also makes greedy output non-reproducible run to run.

## Concurrency

Two concurrent streams **without** the drafter reach 19.12 tok/s aggregate, which is
below what one stream achieves with the drafter. Two streams **with** the drafter fail:

```
RuntimeError: The expanded size of the tensor (2036) must match the existing size (2034)
at non-singleton dimension 1.  Target sizes: [6, 2036].  Tensor sizes: [6, 2034]
```

`[6, N]` is the speculative verify batch (draft size 5 + 1). Concurrency alone works;
speculation combined with concurrency does not. Single stream with the drafter is the
supported configuration.

## The decode is GPU-bound, and that is what sets the ceiling

The per-expert MoE dispatch does synchronize with the host. A CUDA profile over 63 tokens
counted 2550 `cudaStreamSynchronize`, or 40.5 per token against exactly 40 MoE layers, so the
per-layer readback is real and fires on every forward. Removing it would still gain nothing,
because there is no idle GPU time for the host to fill.

Measured by injecting a known quantity of pure GPU work into every MoE layer and reading
wall-clock decode only, with no profiler involved. One 2048×2048 fp16 `mm`, calibrated at **0.1927 ms** on
an otherwise idle device. One process, one load, three generations per level, `N=0` measured
again last as a drift check:

| injections / layer | ms/token | Δ vs `N=0` | Δ per injection |
|---:|---:|---:|---:|
| 0 | 30.648 | n/a | n/a |
| 1 | 32.469 | +1.821 | 1.821 |
| 2 | 34.240 | +3.591 | 1.796 |
| 4 | 37.537 | +6.889 | 1.722 |
| 8 | 43.542 | +12.894 | 1.612 |
| 16 | 55.947 | +25.298 | 1.581 |
| 0 (repeat) | 30.751 | +0.103 | n/a |

A device with idle gaps absorbs the first increments, so its curve stays flat and then bends.
This one is **linear from the first increment**: injected work is paid in full, immediately,
which is only possible on a saturated device. The slope is self-checking: injected work per
verify round divided by 1.821 ms/token implies ~4.2–4.8 tokens per forward, which is what
block size 5 at 0.889 acceptance actually produces.

This single result explains the rest of this file: the grouped CUDA-graph modes were slower,
`EXL3_MOE_MIXED_BSZ1` gained nothing, and every configuration tried landed between 29.8 and
33.4 tok/s. Going faster needs **less GPU work per token**, meaning a smaller pack, higher draft
acceptance, or faster trellis kernels, rather than better host-side scheduling.

> **Methodology note.** `torch.profiler` cannot measure occupancy on this workload. It issues
> 620+ kernel launches per token, so CUPTI per-kernel overhead swamps the signal and reports
> impossible figures (GPU busy 136–144% of wall, negative idle). Summing
> `self_device_time_total` across `key_averages()` also double-counts, because a parent
> `aten::mm` carries the device time of the kernels listed separately beneath it. Wall-clock
> injection is the instrument that works here.

## Measured negative results

Recorded so they are not re-tried. Same runtime identity as above.

| Change | Result | Disposition |
|---|---|---|
| `EXL3_MOE_GROUP_GRAPH=1` (per quantization-key groups, 11–22 graphs/layer) | 9.69 tok/s vs 10.97 baseline | off by default |
| `EXL3_MOE_GROUP_GRAPH=2` (per projection, 11–16 groups) | 10.43 tok/s vs 10.97 baseline | off by default |
| `EXL3_ATS_HUGEPAGE=1` | ~1–2%, within run-to-run noise | not recommended |
| Everything in CUDA, no aliasing (`EXL3_ATS_MMAP=0`, drafter included) | loads in 45.4 s at `torch_alloc` 114.03 GiB leaving 1.52 GiB `MemAvailable`; memory guard killed it before the first token | **does not fit** |
| 64-byte re-lay of the pack | same harness and settings, one run per layout: re-laid 17.29 median / 20.12–24.68 repeats, as-published 17.67 / 19.82–25.06, acceptance 0.889 both | no effect; not needed |
| Forcing a minimum draft length | slower | rejected |
| Draft early-exit | neutral | not enabled |
| `EXL3_MOE_MIXED_BSZ1=1` | ~5% warm decode, **greedy output not reproducible run to run** | **do not use** |
| Raising `EXL3_MOE_FUSED_ROWS` / `EXL3_MOE_FUSED_ROWS_WIDE` | no effect: the MTP verify shape is 36 rows, already inside both defaults (128 / 256), so the row cap was never the constraint | not a lever |
| Repacking to uniform K to reach the fused path | routed-expert weights go from ~101 GiB to ~268 GiB levelled up, ~126 GiB at an intermediate uniform K, against 119.2 GiB of unified memory before drafter, cache and OS | **does not fit** |
| Device-indexed MoE dispatch, to remove the per-layer host sync | the sync is real (40.5 per token) but costs no wall time: injected GPU work is paid in full from the first increment | **not worth building** |
| Two concurrent streams | 19.12 tok/s aggregate without a drafter, below single-stream with one; with the drafter it raises `RuntimeError` | does not help single-stream |
| Re-sweeping the int8 GEMV work decomposition on this GPU (the constants are tuned for a 3090; GB10 has 48 SMs) | paired in one session: shipped default 32.56 tok/s, best swept grid 32.70; under 0.8 tok/s spread across a 5x range of grid sizes | null; the default `maxb * num_sms` already lands on the optimum |
| Deeper speculation, raising the draft block from 5 (`Generator(num_draft_tokens=N)` with `dspark_block_size` to match) | one session, same prompt: block 5 gives 32.69 tok/s at 0.958 acceptance, block 6 gives 32.27 at 0.921, block 8 gives 30.25 at 0.833; generated text identical at all three | **5 is optimal**; acceptance falls monotonically past the block size the drafter was trained at |

The grouped-MoE result is the important one: an exact per-slot mgemm loses to the int8 GEMV path on
this hardware, so reducing kernel launch count did not help.

The GEMV sweep is the other one worth reading. Since the decode is GPU-bound, the remaining lever
would have to be the kernels themselves, and the int8 GEMV path is already close to its floor: it
never materializes fp16 at all. The `u32` product of the extracted trellis word and the codebook
constant *is* four int8 codebook values, consumed directly by `dp4a` against int8-quantized
activations, so a 32-weight block costs roughly 8 integer multiplies plus 8 `dp4a`. Reducing that
meaningfully is not a tuning exercise. Note also that the GEMV path is gated to `size_m <= 2`, so it
serves the single-row drafter steps; the 6-row MTP verify runs through `exl3_gemm` / `exl3_mgemm`.

## Not measured here

- **Native ExLlamaV3 TP2 / TP4.** ExLlamaV3 tensor parallelism is single-host only: one
  `multiprocessing.Process` per *local* CUDA index, payloads through
  `multiprocessing.shared_memory`, `EXLLAMA_MASTER_ADDR` defaulting to `127.0.0.1`, and no
  multi-host worker. It does not span two Sparks, so no cross-node native numbers exist to publish.
  TP2 and TP4 in this repository are the vLLM path.
- **TabbyAPI throughput.** Not run end-to-end against this pack. See `tabbyapi/README.md`.
- **Quantized KV.** Long context *is* measured above, to 262,144.
- **CUDA-graph capture is not an open item.** The batched decode graph
  (`EXL3_DSV4_BATCH_GRAPH`) defaults to on, so it is already active in every figure above, for
  the sliding-window layers, which are the ones this architecture builds on `DSV4Attention`. The
  remaining V4.1 layers use `DSV41Attention`, which has no graph path at all, and the grouped-MoE
  graph modes are measured and rejected in the table above. There is no unexploited graph setting
  on this path.
- **GPU counter profiling.** Nsight Compute is installed on this host but returns
  `ERR_NVGPUCTRPERM` for a non-admin user, so per-kernel stall reasons and achieved occupancy could
  not be collected. Throughput figures here are wall-clock; the kernel launch and host-sync counts
  come from CUPTI activity tracing, which needs no counter permissions.
