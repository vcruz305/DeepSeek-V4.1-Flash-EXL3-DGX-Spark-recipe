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
| Model | `vcruz305/DSV4.1-Flash-SAGE-EXL3-1.59bpw` **plus the EXL3 attention / MTP overlay** (`exllamav3/` in that repo). Measured on both the as-published and the 64-byte re-laid layout; in this configuration they perform the same (see "Re-lay makes no difference in the copy config" below). |
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
| Fully aliased (`EXL3_ATS_MMAP=1`, no copy) | 13.96 – 14.19 | — | weights stay reclaimable |
| Whole model copied into CUDA | **15.13 – 15.22** | 37.5 s | ~107 GiB |

Copying the whole model is +7.3% over full aliasing.

## Decode, DSpark drafter, confidence 0.7

Warm and cold are reported separately and are never averaged together.

| Loading mode | Fresh prompt | Repeat prompt | Acceptance |
|---|---:|---:|---:|
| Aliased + drafter | 11.46 median | 16 – 17 | — |
| **Main copied, drafter aliased** (`EXL3_ATS_COPY='^(?!mtp\.)'`) | **17.53 median**, 19.82 mean | **20.11 – 24.67** | 0.889 |
| Interactive chat session | 11.8 cold | 17.4 warm | 0.74 |

The "main copied, drafter aliased" row is the configuration this recipe recommends. 107 GiB of main
model plus ~14 GiB of drafter cannot both be resident in 128 GB, so the drafter stays in page cache.

## Prefill

| Loading mode | Chunk | tok/s | Context |
|---|---:|---:|---|
| Warm | 4096 | 254 – 261 | 4k – 6k |
| Model resident in CUDA | 2048 | 154 – 229 | 2k – 6k |

Chunk 4096 does not fit once the model is resident; use 2048 in that configuration.

## Zero-copy aliasing coverage

| Pack layout | Aliased | Copied |
|---|---:|---:|
| As published (tensors on arbitrary offsets) | 48.6 GiB | 67.4 GiB |
| Re-laid at 64-byte alignment | all text-model tensors | none |

The row above counts text-model tensors only. Counted over all 17 shards under the loader's own rule
(`align = 16` for `int16`, `dtype.itemsize` otherwise, tensors below 1 MiB always copied), the
published pack leaves **67.41 GiB** of `int16` trellis data off the 16-byte grid and the re-laid pack
leaves **none**; whole-pack aliased totals are 240.05 GiB and 318.04 GiB respectively, the difference
being the `F8_E4M3` engram tables, which need only 1-byte alignment and alias either way. That is why
`--skip .engram.embed.` costs nothing.

**This coverage only affects the fully-aliased mode.** A misaligned tensor is copied into unevictable
CUDA memory rather than rejected, so an un-re-laid pack never fails to load. In the recommended
copy config it is not even reached — see the next section, where both layouts measure the same.

## Re-lay makes no difference in the copy config

Same harness, same settings (`EXL3_ATS_COPY='^(?!mtp\.)'`, `EXL3_DSPARK_CONF=0.7`, `TOKENS=256`,
prewarm, 12 fresh prompts plus 4 repeats), one run per pack layout, back to back on an idle box:

| pack layout | fresh 5-12 median | fresh 5-12 mean | repeats | mean acceptance |
|---|---:|---:|---:|---:|
| 64-byte re-laid + overlay | 17.29 | 18.32 | 20.12 – 24.68 | 0.889 |
| as published + overlay (no re-lay) | 17.67 | 19.95 | 19.82 – 25.06 | 0.889 |

Both reported `torch_alloc 107.19 GiB` and `aliased/copied GiB [6.66, 0.0]`, and `MemAvailable` fell
from ~118 GiB to ~5–7 GiB in both. With the main model copied into CUDA, tensor alignment is only
consulted for the aliased drafter, whose overlay parts are already on the 64-byte grid, so the
published pack's 67.41 GiB of off-grid `int16` is never reached. n=1 per arm; the fresh mean is the
noisier statistic (both runs are pulled by one ~31 tok/s prompt), which is why the median is quoted.

The re-lay remains necessary for the fully-aliased mode in the first table, where those 67.41 GiB
would otherwise be copied into unevictable CUDA memory.

## Measured negative results

Recorded so they are not re-tried. Same runtime identity as above.

| Change | Result | Disposition |
|---|---|---|
| `EXL3_MOE_GROUP_GRAPH=1` (per quantization-key groups, 11–22 graphs/layer) | 9.69 tok/s vs 10.97 baseline | off by default |
| `EXL3_MOE_GROUP_GRAPH=2` (per projection, 11–16 groups) | 10.43 tok/s vs 10.97 baseline | off by default |
| `EXL3_ATS_HUGEPAGE=1` | ~1–2%, within run-to-run noise | not recommended |
| Forcing a minimum draft length | slower | rejected |
| Draft early-exit | neutral | not enabled |
| `EXL3_MOE_MIXED_BSZ1=1` | ~5% warm decode, **greedy output not reproducible run to run** | **do not use** |

The grouped-MoE result is the important one: an exact per-slot mgemm loses to the int8 GEMV path on
this hardware, so reducing kernel launch count did not help.

## Not measured here

- **Native ExLlamaV3 TP2 / TP4.** ExLlamaV3 tensor parallelism is single-host only: one
  `multiprocessing.Process` per *local* CUDA index, payloads through
  `multiprocessing.shared_memory`, `EXLLAMA_MASTER_ADDR` defaulting to `127.0.0.1`, and no
  multi-host worker. It does not span two Sparks, so no cross-node native numbers exist to publish.
  TP2 and TP4 in this repository are the vLLM path.
- **TabbyAPI throughput.** Not run end-to-end against this pack. See `tabbyapi/README.md`.
- **Long context beyond 6144**, quantized KV, and CUDA-graph capture for the heterogeneous mixed-K
  path.
