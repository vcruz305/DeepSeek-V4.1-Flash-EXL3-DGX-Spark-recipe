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
