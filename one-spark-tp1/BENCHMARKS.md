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

0.7 is optimal on this prompt, and 0.95 costs 10%. A 10-repetition run at 0.7 held
32.7-33.33 tok/s with no decay. The gate keeps the longest prefix of draft positions whose
confidence clears the threshold, so raising it shortens the draft block and lowering it
lengthens it. A threshold of 0 for a round means that round skips drafting entirely.

> **This table is confounded, and the real effect is far smaller than it looks.** Changing
> `EXL3_DSPARK_CONF` changes the generated text on most prompts, so these rows are not all
> decoding the same output, and a row that looks faster may simply have produced an easier
> continuation. Re-measured across six subjects with output hashes recorded: on the two
> subjects whose text stayed identical across 0.5 / 0.7 / 0.9 the whole span was 9% and 3%,
> in opposite directions and at the noise floor. Apparent large wins elsewhere, including a
> 35% one, were different continuations rather than faster decoding of the same one. Treat
> 0.7 as a sound default rather than a tuned optimum, and see "Speculation is not
> output-exact" below for why the text moves at all.

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
again last as a drift check. As with the context and confidence tables above, these are **warm
repeated-prompt** figures and are **not** comparable to the fresh-prompt numbers under "Decode,
DSpark drafter": they isolate the cost of added GPU work, not a realistic workload.

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
33.4 tok/s **on warm repeats**, where the recommended configuration's fresh-prompt median is
17.53. Going faster needs **less GPU work per token**, meaning a smaller pack, higher draft
acceptance, or faster trellis kernels, rather than better host-side scheduling.

> **Methodology note.** `torch.profiler` cannot measure occupancy on this workload. It issues
> 620+ kernel launches per token, so CUPTI per-kernel overhead swamps the signal and reports
> impossible figures (GPU busy 136–144% of wall, negative idle). Summing
> `self_device_time_total` across `key_averages()` also double-counts, because a parent
> `aten::mm` carries the device time of the kernels listed separately beneath it. Wall-clock
> injection is the instrument that works here.

## Engram row prefetch

Engram gathers FP8 n-gram rows on every forward. The prefetch reads the rows it is about to need on
the host first, so their pages enter the page cache in parallel instead of the GPU faulting on them
one page at a time. It is **on by default**. Both `_gather` return values are discarded, so it is a
page-cache hint and cannot change output, which makes it safe to disable on correctness grounds.

Two workload shapes were measured separately, because they disagree. Per `AGENTS.md` rule 9 this is
an explicit Engram variant, not part of the baseline: every other figure in this file was produced
with the prefetch at its default.

### Repeated prompt: prefetch off is faster

Same harness as the tables above, four runs in one session, strictly alternating to control for
drift:

| Run | `EXL3_ENGRAM_PREFETCH` | ms/token | tok/s |
|---:|---|---:|---:|
| 1 | 1 (default) | 30.721 | 32.55 |
| 2 | **0** | 29.737 | **33.63** |
| 3 | 1 (default) | 30.607 | 32.67 |
| 4 | **0** | 29.713 | **33.66** |

**+3.2%.** Both off runs beat both on runs with no overlap, and the strict alternation means session
drift cannot produce that ordering. This is exactly the regime this harness measures: one prompt
repeated, so the rows are already resident, the gather finds every page present, and its work plus a
per-layer host sync is pure overhead.

### Varied prompts: a wash, and the statistic decides the sign

Six distinct subjects, one per generation, 255 new tokens and ~1900 prompt tokens each, so no
generation reuses the previous one's rows. Median and mean over the same six reps:

| Run | `EXL3_ENGRAM_PREFETCH` | median ms/tok | mean ms/tok |
|---:|---|---:|---:|
| 1 | 1 (default) | 42.819 | 51.837 |
| 2 | 0 | **41.671** | 53.448 |
| 3 | 1 (default) | 37.953 | **48.078** |
| 4 | 0 | **37.521** | 51.475 |

Off wins on median in both pairs and loses on mean in both pairs, so on varied traffic this is a
wash rather than a win. A single subject causes the entire disagreement: on the marine-biology
prompt the prefetch is worth 14 to 20 ms/token (62.10 and 67.86 with it on, 80.89 and 81.69 with it
off), which is the cold-row case the prefetch exists for. On the other five subjects off is level or
slightly ahead. The mean carries that one subject and the median discards it.

Compare only within pairs. Runs 3 and 4 are faster than runs 1 and 2 on nearly every subject
regardless of the setting, which is the page cache warming across the session.

## Decode speed varies 2x to 3x with prompt subject

This is the most consequential caveat in this file, and it applies to every other number in it.

Same process, same settings, same drafter, ~1900 prompt tokens and 255 new tokens per generation.
Only the subject of the prompt changes:

| Subject | ms/token (4 runs) | tok/s |
|---|---|---:|
| Legal systems and precedent | 31.38, 32.05, 32.09, 33.10 | 30.2 – 31.9 |
| Orbital mechanics | 31.62, 31.71, 31.77, 38.75 | 25.8 – 31.6 |
| Bread fermentation | 32.76, 33.41, 33.87, 34.59 | 28.9 – 30.5 |
| History of computing | 41.63, 42.04, 46.88, 48.76 | 20.5 – 24.0 |
| Marine biology / hydrothermal vents | 62.10, 67.86, 80.89, 81.69 | 12.2 – 16.1 |
| Medieval trade routes | 85.75, 88.33, 92.71, 93.31 | 10.7 – 11.7 |

**31.4 ms/token to 93.3 ms/token in that session**, a factor of 3.0 at identical prompt length and
settings. A later session over the same six prompts, with the page cache warmed by the runs above,
was faster throughout and spanned 31.9 to 69.8 ms/token, a factor of 2.2. Absolute values move with
page-cache warmth between sessions; the grouping does not. Three subjects sit together near 32 ms,
computing sits mid-range, and marine biology and medieval trade are slowest in every run.

### Why: the confidence gate, not draft quality

Instrumenting the same six prompts for draft accounting separates the two candidate explanations.
`tokens_per_round` is total new tokens divided by the number of verify rounds, so it is how many
tokens each forward actually yields. One process, `EXL3_DSPARK_CONF=0.7`, block size 5:

| Subject | ms/token | acceptance | tokens/round | ms/round |
|---|---:|---:|---:|---:|
| Orbital mechanics | 31.93 | 0.986 | 5.447 | 173.9 |
| Bread fermentation | 32.66 | 0.931 | 4.830 | 157.8 |
| Legal systems and precedent | 32.89 | 0.953 | 4.923 | 161.9 |
| History of computing | 45.44 | 0.984 | 3.556 | 161.6 |
| Marine biology / hydrothermal vents | 61.69 | 0.847 | 2.081 | 128.4 |
| Medieval trade routes | 69.84 | 0.930 | 1.869 | 130.5 |

Acceptance is **not** the variable. It stays between 0.85 and 0.99 everywhere and it does not track
speed: medieval trade has higher acceptance than marine biology, 0.930 against 0.847, and is still
the slowest subject of the six. The drafter is not guessing wrong on the hard prompts.

`tokens_per_round` is the variable. It moves 2.9x, from 5.447 down to 1.869, and tracks ms/token
almost exactly inversely, because ms/token is just `ms_per_round / tokens_per_round` and
`ms_per_round` moves only 1.4x. On less predictable text the confidence gate stops the draft block
early, so fewer tokens come out of each forward and throughput falls in proportion. The drafter is
being allowed to guess less, rather than guessing badly.

That also explains why `ms_per_round` *rises* as `tokens_per_round` rises: a longer accepted block
means more drafter steps and a wider verify batch, so each round costs more GPU work while costing
much less per token. This is consistent with the saturation result above.

One open question follows from it. The confidence sweep earlier in this file found 0.7 optimal on a
single prompt, which lands in the fast group. Whether the optimum is subject-dependent was not
measured, and on the slow subjects a lower gate might trade acceptance for block length favourably.

Every headline figure in this file and in `README.md` comes from a single repeated paragraph, which
lands mid-range. Treat the published decode numbers as one point on this distribution, not as a
number your traffic will reproduce. To measure your own prompts, use
[`scripts/prompt_variance.py`](scripts/prompt_variance.py), which produced the table above.

## Speculation is not output-exact

Greedy decoding with the DSpark drafter does not always emit the same tokens as greedy
decoding without it. This is reproducible, it is a property of the speculative path on this
pack, and it is worth knowing before these numbers are used to compare model quality.

The control comes first, because without it nothing here is interpretable. **Plain greedy is
self-deterministic**: two no-drafter runs of the same prompt produced identical token ids on
every prompt tested, across separate processes and model loads. The differences below are
therefore not run-to-run noise.

Twelve prompts, 192 new tokens each, `EXL3_DSPARK_CONF=0.7`, each compared against the same
prompt decoded with no drafter. Scored at the **first** differing token, where both sequences
still share an identical prefix, so the no-drafter distribution at that position is
conditioned on exactly the context the speculative run saw:

| Outcome | Prompts | Meaning |
|---|--:|---|
| Identical output | 4 | speculation was exact |
| Near-tie divergence | 4 | target top-1 and top-2 within 0.032, fp16 tie-breaking |
| Clear-preference divergence | 4 | target preferred its top-1 by 0.137 to 0.699 |

The margins fall into two groups with a 4x gap and nothing in between, so the split is in the
data rather than in the choice of cutoff:

```
near ties          0.006  0.020  0.023  0.032
clear preferences  0.137  0.179  0.587  0.699
```

In the largest case the no-drafter run assigned **0.812** to its top token while the
speculative run emitted one the model gave **0.113**. In another the speculative run emitted
the target's **third** choice at 0.092 against a top-1 of 0.694. Those are not rounding
effects.

**The acceptance logic is not the cause**, which is worth stating because it is the obvious
suspect. A draft token is accepted only when it already equals the token sampled from the
target's own verify logits:

```python
if draft_tokens[j, i].item() != sampled_token.item() or cp_boundary:
    rejected = reject_remainder(job, j, i, batch_states)
else:
    job.accepted_draft_tokens += 1
```

The emitted token is always the target's sampled token, never the raw draft, so speculation
cannot emit something the verify pass did not choose. The difference is upstream: **the
target's logits in the batched verify window differ from its logits in single-token decode**.
Those are not the same call. Verification runs the forward with `recurrent_history=True` plus
the drafter's `draft_verifier_params`, over a multi-token window, on an architecture carrying
recurrent state and a sparse-attention indexer. Which of those is responsible is **not**
established here.

What it means in practice:

- Throughput figures in this file stand. They measure how fast tokens are produced, and that
  is unaffected.
- Any A/B that perturbs the numeric path can end up comparing different generated texts. That
  is exactly what confounded the confidence-gate table above, and it is why output hashes
  belong in any future comparison on this stack.
- If you need output identical to the target model, run without the drafter and accept
  15.13 to 15.22 tok/s.

Reproduce with [`scripts/spec_exactness.py`](scripts/spec_exactness.py).

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
| `EXL3_INT8_GEMV=0`, disabling the int8 GEMV path entirely | paired in one session: 32.57 tok/s baseline, 30.37 with it off, **−6.8%** | keep the default; this lever inverts on this pack |
| `EXL3_INT8_GEMV=1` (residual int8 mode; the default is 2, plain int8) | 31.90 vs 32.50 baseline, −1.8% | keep the default |
| `EXL3_MGEMM_N_THRESHOLD` lowered to 2048 / 4096 from its 8192 default | 32.59 / 32.46 against a 32.50 baseline, +0.3% / −0.1% | null; inside run-to-run noise |
| `EXL3_ENGRAM_ATS=0` | 32.45 vs 32.50 baseline, −0.2% | null; inside run-to-run noise |
| Pinning the process to the big-core cluster with `taskset` | 32.83 vs 32.57 baseline, +0.8%; combined with `EXL3_INT8_GEMV=0` it reaches only 30.84, still well below baseline | null; inside noise, and it cannot rescue the int8 result |
| Porting a batched-MTP-verify, device-resident draft chain and GPU-side embedding change set from a sibling EXL3 recipe | four paired runs in one session: 32.60 unpatched, 32.54 patched, 32.55 patched with its own knobs off, 32.60 patched again | **null**; reverted, not carried into this recipe |
| The cooperative fused-MoE kernel (`exl3_moe_coop`) | it takes one `Kg` / `Ku` / `Kd` per launch; 834 of 15360 experts in this pack have `Kg != Ku`, and gate widths span 4 distinct K values inside a single layer, so no single launch can cover a layer | **structurally closed to this pack** |

The grouped-MoE result is the important one: an exact per-slot mgemm loses to the int8 GEMV path on
this hardware, so reducing kernel launch count did not help.

The last seven rows came from porting every portable tuning lever off a sibling ExLlamaV3 EXL3
recipe running a different model on this same hardware. **None of them transferred.** The int8 GEMV
switch transferred with its sign reversed, costing 6.8% here; the threshold, affinity and ATS knobs
were null; the source-level change set was null across four paired runs; and the cooperative MoE
kernel cannot be called on this pack at all. The one lever that did pay is in "Engram row prefetch"
above, and it was found here rather than ported.

Mixed-K is why, and it is worth stating plainly because it closes a whole class of future work. This
pack stores every expert at its own bit width. That is what buys 1.59 bpw at usable quality, and it
is also what makes every uniform-width fast path in the library unreachable: the fused MoE path, the
uniform-K repack, and the cooperative kernel all require one quantization per launch. A recipe for a
uniform-K pack will hand you levers that this pack structurally cannot use, so measure before
porting rather than after.

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
