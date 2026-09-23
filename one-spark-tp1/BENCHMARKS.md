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

### Re-measured with output hashes: off wins on five of six subjects

A second, larger measurement with `scripts/ab_arms.sh`, which records the output token ids of
every generation and alternates arms strictly (A, AP, A, AP, one process each). It was taken on
the **second Spark** with the pack read over NFS from the first (runtime identity below), so
compare its rows to each other and not to the tables above.

Varied mode, the same six subjects, drafter on, two rounds per arm, median ms/token:

| Subject | prefetch on (default) | prefetch off | off faster by |
|---|---:|---:|---:|
| orbital | 37.95 | 32.38 | **14.7%** |
| bread | 36.32 | 31.71 | **12.7%** |
| law | 35.24 | 31.34 | **11.0%** |
| marine biology | 43.82 | 39.30 | **10.3%** |
| medieval trade | 65.77 | 59.38 | **9.7%** |
| computing | 49.69 | 49.80 | -0.2% |

Computing is the first generation in each process and swings 40 to 60 ms/token in both arms, so
it carries no signal either way. **Token ids were identical between the arms on every subject**
and each arm was self-deterministic, as the mechanism predicts: the prefetch's return values are
discarded, so it cannot change output. Draft accounting was identical too (acceptance 0.946,
4.622 tokens/round); only the cost of a round moved, from 175.4 to 157.3 ms.

Repeated prompt, same session and hardware, drafter on: 34.61 ms/token (29.2 tok/s warm) with the
prefetch, **32.57 ms/token (30.7 tok/s) without, +5.9%**, ids identical.

This supersedes "a wash" for warm-to-lukewarm traffic. It does **not** establish that off is free on
genuinely cold rows: the second round of each arm ran with rows the first round had paged in, and
the marine-biology result in the earlier table is the cold-row case the prefetch exists for. The
default is unchanged; see `README.md` for when to turn it off.

Runtime identity for this subsection and for "Measured 2026-09-23" below (`AGENTS.md` rule 8):
second DGX Spark (GB10, 128 GB, ATS addressing mode), driver 580.159.03, the pack
`vcruz305/DSV4.1-Flash-SAGE-EXL3-1.59bpw` at `5dc954019183ab3d994b60433256001a3f1780e7` plus the
`exllamav3/` overlay, **read over NFS (CX7) from the first Spark**, which moves Engram and drafter
reads onto the network; native ExLlamaV3 `954a8ca`, built in place for `12.1a` with CUDA 13.0 and
torch 2.13.0+cu130 on Python 3.12; TP1, `CTX=6144`, `CHUNK=2048`, `max_batch_size=1`,
`EXL3_ATS_MMAP=1`, `EXL3_ATS_COPY='^(?!mtp\.)'`, DSpark at `EXL3_DSPARK_CONF=0.7` block 5
where the drafter is on; per-expert mixed-K MoE path (K1 to K6).

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
cannot emit something the verify pass did not choose. The difference is upstream, and it has
since been isolated.

### The cause: decode and verify run different kernels

The EXL3 int8 GEMV path is gated on row count. `exl3_gemv_int8.cu:127` returns false for
`size_m > 2` (and line 254 tightens that to `size_m <= 1` in residual mode), and
`exl3_gemm.cu:182` dispatches to it only when that gate passes. So the row count decides the
kernel:

| path | rows (`size_m`) | kernel |
|---|--:|---|
| plain decode | 1 | int8 GEMV |
| verify, `num_draft_tokens = 1` | 2 | int8 GEMV |
| verify, `num_draft_tokens >= 2` | 3+ | `exl3_gemm` / `exl3_mgemm` |

Sweeping the verify width matches that boundary exactly. At 2 verify rows there are **no**
clear-preference divergences, only benign near-ties. At 3 rows and above, three of four
prompts flip to clear preference with **identical** first-difference indices and margins at
widths 2, 3 and 5 (0.699 at index 2, 0.587 at index 96, 0.179 at index 5). The effect switches
on at the gate and does not worsen with more rows, which is a discrete dispatch change rather
than an accumulating error.

Disabling the GEMV path confirms it from the other side. With `EXL3_INT8_GEMV=0` the
**no-drafter baseline itself changes** on 3 of 4 prompts: plain greedy decode, no speculation
involved, produces different output hashes. That is direct evidence the two kernels do not
agree. And with both sides then on the same kernel, clear-preference divergence is absent at
width 1, exactly as it is when both sides are on GEMV.

So the rule is simple: **clear-preference divergence appears when the two sides use different
kernels, and not otherwise.**

One consequence deserves stating plainly, because it inverts the obvious reading.
`exl3_gemv_int8` consumes **int8-quantized activations**; the GEMM path does not. The
lower-precision kernel is therefore the one serving *plain decode*, while the speculative
verify window runs the higher-precision path. On that reading the divergent positions are
places where int8 activation quantization changes the argmax, and the no-drafter output is not
automatically the more faithful of the two.

### Measured: the GEMM path is the more accurate one

An operator-level comparison settles it. Real activations were captured from six live
`LinearEXL3` modules during a forward, then the same input row was pushed through each kernel
and compared against `reconstruct_hgemm`, which dequantizes the trellis to a dense weight and
does an ordinary hgemm with **no activation quantization**. Same weights, same row, same
process, one reference, so kernel noise is the only variable.

| module in_features | GEMV SQNR | GEMM SQNR | gap | GEMV mean rel err | GEMM mean rel err |
|---:|---:|---:|---:|---:|---:|
| 5120 | 46.99 dB | 67.08 dB | 20.1 | 3.3% | 0.23% |
| 1280 | 46.72 dB | 69.66 dB | 22.9 | 2.8% | 0.12% |
| 5120 | 51.86 dB | 65.99 dB | 14.1 | 1.1% | 0.21% |
| 4096 | 39.61 dB | 65.48 dB | 25.9 | 4.8% | 0.17% |
| 4096 | 37.66 dB | 64.99 dB | 27.3 | 17.0% | 1.2% |
| 4096 | 38.91 dB | 65.12 dB | 26.2 | 3.8% | 0.15% |

**The GEMM path is closer to the reference in 6 of 6 modules, by 14 to 27 dB.** Cosine error
tells the same story: GEMM lands between 0 and 1.8e-7, GEMV between 3.2e-6 and 8.5e-5. For
scale, `tests/deepseek_v41/moe_grouped_equiv.py` records ordinary fp16 kernel differences at
around 1e-3 relative. GEMM sits inside that band; GEMV sits 10x to 170x above it.

So the int8 GEMV path is not merely a different approximation, it is a **measurably worse**
one, and it is the path serving plain single-token decode. Two caveats on the method: the
reference is itself an fp16 hgemm rather than an fp32 oracle, so these are relative rankings
rather than absolute error; and `max_rel` is not quoted because near-zero reference elements
make it meaningless (values above 90 appear), which is why SQNR and mean relative error are
the metrics used.

### The residual mode is the setting you want

`EXL3_INT8_GEMV` takes three values, and the default is 2 (`exl3_gemv_int8.cu:27`). Mode 1 is
the residual int8 path, which narrows its own gate to `size_m <= 1` rather than 2. Running the
same comparison under each mode, against the same reference:

| mode | decode tok/s | cost | SQNR range | mean rel err |
|---|---:|---:|---:|---:|
| 2, default | 32.50 | baseline | 37.7 to 51.9 dB | 1.1% to 17% |
| **1, residual** | **31.90** | **−1.8%** | **65.8 to 69.3 dB** | **0.10% to 0.33%** |
| 0, off (falls through to GEMM) | 30.37 | −6.8% | 65.0 to 69.7 dB | 0.12% to 1.2% |

Mode 1 gains **17 to 28 dB** over the default and reaches parity with the GEMM path, beating it
in 5 of 6 modules and carrying a lower mean relative error, at a quarter of the throughput cost
of turning the path off entirely. On per-token numerical accuracy, mode 1 is the best value of
the three.

### But no mode makes drafted and non-drafted output agree

Per-layer accuracy and end-to-end agreement turn out to be different properties, and it is
worth recording that the obvious inference from the table above is wrong. Re-running the
12-prompt exactness comparison under each mode:

| mode | exact | near-tie | clear preference |
|---|---:|---:|---:|
| 2, default | 4 | 4 | **4** |
| 1, residual | 6 | 1 | **5** |
| 0, off | 5 | 4 | **3** |

Mode 1 has the best per-layer SQNR yet the *most* clear-preference divergences, and mode 0 does
not clear them either. Two causes, both mechanical:

1. **Mode 1 widens the split rather than closing it.** Its gate is `size_m <= 1`, so decode
   (m=1) runs residual GEMV while every verify window (m>=2) runs GEMM. The default at least
   shares the kernel at m=2.
2. **Mode 0 does not unify the paths.** `exl3_gemm.cu:220` dispatches a second, QTIP-style GEMV
   for small m through `exl3_gemv_try_launch`, with its own env mode at `exl3_gemv.cu:29`.
   `EXL3_INT8_GEMV` governs only the *int8* GEMV, so at m=1 decode still lands on a GEMV
   kernel, just a different one.

Underneath both is the reason small numerical differences do not stay small here: this
architecture makes several **discrete** decisions per token, MoE expert top-k, DSA block top-k,
and the final argmax. Any difference between two kernels can flip one of those and send the two
runs down different trajectories. Two individually accurate kernels still disagree.

**Practical reading.** Set `EXL3_INT8_GEMV=1` if you want the most accurate decode arithmetic
for 1.8%, which is a real and cheap gain. Do **not** expect any value of this variable to make
a drafted run reproduce a non-drafted one; that is not what it controls. Making the paths truly
identical would need the QTIP GEMV disabled as well, which is untested here and is the open
thread on this topic.

### Three regimes, not two

`AUTO_RECONSTRUCT_THRESHOLD` is 144, so the dispatch has a second boundary above the one that
causes the divergence:

| rows | path | relative accuracy |
|---|---|---|
| 1 to 2 | int8 GEMV | lowest |
| 3 to 144 | `exl3_gemm` / `exl3_mgemm` | middle |
| over 144 | `reconstruct_hgemm` (dense dequantized weight) | highest |

That reaches past speculation entirely. **Prefill runs chunks far above 144 rows, so prefill
takes the most accurate path while single-token decode takes the least accurate one.** Any
comparison crossing either boundary is comparing kernels, not configurations.

What it means in practice:

- Throughput figures in this file stand. They measure how fast tokens are produced, and that
  is unaffected.
- Any A/B that perturbs the numeric path can end up comparing different generated texts. That
  is exactly what confounded the confidence-gate table above, and it is why output hashes
  belong in any future comparison on this stack.
- For output consistency, the lever is the kernel boundary rather than the drafter.
  `EXL3_INT8_GEMV=0` puts decode and verify on the same path and removes the
  clear-preference class, at a measured 6.8% throughput cost. Running without the drafter at
  15.13 to 15.22 tok/s does **not** by itself give you the higher-precision path, since plain
  decode is the side that uses int8 activations.

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
| The cooperative fused-MoE kernel (`exl3_moe_coop`) | it takes one `Kg` / `Ku` / `Kd` per launch; 834 of 15360 experts in this pack have `Kg != Ku`, and gate widths span 4 distinct K values inside a single layer, so no single launch can cover a layer | closed **to one launch per layer**; see the note below |

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

> **Correction, and an open lead.** "One quantization per launch" does not mean one launch per
> layer. Partition the layer's experts by their `(Kg, Ku, Kd)` tuple and each partition is
> uniform, which is what `vcruz305/exllamav3` added in `785f206` and `5e0ba47`, both **after**
> the `954a8ca` pinned here. Counted from this pack's own tensor headers, **75.7%** of routed
> experts sit in a group of four or more whose gate, up and down K all agree (a compile-time-K
> launch), **21.9%** in a group whose K differ (the runtime-K launch), and only **2.4%** fall
> below the four-expert cutoff and stay per-expert, at 8.6 such groups per layer. So the
> grouped path is reachable on this pack. **It has since been measured, and it is slower**; see
> "Measured 2026-09-23" below. The structural conclusion above was wrong, but the practical one
> stands: on GB10 no fused MoE path tested beats the dense per-expert loop on this pack.

The GEMV sweep is the other one worth reading. Since the decode is GPU-bound, the remaining lever
would have to be the kernels themselves, and the int8 GEMV path is already close to its floor: it
never materializes fp16 at all. The `u32` product of the extracted trellis word and the codebook
constant *is* four int8 codebook values, consumed directly by `dp4a` against int8-quantized
activations, so a 32-weight block costs roughly 8 integer multiplies plus 8 `dp4a`. Reducing that
meaningfully is not a tuning exercise. Note also that the GEMV path is gated to `size_m <= 2`, so it
serves the single-row drafter steps; the 6-row MTP verify runs through `exl3_gemm` / `exl3_mgemm`.

### Measured 2026-09-23: grouped mixed-K dispatch and a pruned draft head

Same runtime identity as "Re-measured with output hashes" under Engram row prefetch (second Spark,
pack over NFS), repeated prompt, 6 generations x 256 tokens per process, two strictly alternating
rounds per arm, measured with `scripts/ab_arms.sh` and scored with `scripts/ab_compare.py`.

The grouped arms run a second tree, `vcruz305/exllamav3` `feat/gb10-ats-load` plus master's MoE
kernel merge (`a11349d`), the `f6e42ec` `EXL3_MIXEDK_LEGACY` knob and local fixes (branch
`wip/tp1-mixedk-legacy`, tip `4fbb88c`), always with `EXL3_GR_INT8=0`.

| Arm | Drafter | median ms/token | vs pinned | Deterministic | ids = pinned |
|---|---|---:|---:|---|---|
| Pinned `954a8ca`, dense per-expert MoE | off | **70.20** | baseline | yes | n/a |
| Grouped per-K-group dispatch (`EXL3_MIXEDK_LEGACY=1`), as shipped | off | 109.33 | **-36% tok/s** | **no**: up to 6 different outputs across 6 identical generations | no |
| Grouped + deterministic slot accumulation (`EXL3_MIXEDK_DET_GROUPS=1`) | off | 94.66 | **-26% tok/s** | yes | no |
| Pinned | on | **34.17** | baseline | yes | n/a |
| Grouped + deterministic slots | on | 87.14 | **-61% tok/s** | no (first generation differs) | no |
| Pinned + pruned draft head (`EXL3_MTP_HEAD_N=65536`) | on | 66.95 | **-48% tok/s** | yes | no |

Three findings, in order of how much they should change future work:

1. **The grouped dispatch loses on GB10 even where it has a compile-time K.** 75.7% of this
   pack's experts get the unrolled instance and it is still 26% slower at one row than launching
   each expert separately. Upstream measured the unified kernel at -15.7% on this pack
   (`13f1c16`); the grouped path is worse, not better. With the drafter on it also cut tokens per
   round from 4.27 to 1.46, which the per-round cost alone does not explain and which was not
   chased further, since the one-row result already rules the arm out.
2. **Upstream's legacy path is not reproducible.** Called without a slot table, `exl3_moe`
   accumulates with atomics, and identical inputs produced up to six distinct generations in six
   runs. The deterministic variant fixes that and was faster than the atomic one, but still
   loses to the dense loop, and its sums run in a different order, so its tokens differ from the
   pinned tree's anyway. The same local branch also fixes the legacy path dropping any expert
   the kernel skipped for exceeding its 128-row window, which only prefill chunks reach.
3. **Pruning the draft head by vocabulary id collapses drafting.** Restricting the drafter's
   argmax to the first 65536 of 129280 ids cut tokens per round from 4.27 to 1.35 at 0.917
   acceptance: the ids this text needs are not concentrated at the low end of DeepSeek's
   vocabulary, so the drafter proposes the wrong tokens and the confidence gate stops the block.
   A frequency-ranked subset might behave differently; an id-range slice does not work here.

Engram prefetch off (above) is the one lever from this round that passed every gate.

## Not measured here

- **Native ExLlamaV3 TP2 / TP4.** No cross-node native numbers exist to publish, so TP2 and TP4
  in this repository remain the vLLM path. Native TP **at `world_size=1` on one Spark is now
  measured and working**; see "Measured: native TP now runs" below. The transport and attention
  analysis that follows was written from reading the runtime, and the measured section corrects
  several of its conclusions. Read them in that order.

  **Multi-host transport is closer than it looks.** `TPBackendNCCL` already exists, is the
  default backend (`model.py:348`), and calls a real
  `dist.init_process_group("nccl", rank, world_size, init_method)`. `EXLLAMA_MASTER_ADDR` and
  `EXLLAMA_MASTER_PORT` are already environment-configurable rather than hardcoded, and there is
  **no CUDA IPC anywhere** in `exllamav3/model/`, so nothing is pinned to one host at the
  memory-handle level. Three things are genuinely host-local: rank is derived from the local
  device list (`world_size = len(active_devices)`, `rank = active_devices.index(device)`, so two
  single-GPU hosts both compute rank 0), the control plane dispatches over
  `multiprocessing.Pipe` to locally spawned processes, and `broadcast`, `gather` and
  `gather_small` all delegate to the shared-memory `TPBackendNative` fallback. Only `all_reduce`
  and `fwd_barrier` are native NCCL. `gather_small` runs **per token** for the argmax, so it sits
  on the critical path.

  **The attention path needs work, but far less than a file-level grep suggests, and an earlier
  revision of this file got that wrong.** `layer_types` is derived from `compress_ratios`, which
  on this pack is `{0: 2, 2: 18, 1: 20}`, giving **2 layers of `DSV4Attention` and 38 of
  `DSV41Attention`**. `modules/dsv41.py` contains zero occurrences of `backend`, `all_reduce` or
  `tp_`, which reads as "38 of 40 layers have no tensor-parallel path". That conclusion was
  wrong, because it grepped the file rather than the class hierarchy.

  `DSV41Attention` **subclasses `DSV4Attention`** (`dsv41.py:145`) and defines no `forward` of
  its own, so it inherits the parent's, which already carries the collective:

  ```python
  if self.num_q_heads == 0:
      # Zero-width TP shard: contribute nothing, keep the collective aligned
      y = torch.zeros_like(x, dtype = out_dtype or self.out_dtype)
      if self.tp_reduce:
          params["backend"].all_reduce(y, False)
      return y
  ...
  if self.tp_reduce:
      params["backend"].all_reduce(y)
  ```

  It also overrides none of `tp_export`, `tp_import` or `make_tp_allocation`, so it uses the
  parent's. Those split only `q_b`, `wo_a`, `wo_b` and the sinks over
  `channels_to_split = o_groups`, and **replication of everything else is the deliberate
  design**, stated in the source:

  > Everything KV-side is shared-MQA and replicated per rank (q_a, wkv, norms, compressor,
  > indexer, pools, rings); only q_b / wo_a / wo_b / sinks split

  The parent's allocation already counts `idx_wq_b`, `idx_weights`, `compressor` and `indexer`
  by name, and its `tp_export` exports all four. That has a useful consequence: because the
  indexer is replicated, the DSA top-k runs on complete scores on every rank, so no score
  all-reduce is required and the distributed-top-k problem never arises. Cross-layer KV and
  index sharing is consistent for the same reason.

  The genuine gaps are therefore narrow and specific:

  1. `tp_export` hardcodes `"cls": DSV4Attention`, so an import would rebuild the wrong class.
  2. Its `kwargs` omit the DSV4.1-only constructor arguments (`is_kv_source`, `is_index_source`,
     `kv_source_layer`, `index_source_layer`, candidate role).
  3. `idx_wk` and `idx_k_norm` (`dsv41.py:195,197`) are absent from the export list.
  4. `DSV41Compressor` (`dsv41.py:71`) is a standalone class, not a subclass of the DSV4
     compressor, and its methods are `__init__`, `modules`, `project`, `pool` only, so it has no
     `tp_export` / `tp_import` of its own.

  That is a `tp_export` / `tp_import` override on `DSV41Attention` plus an export pair on
  `DSV41Compressor`, not a tensor-parallel implementation from scratch. Concretely, reading the
  constructors and the parent's import:

  - `DSV41Attention.__init__` is
    `(config, key, layer_idx, compress_rate, is_kv_source, is_index_source, kv_source_layer,
    index_source_layer, candidate_role, candidate_topk_blocks, candidate_block_size, ref_quant,
    select_hq_bits, qmap, **kwargs)` and calls `super()` with `layer_type="v41"`. A
    `tp_export` override has to carry those DSV4.1-only arguments, which the parent's `kwargs`
    block does not.
  - It already early-returns on `num_q_heads == 0`, so the zero-width shard case the parent's
    collective expects is **already handled**.
  - Its submodules are conditional: `compressor` on `is_kv_source`, `idx_wq_b` / `idx_weights`
    on `is_index_source`, `idx_wk` / `idx_k_norm` on `owns_index_keys`. An import has to
    reproduce those conditions rather than assume all are present.
  - The parent's `tp_import` injects pre-built submodules as constructor kwargs and passes
    `tp_defer_compressors=True` to stop `__init__` rebuilding them. `DSV41Attention.__init__`
    does not honour that flag, so it would overwrite injected modules; it needs the same guard.
  - `DSV41Compressor.__init__` does **not** accept injected `wkv` / `wgate` / `norm` the way
    `DSV4Compressor` does, and `wgate` is `None` when `compress_rate == 1`. Both need handling.
  - The parent's `tp_import` hardcodes `DSV4Compressor.tp_import` for both `compressor` and
    `indexer`, so a DSV4.1 import must dispatch to `DSV41Compressor` instead.

  Validation does not need a second machine. A single-device backend running N logical ranks
  sequentially with real reductions, compared against the unsharded module on the same
  captured activations, would exercise all of the above. `tests/test_cpu_cache_tp.py` already
  establishes the pattern of standing in for rank workers with plain dicts, and
  `scripts/kernel_accuracy.py` in this folder shows the capture-and-compare method, with the
  ~1e-3 fp16 noise floor as the pass threshold.

  ### Measured: native TP now runs on one Spark

  Everything above this heading was written from reading the runtime. It is kept because the
  transport analysis held up, but its conclusions about the attention path were incomplete and
  its status verdict is superseded. What follows is measured.

  **Native tensor parallelism loads and generates correctly on a single Spark at
  `world_size=1`**, after fourteen fixes to the engine. Clean tree, 64 tokens, greedy, no
  drafter, 12-token prompt:

  | path | backend | decode | load |
  |---|---|---|---|
  | native TP, `world_size=1` | NCCL | **11.98 tok/s** | 87-90 s |
  | no TP (same tree) | n/a | **12.54 tok/s** | 37 s |

  TP costs about 4.5% here, which is the expected shape: real collective overhead with no
  parallelism to offset it at one rank. **This is a plumbing validation, not a performance
  result.** It exercises export, import, spawn, dispatch and the NCCL collective path. It does
  **not** exercise the splitting arithmetic, because at one rank every split spans the full
  range. Two Sparks would still be needed for that, and the single-host limit described in the
  README is unchanged.

  **Use the NCCL backend, not the default.** `tp_backend` defaults to `"native"`
  (`model.py:268`), whose `all_reduce` routes through `ext.pg_all_reduce_cpu` and hits
  `TORCH_CHECK(is_avx2_supported())` at `all_reduce_cpu.cu:544`. AVX2 is x86 and GB10 is
  aarch64, so the native backend cannot reduce on a Spark at all. Pass `tp_backend="nccl"`:
  `TPBackendNCCL.all_reduce` is `dist.all_reduce` on the GPU and touches no CPU SIMD path. NCCL
  is also the only backend that could ever span two hosts, since `TPBackendNative` coordinates
  through named shared memory.

  **Correctness was established numerically, not from fluent-looking text.** A run that loads,
  returns a full token count and a plausible tok/s can still be wrong: an earlier state of this
  work produced `'1. The history of computing is a history of abstraction. 2. ...'` counting
  upward, at 13.78 tok/s, with no error anywhere. Both paths are byte-deterministic on this
  stack (four no-TP runs share one output hash, three TP runs share another), so output hashing
  is a valid check here. TP is numerically equivalent to within fp16 reduction error, layer-0
  attention output 0.559040 on both with std differing by ~4e-6, but it is **not bit-exact**,
  and on this architecture the discrete per-token choices (MoE top-k, DSA block top-k, argmax)
  turn that into a different but equally fluent greedy trajectory.

  **The bug that mattered most was one missing dictionary key.** `tp_export` enumerated sixteen
  of `DSV4Attention.__init__`'s parameters by hand and omitted `q_head_norm`, which defaults to
  `True` while every DSV4.1 layer is constructed with `False`
  (`architecture/deepseek_v41.py:170` and `:201`). Workers therefore rebuilt attention with the
  default, `_rope_qkv` passed `q_ones` instead of `None`, and every query head received an
  unweighted RMS norm the architecture does not use. Query reached the attention kernel at
  std 0.999784 instead of 2.020987. Nothing raised. **A hand-enumerated parameter whitelist
  fails silently and numerically rather than loudly; diff the constructor signature against the
  exported kwargs.**

  **Two silent fallbacks made this expensive to find**, and they are worth knowing about for any
  work in this file. `dsv4.py` catches `AssertionError` around the fused-runner construction and
  nulls both runners; `attention_fn/bc_dsa.py` catches bare `Exception` in `build_bc_dsa` and
  declines. Neither logs anything. The second one fires under TP because `bc_dsa.py:89`
  dereferences `rs.cache.max_num_tokens` while TP has replaced the cache with `id(cache)`
  (`model_tp.py:570-571`), so TP silently runs the eager attention path where no-TP runs the
  fused one. The eager path is correct; it was verified by forcing `EXL3_BC_DSA=0` with no TP.

  **Class preservation across the TP boundary was a recurring defect, at four sites.**
  `tp_export` hardcoded a concrete class where a DSV4.1 subclass was required, so workers
  rebuilt the parent. Walking the class hierarchy rather than recalling which subclasses had
  already caused trouble found all of them: `DSV4Attention` (`dsv4.py`), `DSV4LayerState`
  (`cache/dsa.py`), `TransformerBlock` (`transformer.py`) and `HyperConnection`
  (`hyperconnections.py`). The `TransformerBlock` one is the instructive case: it sat in an
  earlier sweep of all 40 `"cls":` sites and was dismissed, because the sweep was filtered
  through an assumption about which classes had subclasses instead of a check.

  **A TP worker never calls `load()`.** It is built by `tp_import` and receives weights over
  shared memory, so anything `load()` would have done has to be reproduced. That single cause
  produced three separate failures here: the Engram table needing a per-rank
  `Config.from_directory`, the recurrent layers needing to exist, and those layers needing
  `alloc()` (their state starts on `meta`). Most modules get this for free because their
  `tp_import` ends with `module.load_local(device)`; `EngramLayer` has no `load_local`.

  The fix set is eight files, 20 hunks: `cache/dsa.py`, `modules/dsv4.py`, `dsv41.py`,
  `dsv41_hc.py`, `engram.py`, `hyperconnections.py`, `module.py`, `transformer.py`.

  ### Superseded: where a TP load used to break first

  The account below was accurate when written and is kept for the method, not the verdict. Its
  closing status of "untested" is now wrong, and its advice to set `supports_tp = False` should
  **not** be followed: that would disable a path which works.

  Rather than continue reasoning from source, a tensor-parallel load was attempted on a single
  device with no code changes (`tensor_p=True`, `use_per_device=[80.0]`, one GPU, so nothing is
  actually split). It is **not** refused, because `supports_tp` is `true` for this
  architecture, and it runs for 0.88 s, well past argument validation and into real work,
  before failing:

  ```
  AttributeError: 'NoneType' object has no attribute 'storage_size'
  ```

  That is `make_tp_allocation`, at its one call against a possibly-absent submodule:

  ```python
  for comp in (self.compressor, self.indexer):
      if comp is not None:
          storage_dev += comp.wkv.storage_size() + comp.wgate.storage_size()
  ```

  `DSV41Compressor` sets `self.wgate = Linear(...) if self.gated else None` with
  `gated = compress_rate > 1`. This pack's `compress_ratios` histogram is `{0: 2, 2: 18, 1: 20}`,
  so **20 of 40 layers have ratio 1 and carry no `wgate` at all**. The parent's allocation
  assumes one exists. The `comp is not None` guard covers a missing compressor but not a missing
  gate.

  So the first concrete fix is a None-guard on `comp.wgate` in `make_tp_allocation`, and this
  confirms by traceback the `DSV41Compressor` gap that was previously only inferred from
  reading. It does not tell us what breaks *next*; each fix reveals the following one.

  Two practical notes for anyone repeating this, both learned the hard way:

  - A script that triggers a TP load **must** use `if __name__ == "__main__":`. TP sets the
    multiprocessing start method to `spawn`, which re-imports the main module in each child, so
    an unguarded script silently executes itself twice.
  - A failed TP init is **not** idempotent. It leaves `mp_children` populated, so a retry trips
    `assert not self.mp_children` in `create_tp_context`, and leaks shared-memory objects.
    Call `unload_tp()` before attempting again.

  **A note on how this assessment moved.** It has been revised three times as the reading got
  deeper, each time downward: first "no TP path at all" (from grepping the file, missing that
  the class inherits `forward` from `DSV4Attention`), then "the DSV4.1 submodules are outside
  the allocation" (missing that the parent already names them), and now the four items above.
  Treat the first two framings as superseded.

  **So the status is "untested", not "absent".** What remains unverified is whether the
  inherited allocation composes with DSV4.1's recurrent state and its cross-layer KV and index
  sharing (`kv_source_layer_ids`, `index_source_layer_ids`). That is exactly what a
  single-device multi-rank simulation would settle, with no second machine required.

  **And it fails silently rather than refusing.** `supports_tp` defaults to `True`
  (`model.py:26`), this architecture never overrides it, and `model.py:443` raises
  `NotImplementedError` only when the cap is `False`. A TP load therefore starts and runs layers
  that never reduce their partial sums. Anyone attempting native TP on this pack should set that
  cap to `False` first.

  **TP3 is not representable at all**, independently of any of the above: `num_attention_heads`
  is 64, `hidden_size` 5120, `num_key_value_heads` 1 (MLA compressed latent, unsplittable), and
  none divide by three. vLLM enforces this directly (`vllm/config/model.py:1367`,
  `total_num_attention_heads % tensor_parallel_size != 0`), so the legal sizes are the divisors
  of 64. TP2 divides cleanly everywhere: 64 to 32 heads, 5120 to 2560, 384 experts to 192,
  `moe_intermediate_size` 2304 to 1152.

  **One encouraging detail for anyone costing out the attention work.** The DSA indexer looks
  like it would need a distributed top-k, which would be hard, but it does not.
  `qsa_indexer.py` computes `s = einsum("bshd,bnd->bshn", q, pooled)` then
  `F.relu(s).sum(dim = 2)`, summing over heads **before** the top-k. Since the relu is
  elementwise per head, each rank holds an exact partial sum, so head sharding needs only an
  `all_reduce` of the scores followed by an identical local top-k on every rank. No all-gather
  and no custom kernel work. Note the cross-layer constraint though: `kv_source_layer_ids` and
  `index_source_layer_ids` mean consumer layers reuse a producer layer's latents and selection,
  so any shard layout has to stay consistent across that dependency.

  The MTP drafter sets `supports_tp: False`, but `attach_to` has an explicit
  `if target.loaded_tp: self._load_own_embed_head()` branch, so it is designed to run unsharded
  beside a tensor-parallel target rather than being unusable.
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
