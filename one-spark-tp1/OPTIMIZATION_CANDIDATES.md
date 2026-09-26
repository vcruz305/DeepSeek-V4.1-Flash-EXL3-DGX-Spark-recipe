# Candidates for a faster TP1 decode, and what each one would have to prove

> **Update 2026-09-23: candidates 1, 6 and 9 have been measured.** The grouped dispatch
> (candidate 1, and C2 with deterministic slots) is **26 to 61% slower** and upstream's variant is
> not reproducible; the pruned draft head (candidate 6) **collapses drafting** from 4.27 to 1.35
> tokens per round; `EXL3_ENGRAM_PREFETCH=0` (candidate 9) **passed every gate**: +5.9% on the
> repeated prompt and +9.7% to +14.7% on five of six varied subjects, output byte-identical. The
> numbers and runtime identity are in `BENCHMARKS.md`. Candidates 2 to 5 depend on the grouped
> path and are now low priority; 7 and 8 remain open.

**Nothing below this note was measured when it was written.** `BENCHMARKS.md` holds the measured record;
this file holds the open leads, the arithmetic behind them and the gates each one has to clear.
Every figure here is either quoted from another repository's commit (cited inline), computed
from the checkpoint's own tensor headers, or labelled an estimate. Per `AGENTS.md` rule 7, a
candidate is not a result.

The baseline being attacked is the warm repeated-prompt figure, 32.5 to 33.4 tok/s, not the
17.53 fresh-prompt median. A lever must be paired against the same prompt in the same session.

## The claim this file corrects

`BENCHMARKS.md` records the cooperative fused-MoE kernel as **"structurally closed to this
pack"**, and concludes that mixed-K "closes every uniform-width fast path in the library." That
was true of the library at the pinned revision. It is **no longer true upstream**, and the
checkpoint is far friendlier to grouped dispatch than that line implies.

Counted from the pack's own safetensors headers (trellis last dim / 16 = K, 40 routed layers x
384 experts):

| Expert population | Share | Dispatch it can reach |
|---|---:|---|
| Groups of >= 4 experts sharing `(Kg, Ku, Kd)`, all three equal | **75.7%** | `exl3_moe` with a **compile-time** K instance (unrolled trellis loop) |
| Groups of >= 4 experts whose K differ across gate/up/down | **21.9%** | `exl3_moe` with `K = 0`, the runtime-K instance (no unroll) |
| Groups smaller than the `len(experts) < 4` cutoff | **2.4%** | per-expert fallback, as today |

Per layer that is 11 to 22 distinct `(Kg, Ku, Kd)` tuples, of which **8.6 on average** hold four
or more experts. 93.8% of experts sit in a group with `Kg == Ku`. K itself is heavily skewed:
27210 projections at K1, 11811 at K2, 5774 at K3, 1221 at K4, 62 at K5, 2 at K6.

The MTP drafter is a separate case and a simpler one: its 3 x 128 experts are **uniform K=4**,
one tuple per layer, so the drafter's MoE layers are eligible for the stock fused path with no
mixed-K machinery at all.

The upstream work that makes this reachable, all of it **after** the pinned `954a8ca`
(`vcruz305/exllamav3`):

| Commit | What it adds |
|---|---|
| `785f206` | per-K-group fused dispatch for mixed-K packs (38 -> 60 tok/s on sm_120) |
| `5e0ba47` | `exl3_moe_mixedk`, one launch per layer, per-expert K from device arrays (84 -> 90 on sm_120); reports token ids identical to the legacy path |
| `329e051` | drops the per-layer `.tolist()` host readback for the unified path (+2%) |
| `f6e42ec` | `EXL3_MIXEDK_LEGACY=1`: on **GB10 (sm_121)** the grouped path measured +9% over unified on Qwen3.8-Flash-Next 4.06bpw, the grouped coop kernel being ~2.9x cheaper per call (216+115 us vs 953 us) |
| `ecb5013`, `13f1c16` | `EXL3_MOE_MIXEDK_MIN_ROWS`, a per-call row floor so decode and prefill can take different paths |

One datapoint exists on **this** pack on GB10, in `13f1c16`: with the unified kernel at its
default, decode fell from 12.574 to 10.600 tok/s (**-15.7%**), TTFT and load time unchanged. The
isolation runs that would have told us whether the *legacy grouped* path recovers that were
never finished; that commit records the test model being removed from the box first. **That
unfinished experiment is the first thing to run here.**

## Why dispatch, and not bandwidth, is where the headroom is

Bytes the resident main model must read per decode token, from the same headers (the
`exllamav3/` overlay path where it exists, active experts only):

| Component | Per token |
|---|---:|
| Attention (overlay trellis: `wq_b`, `wo_b`, `wo_a`, `wq_a`, `wkv`, indexer, compressor) | ~3.06 GiB |
| Routed experts, 6 of 384 per layer x 40 layers at ~6.76 MiB each | ~1.58 GiB |
| `head`, fp16, 129280 x 5120 | 1.23 GiB |
| Shared experts (overlay) | 0.83 GiB |
| Engram `wkv`, router gates, norms | ~0.48 GiB |
| **Total** | **~7.2 GiB** |

A verify round at 6 rows reads the dense part once and roughly 20 to 25 distinct experts per
layer, so **~11 GiB**, plus the drafter's own ~2.7 GiB (3 layers of uniform-K experts, its
attention, and the shared trunk head again). At GB10's ~250 GB/s that is an estimated **~55 to
60 ms per round**, against the **130 to 175 ms per round** `BENCHMARKS.md` measures. The GPU is
saturated in the sense that injected work is paid immediately, but it is spending roughly two
thirds of each round on something other than moving weights: with ~620 kernel launches per
token, a round issues on the order of 3000 launches, ~53 us apiece.

That is the gap the fused dispatch aims at, and it is consistent with the sm_120 results above,
where the same change was worth 2.4x. It is **not** consistent with the recipe's earlier reading
that only "a smaller pack, higher acceptance or faster kernels" remain: fewer, larger launches
is a fourth option that the pinned tree could not express.

## Candidates, ranked

Priority is expected payoff divided by the risk of changing the output. Every one of them is
unmeasured on this pack.

### 1. Per-K-group fused dispatch (`EXL3_MIXEDK_LEGACY=1`), integration tree

Three quarters of the routed experts get a compile-time-K fused launch instead of a per-expert
dense loop. This is the arm with the structural argument and the sm_121 precedent behind it.

**Two defects were found in that path while reading it**, both fixed on the local branch
`wip/tp1-mixedk-legacy` in the integration checkout and both **unverified on hardware**:

- *Dropped experts (`e76c0e0`).* The dispatch passes `count_lo=1, count_hi=TEMP_ROWS_FUSED`, so
  `exl3_moe` skips any expert outside that row window, but every group expert was added to
  `mixedk_handled` regardless, and the per-expert loop skips everything in that set. A skipped
  expert therefore contributed nothing to the routed sum. Decode (6 rows) and MTP verify (36)
  never put 128 rows on one expert; a prefill chunk does, so this would have corrupted the A/B
  through the prompt rather than the generation.
- *Non-deterministic accumulation (`b715d5f`, gated by `4fbb88c`).* The group launches pass
  `output_scratch=None`, which the kernel's own comment describes as atomically adding into the
  token row "in arrival order, which is not bit-reproducible run to run". Without this the arm
  fails the same gate `EXL3_MOE_MIXED_BSZ1` failed, however fast it is. The fix gives each group
  the same `FUSED_DET` slot table the fused and unified tiers use, with the group's slot bases
  gathered on the device from the layer's table, so no host round trip is added. It sits behind
  `EXL3_MIXEDK_DET_GROUPS`, **default off**, so arm C measures the path as upstream ships it and
  arm C2 prices what determinism costs. Measuring a modified path and reporting it as the path
  is exactly what `AGENTS.md` rule 8 exists to prevent.

**One hazard remains to measure, not assume.** Each active group still builds
`torch.tensor(counts)` (a pageable H2D) plus two `torch.cat` calls per layer before its launch.
At ~6 active groups per layer at decode and ~10 at verify, that is several hundred host round
trips per forward against the pinned tree's 40. Candidate 4 below removes them if this arm is
close but not ahead.

### 2. Row floor (`EXL3_MOE_MIXEDK_MIN_ROWS=N`), integration tree

`bsz * top_k` is **6** at decode and **36** at MTP verify. A floor in 7..36 puts decode on the
legacy grouped path and verify on the unified kernel; a floor above 36 sends both to legacy by a
different route than candidate 1. The commit that added the knob frames it as decode-vs-prefill;
on this recipe the interesting boundary is decode-vs-verify, and 36 is the number.

### 3. Unified-kernel launch knobs (`EXL3_MK_BPS`, `EXL3_MK_SHPIPE`, `EXL3_MK_SMEM`, `EXL3_MOE_MIXEDK_CMAX`)

Only meaningful in arms where the unified kernel actually runs, so they follow candidate 2, not
candidate 1. `f6e42ec` sizes dynamic shared memory to the shape instead of always reserving
`SMEM_MAX`, which on sm_121 (99 KB per block) is what caps the kernel at one block per SM; its
own comment records `EXL3_MK_BPS=2` measuring identical on GB10, so treat the grid knob as a
sweep, not an expected win.

### 4. Group-major sort: a sync-free grouped dispatch (code, designed, not written)

Determinism is already handled by `b715d5f` above. What remains is the host traffic, and if
candidate 1 measures close but not ahead, this is the next thing to try. It is contained to
`block_sparse_mlp.py`:

- at load, assign every expert a **position** in a group-major layout (group 0's experts, then
  group 1's, ..., then the ungrouped tail), and keep `position -> expert` for the fallback loop;
- at forward, sort the flattened assignments by `position[expert]` instead of by expert id. Each
  group's rows are then a **contiguous slice** of `token_sorted` / `weight_sorted` (a view, no
  `cat`), and its per-expert counts are a contiguous slice of `bincount` (a device view, no
  H2D). The kernel reads `num_experts = size(0) - 1`, so the next group's first count serves as
  the sentinel it never dereferences;
- the slot table per group is already wired (`b715d5f`), and stays as it is.

Net: one host sync per layer (the existing `.tolist()`, which `BENCHMARKS.md` already measured
as costing no wall time), no per-group H2D and no `cat`.

The per-expert fallback loop walks `expert_count_list` cumulatively, so it has to walk the same
group-major order and map back through `position -> expert`. That is the part to get right.

### 5. `(Kgu, Kd)` kernel instances for the mixed tuples

`exl3_moe` sets `K = K_gate` only when gate, up and down K all agree, and falls to the runtime-K
instance otherwise, which is exactly the "cannot unroll the trellis decode loop" cost that makes
the unified kernel lose on sm_121. Since 93.8% of this pack's experts have `Kg == Ku`, a
template taking the gate/up K and the down K separately would move the **21.9%** mixed-tuple
population onto compile-time instances too, putting ~97.6% of experts on unrolled kernels. The
tuples that matter are few: `(2,2,3)`, `(1,1,2)`, `(3,3,4)`, `(2,2,4)`, `(1,1,3)`, `(4,4,5)`.
This is real CUDA work (new instance files, a second template parameter) and belongs after
candidates 1 and 2 have said whether grouped dispatch is the right direction at all.

### 6. Pruned drafter head

`DeepseekV41MTPModel.sample_from_state` runs the **shared trunk head** over every block position,
so the fp16 1.23 GiB head is read **twice per round**: once for the draft block and once for the
verify. `qwen4_exp_mtp.py` already carries `EXL3_MTP_HEAD_N`, which argmaxes over a column slice of the
shared head for drafting only. **Ported to `deepseek_v41_mtp.py` in `9b0a996`** on the local
branch, default 0 (off) and unverified on hardware; arm `E` in `ab_arms.sh` turns it on at
65536. This pack's head is fp16 rather than an EXL3 trellis, and `LinearFP16` stores the weight
as `(in_features, out_features)`, so the slice is a strided view and costs no memory. A 64k
slice of the 129280-token vocabulary halves the draft-side read, an estimated **2 to 3%** of a
round. It cannot change which tokens are emitted (the verify pass still decides every token),
only which ones get proposed, so the risk is acceptance, not correctness. Measure acceptance
alongside tok/s, and compare against arm C rather than arm A.

Two things the slice could get wrong, one already settled. `Linear` rounds both dims up to
`pad_to = 128`, so a padded head would make the slice index into padding rather than vocabulary;
129280 is 1010 x 128 and 5120 is 40 x 128, so **there is no padding here** and `w.shape[1]` is
the real vocabulary. The other is that TP1 on one device leaves `full_out_features ==
out_features`, so `LinearFP16.forward`'s shard-offset handling is inert, but that is read from
the source rather than observed: if arm E's acceptance drops sharply rather than slightly,
suspect the slice before concluding the pruned head costs acceptance.

### 7. A decode-graph path for `DSV41Attention` (code, not designed)

Attention weights are the **largest** per-token read on this pack, ~3.06 GiB against ~1.58 GiB
of routed experts at one row, and `EXL3_DSV4_BATCH_GRAPH` covers only the 2 layers built on
`DSV4Attention`. The other **38** have no graph path, so every q/kv projection, the compressor,
the indexer, the DSA top-k and the output projections are individually launched, 40 times per
forward. `BENCHMARKS.md` records this as "not an open item" on the grounds that there is no
unexploited graph *setting*, which is true, and on the grounds that the device is saturated,
which is the inference this file's per-round arithmetic questions. It is a bigger and riskier
project than the MoE dispatch (recurrent state, cross-layer KV and index sharing, DSA paging all
have to survive capture), so it is listed, not recommended, until `module_timing.py` says what
share attention actually holds.

### 8. Tree speculation: verify more candidates per round (research-grade)

The strongest single fact in `BENCHMARKS.md` is that `tokens_per_round` moves **2.9x** across
subjects (5.45 down to 1.87) while `ms_per_round` moves only **1.4x**. Throughput is
`ms_per_round / tokens_per_round`, so the slow subjects are slow because the confidence gate
stops the draft block early, not because the drafter is wrong (acceptance stays 0.85 to 0.99
throughout).

A round's cost is dominated by reading weights that every verify row shares. Going from 6 rows
to 11 or 12 reads the same experts and the same 1.23 GiB head once, so the marginal cost of
extra candidate tokens is small, which is precisely what the 1.4x-vs-2.9x split says. Verifying
a **tree** of candidates (two or three branches from the first uncertain position) instead of
one chain converts that headroom into accepted tokens on exactly the prompts that are slow
today.

Output-preserving in the same sense linear speculation is: the target still samples every
emitted token; the tree only changes which continuations are offered. The work is real, though:
a tree needs a block-diagonal verify mask, per-branch cache handling and a generator that can
retire a partial branch, and `BENCHMARKS.md` already records that concurrency plus speculation
raises a `RuntimeError` in the recurrent state, which is adjacent to what a tree would stress.
Not a next step; the biggest one on the list if the dispatch arms disappoint.

### 9. `EXL3_ENGRAM_PREFETCH=0`

Already measured and documented: **+3.2%** on repeated prompts, a wash on varied ones, cannot
change output. It is a published variant rather than a default. If the workload is repeated
prompts, it is the one free win available today.

## Ruled out, with reasons

| Lever | Why not |
|---|---|
| Integration tree at its default (unified kernel everywhere) | `13f1c16` already measured **-15.7%** on this pack on GB10. Re-run only as a cheap sanity anchor. |
| `EXL3_GR_INT8` (int8 decode mixer, on by default in the integration tree) | It only touches `GatedResidual`; DSV4.1 uses `DSV41HyperConnection`, which subclasses `HyperConnection` and never reaches that code. Inert here, and it quantizes weights, so set it to 0 in every arm to keep the diff to MoE dispatch. |
| Generator-side batched MTP verify / device-resident draft chain (`4e8133f`) | The same change set was ported and measured **null across four paired runs** in `BENCHMARKS.md`. It rides along in the integration tree; it is not the variable under test. |
| Repacking to uniform K, raising fused row caps, grouped CUDA-graph modes, int8 GEMV modes, `taskset`, huge pages | Measured in `BENCHMARKS.md`. Unchanged by any of the above. |

## The gates a candidate has to clear

1. **Deterministic.** The same arm, the same prompt, twice: identical token ids. This is the
   check the atomic accumulation in candidate 1 can fail.
2. **Output-identical to the baseline arm.** Same prompt, arm A vs the candidate: identical
   token ids. If they differ, `spec_exactness.py` decides whether the divergence is an fp16
   near-tie or a clear preference before any throughput number is read.
3. **Faster on the same subject in the same session.** Per subject, never pooled: decode speed
   spans 2 to 3x across subjects, which is larger than any effect being chased here.

## Running it

`scripts/ab_arms.sh` runs the arms in strict alternation, one process per arm (the dispatch
knobs are read at load), and prints the `ab_compare.py` invocation for the results:

```bash
PIN_SRC=~/dev/vcruz305/exl3-pin \
INTEG_SRC=~/dev/vcruz305/exl3-integ \
MODEL_DIR=/models/DSV4.1-Flash-SAGE-EXL3-1.59bpw \
ARMS="A C C2" ROUNDS=2 MODE=repeat DRAFT=0 \
  bash scripts/ab_arms.sh
```

`DRAFT=0` first, deliberately. Without the drafter a round is one forward of the main model at
one row, so a dispatch change is not diluted by acceptance, draft-block length or the kernel
switch the verify window crosses, and plain greedy is self-deterministic here, which makes the
id comparison exact. Re-run the surviving arm at `DRAFT=1` afterwards, because that is the
configuration the recipe ships.

Prerequisites, neither of which is a code problem:

- **318.3 GiB of local NVMe for the pack** (188.8 GiB of it Engram tables, 101.4 GiB routed
  experts, 10.6 GiB overlay). Engram rows are gathered from the mapping on every forward, so
  network storage is not a substitute.
- **The box to itself**: the load drives `MemAvailable` to roughly 5 GiB, and a second model
  process on the same Spark takes it down.
