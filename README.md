# DeepSeek-V4.1-Flash EXL3 on DGX Spark

Serving and qualification tooling for **DeepSeek-V4.1-Flash EXL3** on NVIDIA DGX Spark / GB10.

> vLLM owns the DeepSeek-V4.1 model graph. `vllm-exl3` owns EXL3 routed-expert storage/execution. ExLlamaV3 supplies EXL3 kernels; it is not the V4.1 graph owner.

## Measured performance

One Spark, `vcruz305/DSV4.1-Flash-SAGE-EXL3-1.59bpw` plus the `exllamav3/` attention/MTP overlay:

| Measurement | Result |
|---|---:|
| Decode, no drafter | **15.13 – 15.22 tok/s** |
| Decode, DSpark drafter, fresh prompt | **17.53 median**, 19.82 mean |
| Decode, DSpark drafter, repeat prompt | **20.11 – 24.67 tok/s** |
| Draft acceptance | 0.889 |
| Prefill, chunk 4096 | 254 – 261 tok/s |
| Load time, resident size | 37.5 s, ~107 GiB |

Runtime identity for every figure above, per `AGENTS.md` rule 8: 1 × DGX Spark (GB10), 128 GB unified
LPDDR5X, ATS addressing mode; **native ExLlamaV3** (not vLLM, not `vllm-exl3`) from
`vcruz305/exllamav3` branch `feat/gb10-ats-load` at `954a8ca6e59d`; CUDA 13.0,
`TORCH_CUDA_ARCH_LIST=12.1a`; `DeepseekV41ForCausalLM`; model revision
`5dc954019183ab3d994b60433256001a3f1780e7`; TP1, one local CUDA device; `CTX=6144`;
`max_batch_size=1`, single sequence; DSpark/MTP block drafting at `EXL3_DSPARK_CONF=0.7`, block size
5; per-expert mixed K (K1–K6).

On a repeated prompt the same configuration reaches about 33 tok/s warm, and context is nearly free
(131,072 measures the same warm decode as 6,144). Those are best-case prefix-cache figures and are
**not** comparable to the fresh-prompt numbers above. Full tables, methodology and the measured
negative results are in [`one-spark-tp1/BENCHMARKS.md`](one-spark-tp1/BENCHMARKS.md).

**TP2 and TP4 have no measured serving numbers.** Both are in qualification; see
[Which path do I need](#which-path-do-i-need).

## Contents

- [Measured performance](#measured-performance)
- [Quick start (one Spark)](#quick-start-one-spark)
- [Which path do I need](#which-path-do-i-need)
- [TP2 and TP4 qualification](#tp2-and-tp4-qualification): the seven-step workflow
- [TP4 release boundary](#tp4-release-boundary)
- [TP2 active qualification](#tp2-active-qualification)
- [Architecture target](#architecture-target): expert geometry per layout
- [Locked runtime](#locked-runtime): the pin and what it contains
- [Documentation map](#documentation-map): every file in `docs/` and `overlays/`
- [Repository layout](#repository-layout)
- [License](#license)

## Quick start (one Spark)

This is the only path with measured serving numbers. It runs **native ExLlamaV3**, so it does not use
`runtime.lock.json` and does not advance it. [`one-spark-tp1/README.md`](one-spark-tp1/README.md) is
the authoritative version of these steps; if the two ever disagree, that file wins.

**1. Confirm the GPU is in ATS addressing mode.** Zero-copy aliasing depends on it.

```bash
nvidia-smi -q | grep -i "addressing mode"
    Addressing Mode                   : ATS
```

**2. Build ExLlamaV3 from the fork branch.** The aarch64 patch must run before `pip install`, or the
build fails on x86 intrinsics.

```bash
git clone https://github.com/vcruz305/exllamav3.git
cd exllamav3
git checkout 954a8ca6e59d

python3 -m venv ~/venvs/exl3_v41
source ~/venvs/exl3_v41/bin/activate

curl -fsSL -o util/patch_exllamav3_aarch64.py \
  https://raw.githubusercontent.com/vcruz305/vllm-exl3/28c3585620df228de07e6f5115fbdc82877ba888/tools/patch_exllamav3_aarch64.py
python3 util/patch_exllamav3_aarch64.py exllamav3/exllamav3_ext

CUDA_HOME=/usr/local/cuda-13.0 TORCH_CUDA_ARCH_LIST=12.1a MAX_JOBS=8 \
  pip install --no-build-isolation --no-deps .
```

**3. Add the EXL3 attention/MTP overlay.** Every measured number depends on it. Without it you are
running the FP8 attention path.

```bash
hf download vcruz305/DSV4.1-Flash-SAGE-EXL3-1.59bpw \
  --revision 5dc954019183ab3d994b60433256001a3f1780e7 \
  --include 'exllamav3/*' --local-dir /models/overlay

# the parts must sit BESIDE model-*.safetensors, not in a subdirectory
cp /models/overlay/exllamav3/requant_*.safetensors \
   /models/DSV4.1-Flash-SAGE-EXL3-1.59bpw/
```

**4. Use the placement split that fits.** 107 GiB of main model plus ~14 GiB of drafter does not fit
in 128 GB, so everything except the drafter is copied into CUDA and the drafter stays aliased.

```bash
export EXL3_ATS_MMAP=1
export EXL3_ATS_COPY='^(?!mtp\.)'     # copy every tensor NOT starting with "mtp."
export EXL3_DSPARK_CONF=0.7           # measured optimum
export CHUNK=2048                     # 4096 does not fit once the model is resident
export CTX=6144                       # raise freely; must be a multiple of 256
```

**5. Launch.** `scripts/run_tp1.sh` in that folder is the exact launcher, including a pre-flight
`MemAvailable` check. Load takes about 40 s and drives `MemAvailable` to roughly 5 GiB, which is
expected. Do not run a second model process alongside it.

For **two or four Sparks**, skip this section and start at
[TP2 and TP4 qualification](#tp2-and-tp4-qualification).

## Which path do I need

| Topology | Artifact | Status |
|---|---|---|
| **TP4 / 4× Spark** | `vcruz305/DSV4.1-Flash-EXL3-4.75bpw` | **Per-expert mixed K3–K8 is supported by the pinned loader.** Resident Engram is a known GB10 UMA capacity failure. Full disk-backed Engram load/serve qualification is still required before calling TP4 deployable. |
| **TP2 / 2× Spark** | `vcruz305/DSV4.1-Flash-SAGE-EXL3-3.30bpw` | Mixed-K format is supported; **two-Spark capacity qualification is now the active test target**. EP2 is the baseline; pure MoE TP2 is an explicit A/B. |
| **TP1 / 1× Spark** | `vcruz305/DSV4.1-Flash-SAGE-EXL3-1.59bpw` | **Serving measured on one Spark** via *native ExLlamaV3* (not vLLM). See [Measured performance](#measured-performance) and [`one-spark-tp1/`](one-spark-tp1/). |

The recipe deliberately separates **loader-format compatibility** from **hardware deployment qualification**.

**TP1 is a different engine.** TP2 and TP4 run the pinned vLLM image with `vllm-exl3`. TP1 runs native ExLlamaV3 from a fork branch, does not use `runtime.lock.json`'s vLLM pin, and does not advance it.

## TP2 and TP4 qualification

The validation-first workflow. For a single Spark use [Quick start](#quick-start-one-spark) instead.

### 1. Host and remote-pack checks

```bash
bash scripts/doctor.sh 4
python3 scripts/probe_remote_pack.py --tp 4
```

The remote probe uses HTTP range reads for safetensors headers only. It validates shard/index membership, offsets, dtype/shape byte counts and physical K geometry without downloading the full checkpoint.

### 2. Build the locked runtime

```bash
bash scripts/build_runtime.sh
```

For Spark capacity qualification, also build the explicit disk-Engram derivative:

```bash
bash scripts/build_disk_engram_runtime.sh
```

This produces `deepseek-v41-exl3:disk-engram`. The baseline image remains unchanged. The same derivative is used by both TP4 and TP2 disk-Engram qualification profiles.

### 3. Materialize and validate the exact model revision

```bash
ALLOW_SOURCE_ARTIFACT_DOWNLOAD=1 \
  bash scripts/materialize_model.sh 4 /large/models/DSV4.1-Flash-EXL3-TP4

python3 scripts/validate_pack.py \
  /large/models/DSV4.1-Flash-EXL3-TP4 \
  --topology tp4 \
  --reserve-gib 32
```

Require:

```text
DEPLOYABLE_CURRENT_LOADER=YES
```

That means the checkpoint is structurally compatible with the pinned loader. It does **not** mean the four-Spark serving path has passed.

Mixed K inside one routed layer is now recorded and accepted; malformed shards, pointer stubs, invalid offsets, unsupported K and byte-count mismatches still fail closed.

### 4. Start every Spark in disk-Engram mode

Use the same local model mount/path on all nodes. Example TP4 head invocation:

```bash
IMAGE=deepseek-v41-exl3:disk-engram \
MODEL_DIR=/large/models \
MODEL=/models/DSV4.1-Flash-EXL3-TP4 \
VLLM_ENGRAM_MODEL_DIR=/models/DSV4.1-Flash-EXL3-TP4 \
HEAD_IP=10.0.0.10 NODE_IP=10.0.0.10 \
  bash scripts/start_disk_engram_cluster.sh head
```

Run the same wrapper with `worker` on the other nodes and their own `NODE_IP` values. TP2 uses `DISK_ENGRAM_PROFILE=profiles/tp2-disk-engram.env` and the materialized TP2 path.

`ENABLE_RDMA=auto` remains the default. Existing containers are never replaced unless `REPLACE_CONTAINER=1` is explicit.

### 5. Distributed preflight

```bash
bash scripts/preflight.sh 4
bash scripts/cluster_collective.sh 4
```

The disk min-fit launcher additionally probes every Ray GPU node for:

- disk-Engram environment;
- local checkpoint/index availability;
- disk-Engram overlay and weight-loader skip;
- mixed-K-capable `vllm-exl3`;
- non-network backing storage.

### 6. Arm the host UMA guards and run the first TP4 load

Resident Engram is already recorded as:

```text
RESIDENT_ENGRAM_TP4=CAPACITY_FAIL
```

Do not reproduce it as the normal first gate. Arm one exact-container watchdog on each Spark first:

```bash
OOM_GUARD_HOSTS="spark-a spark-b spark-c spark-d" \
OOM_GUARD_CONTAINER_NAME=dsv41-exl3 \
  bash scripts/watch_oom_guard.sh start
```

Verify them explicitly if desired:

```bash
OOM_GUARD_HOSTS="spark-a spark-b spark-c spark-d" \
OOM_GUARD_CONTAINER_NAME=dsv41-exl3 \
  bash scripts/check_oom_guards.sh 4
```

Then use the disk path:

```bash
bash scripts/tp4_disk_engram_min_fit.sh --check

OOM_GUARD_HOSTS="spark-a spark-b spark-c spark-d" \
OOM_GUARD_CONTAINER_NAME=dsv41-exl3 \
  bash scripts/tp4_disk_engram_min_fit.sh
```

The disk min-fit is locked to **8K / seq1 / text-only / eager / DSpark-off / native-off** and refuses a real load unless all four guards are alive, unless the explicit debug bypass is set.

The resident launcher remains available only for an intentional regression reproduction:

```bash
ALLOW_RESIDENT_ENGRAM_RETEST=1 bash scripts/tp4_min_fit.sh
```

### 7. Deterministic serving gate

Once the API is live:

```bash
bash scripts/smoke_test.sh
```

The smoke test requires `/v1/models` plus exact deterministic response content:

```text
EXL3 Spark OK
```

## TP4 release boundary

Do not mark TP4 deployment-ready until all of these are captured on real 4× Spark hardware:

1. physical pack validation;
2. runtime identity and NCCL collective pass;
3. disk-Engram all-node preflight pass;
4. all four host UMA guards verified;
5. full model load without the UMA cliff;
6. `/v1/models` ready;
7. deterministic smoke pass;
8. per-node memory receipt proving full Engram stays non-resident.

Only after that should larger context, DSpark and CUDA graphs be qualified independently.

## TP2 active qualification

TP2 no longer needs a layer-uniform repack merely to represent tensor-level mixed K. The active goal is now **capacity + topology qualification** on two Sparks.

Start with EP2:

```bash
DISK_ENGRAM_PROFILE="$PWD/profiles/tp2-disk-engram.env" \
MODEL=/models/DSV4.1-Flash-SAGE-EXL3-TP2 \
VLLM_ENGRAM_MODEL_DIR=/models/DSV4.1-Flash-SAGE-EXL3-TP2 \
  bash scripts/start_disk_engram_cluster.sh head

MOE_PARALLEL_MODE=ep bash scripts/tp2_disk_engram_min_fit.sh --check
MOE_PARALLEL_MODE=ep bash scripts/tp2_disk_engram_min_fit.sh
```

Only after EP2 reaches `/v1/models` and passes deterministic smoke should the same model/runtime be A/B tested with:

```bash
MOE_PARALLEL_MODE=tp bash scripts/tp2_disk_engram_min_fit.sh
```

Capture actual runtime evidence with:

```bash
bash scripts/kernel_dispatch_receipt.sh > kernel-dispatch.txt
```

V4.1 context estimates must use measured backend allocation for capacity claims. The logical global-cache floor is 890 bytes/token, but that is not a substitute for an actual vLLM cache allocation receipt. See `docs/TP2.md` and `scripts/v41_context_receipt.py`.

128K remains unverified.

## Architecture target

DeepSeek-V4.1 has 384 routed experts, hidden size 5120, expert intermediate size 2304 and top-k 6.

| Layout | Local experts | Expert matrix | Status |
|---|---:|---:|---|
| TP4 + EP4 | 96 | 5120 × 2304 | TP4 correctness baseline |
| TP2 + EP2 | 192 | 5120 × 2304 | TP2 correctness baseline |
| **TP1 / EP1 (1× Spark, one device)** | **384** | **5120 × 2304** | **measured serving**; no sharding, every expert local. Runs *native ExLlamaV3*, not vLLM, see [`one-spark-tp1/`](one-spark-tp1/) |
| pure MoE TP2 / EP1 | 384 | 5120 × 1152 | experimental A/B; 1152 is exactly 128-aligned |
| pure MoE TP4 / EP1 | 384 | 5120 × 576 | guarded; 576 is not 128-aligned and 576→640 padding is not implemented here |

There is **no 128-total-expert ExLlamaV3 ceiling**. The historical `>128` fallback concerns rows assigned to one expert in a batch, not experts owned by a rank.

## Locked runtime

`runtime.lock.json` is the single source of truth for the vLLM image, plugin revision, ExLlamaV3 revision, CUDA target, model revisions and first-boot policy.

Current `vllm-exl3` pin:

```text
814d4fe38082cddd838b45418c7d13a95395a36a
```

That pin includes:

- GB10 build compatibility from PR #9 by @fattchris;
- per-MoE TP/EP geometry resolution;
- **per-expert/per-projection mixed-K support from PR #10 by @Blackwellboy**;
- exact K3–K8 trellis shapes for `w1`, `w2` and `w3`;
- physical-trellis K selection for uniform fused layers, even when config/base K differs;
- fused execution for uniform-K layers;
- correctness-first `LinearEXL3` loop for heterogeneous layers;
- a safety guard that disables low-memory prescan when expert placement is not linear;
- TP-aware mixed-K header prescan so pure MoE TP2 allocates the correct 1152-wide local trellis geometry;
- V4.1-specific cache/topology planning that keeps logical 890 B/token separate from measured backend allocation.

Heterogeneous mixed-K execution is **not CUDA-graph-qualified yet**. First boot stays eager.

## Documentation map

Every file in `docs/`:

| Document | Covers |
|---|---|
| [`TP4.md`](docs/TP4.md) | TP4 + EP4 across four Sparks: the preferred qualification topology |
| [`TP2.md`](docs/TP2.md) | TP2 on two Sparks: the aggressive target, and the EP2-vs-TP2 A/B |
| [`TP2_METADATA_ATTESTATION.md`](docs/TP2_METADATA_ATTESTATION.md) | Why the canonical TP2 snapshot predates the explicit mixed-format metadata contract |
| [`DISK_ENGRAM.md`](docs/DISK_ENGRAM.md) | Disk-backed Engram: the current TP4 capacity path, isolated from the baseline image |
| [`VALIDATION.md`](docs/VALIDATION.md) | The fail-closed validation gates and what each one actually proves |
| [`COMPATIBILITY.md`](docs/COMPATIBILITY.md) | The DeepSeek-V4.1 EXL3 compatibility matrix |
| [`TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) | Failure modes and their diagnostics |
| [`SM120_UVA.md`](docs/SM120_UVA.md) | Experimental one-GPU `sm_120` path with large host RAM. **Not** the DGX Spark recipe |
| [`TONOKEN3_VALIDATION.md`](docs/TONOKEN3_VALIDATION.md) | Lna-Lab / TonoKen3 interoperability validation protocol |
| [`SAGE330_OFFLOAD_FINDINGS.md`](docs/SAGE330_OFFLOAD_FINDINGS.md) | Historical two-Spark results, reusable mechanisms and current integration gaps |
| [`SGLANG_V41_OPTIMIZATION_NOTES.md`](docs/SGLANG_V41_OPTIMIZATION_NOTES.md) | Independently implemented lessons from the SGLang reference article |
| [`HF_MODEL_CARD_CORRECTION.md`](docs/HF_MODEL_CARD_CORRECTION.md) | Replacement runtime/compatibility text for the public 4.75bpw model card |

Each overlay in `overlays/` documents itself:

| Overlay | Covers |
|---|---|
| [`disk-engram/`](overlays/disk-engram/README.md) | The disk-backed Engram overlay |
| [`dspark-in-checkpoint/`](overlays/dspark-in-checkpoint/README.md) | In-checkpoint DSpark draft wiring |
| [`dspark-draft-setup/`](overlays/dspark-draft-setup/README.md) | Config additions the 4.75bpw pack does not ship, needed for spec-decode, plus the current draft-class blocker |
| [`gb10-h2d-prefetch/`](overlays/gb10-h2d-prefetch/README.md) | GB10 host-to-device prefetch |
| [`sm120-sparse-fix/`](overlays/sm120-sparse-fix/README.md) | The `sm_120` sparse fix |

The single-Spark recipe keeps its own documentation under [`one-spark-tp1/`](one-spark-tp1/).

## Repository layout

| Path | Contents |
|---|---|
| [`one-spark-tp1/`](one-spark-tp1/) | **Single-Spark TP1 on native ExLlamaV3.** Self-contained; does not use the vLLM pin |
| `runtime.lock.json` | Immutable runtime/model contract |
| `Dockerfile.spark` | Baseline locked runtime |
| `Dockerfile.disk-engram` | Explicit disk-Engram derivative |
| `scripts/` | Validation, launch, preflight and receipt tooling (see below) |
| `profiles/` | Per-topology environment profiles, including `tp2-disk-engram.env` |
| `overlays/`, `configs/` | Runtime overlays and configuration |
| `attestations/`, `tests/` | Recorded attestations and the repository's own tests |
| `AGENTS.md` | Non-negotiable rules for changes to this repository |
| `THIRD_PARTY_NOTICES.md` | Attribution and upstream licenses |

Key scripts:

- `scripts/validate_pack.py` — physical checkpoint validator
- `scripts/check_disk_engram_cluster.py` — all-node disk-Engram preflight
- `scripts/check_oom_guards.sh` — exact-host watchdog verification
- `scripts/tp4_disk_engram_min_fit.sh` — guarded TP4 first load
- `scripts/tp2_disk_engram_min_fit.sh` — guarded TP2 EP2/pure-TP2 first load
- `scripts/v41_context_receipt.py` — V4.1 cache/capacity receipt helper
- `scripts/kernel_dispatch_receipt.sh` — actual runtime/kernel evidence collector
- `scripts/oom_guard.sh` — exact-container UMA safety guard

## License

Recipe code authored here is **AGPL-3.0-only**. Model weights, vLLM, ExLlamaV3, CUDA components and container layers retain their own licenses.
