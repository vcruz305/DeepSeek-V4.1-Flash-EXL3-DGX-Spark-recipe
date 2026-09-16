# DeepSeek-V4.1-Flash EXL3 on DGX Spark

Serving and qualification tooling for **DeepSeek-V4.1-Flash EXL3** on NVIDIA DGX Spark / GB10.

> vLLM owns the DeepSeek-V4.1 model graph. `vllm-exl3` owns EXL3 routed-expert storage/execution. ExLlamaV3 supplies EXL3 kernels; it is not the V4.1 graph owner.

## Current status

| Topology | Artifact | Status |
|---|---|---|
| **TP4 / 4× Spark** | `vcruz305/DSV4.1-Flash-EXL3-4.75bpw` | **Per-expert mixed K3–K8 is supported by the pinned loader.** Resident Engram is a known GB10 UMA capacity failure. Full disk-backed Engram load/serve qualification is still required before calling TP4 deployable. |
| **TP2 / 2× Spark** | `vcruz305/DSV4.1-Flash-SAGE-EXL3-3.30bpw` | Mixed-K format is supported; **two-Spark capacity qualification is now the active test target**. EP2 is the baseline; pure MoE TP2 is an explicit A/B. |
| **TP2 / 2× Spark reference** | `vcruz305/DSV4.1-Flash-SAGE-EXL3-1.59bpw` | **Externally derived runtime qualified at 131,072 configured context with exact recall at 126,067 prompt tokens.** This does not change the canonical runtime lock. See [`docs/SAGE159_TP2_QUALIFICATION.md`](docs/SAGE159_TP2_QUALIFICATION.md). |

The recipe deliberately separates **loader-format compatibility** from **hardware deployment qualification**.

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

## Architecture target

DeepSeek-V4.1 has 384 routed experts, hidden size 5120, expert intermediate size 2304 and top-k 6.

| Layout | Local experts | Expert matrix | Status |
|---|---:|---:|---|
| TP4 + EP4 | 96 | 5120 × 2304 | TP4 correctness baseline |
| TP2 + EP2 | 192 | 5120 × 2304 | TP2 correctness baseline |
| pure MoE TP2 / EP1 | 384 | 5120 × 1152 | experimental A/B; 1152 is exactly 128-aligned |
| pure MoE TP4 / EP1 | 384 | 5120 × 576 | guarded; 576 is not 128-aligned and 576→640 padding is not implemented here |

There is **no 128-total-expert ExLlamaV3 ceiling**. The historical `>128` fallback concerns rows assigned to one expert in a batch, not experts owned by a rank.

## Validation-first workflow

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

128K remains unverified for the canonical TP2/3.30 path. The separate SAGE
1.59 reference qualification passed exact recall at 126,067 server-counted
prompt tokens; see [`docs/SAGE159_TP2_QUALIFICATION.md`](docs/SAGE159_TP2_QUALIFICATION.md).

## Important files

- `runtime.lock.json` — immutable runtime/model contract
- `Dockerfile.spark` — baseline locked runtime
- `Dockerfile.disk-engram` — explicit disk-Engram derivative
- `scripts/validate_pack.py` — physical checkpoint validator
- `scripts/check_disk_engram_cluster.py` — all-node disk-Engram preflight
- `scripts/check_oom_guards.sh` — exact-host watchdog verification
- `scripts/tp4_disk_engram_min_fit.sh` — guarded TP4 first load
- `scripts/tp2_disk_engram_min_fit.sh` — guarded TP2 EP2/pure-TP2 first load
- `scripts/v41_context_receipt.py` — V4.1 cache/capacity receipt helper
- `scripts/kernel_dispatch_receipt.sh` — actual runtime/kernel evidence collector
- `scripts/oom_guard.sh` — exact-container UMA safety guard
- `docs/TP4.md` — TP4 qualification details
- `docs/DISK_ENGRAM.md` — disk-backed Engram design/qualification
- `docs/TP2.md` — TP2 qualification and EP2-vs-TP2 A/B
- [SAGE 3.30 offload findings](docs/SAGE330_OFFLOAD_FINDINGS.md) — historical two-Spark results, reusable mechanisms and current integration gaps
- [SAGE 1.59 TP2 qualification](docs/SAGE159_TP2_QUALIFICATION.md) — bounded two-Spark 131K reference receipt on the exact published revision
- `docs/SGLANG_V41_OPTIMIZATION_NOTES.md` — independently implemented lessons from the SGLang reference article
- `THIRD_PARTY_NOTICES.md` — attribution and upstream licenses

## License

Recipe code authored here is **AGPL-3.0-only**. Model weights, vLLM, ExLlamaV3, CUDA components and container layers retain their own licenses.
