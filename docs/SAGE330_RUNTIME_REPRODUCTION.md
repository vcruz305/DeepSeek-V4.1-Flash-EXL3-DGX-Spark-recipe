# Reproducing the SAGE 3.30 runtime experiments

The companion **vllm-exl3 runtime contribution** supplies the actual loader,
grouped/mixed-K kernels, bounded asynchronous reads, Engram caching and later
native-service/CUDA-graph fixes. This document records how the two-Spark runs
were assembled and measured. It complements the earlier findings-only recipe
PR; it does not advance this repository's default pins or change TP2/TP4 scripts.

Companion source: [runtime source at e4b8d4f](https://github.com/joeynyc/vllm-exl3/tree/e4b8d4f3a76eee87e5aa5e72f61194b72ab94d53/experiments/dsv41_runtime).

## Compatibility boundary

The measured runtime was a locally derived dedicated V4.1 image, reporting
`vllm 0.1.dev20904+g179dd0fa9`, Torch `2.13.0+cu130`, CUDA 13.0,
vllm-exl3 `8f4517e80416466fa4a3ad2eb28685021d39e95f`, and ExLlamaV3
`be57335b087e4f001c5caae061544df3c06ba01e`. The checkpoint was
`vcruz305/DSV4.1-Flash-SAGE-EXL3-3.30bpw` at
`e831e9e4d6bfeafa6d630848296417b1393404a3`.

The retained parent system-cache image on rank 0 was
`sha256:6efa34f3a352b6d873e541558bfad63cdef1c0b727212c57e5ffc8378f5a51a7`;
the final graph-native image was
`sha256:fda0fd530a93a347d2ed6fe5e54195f1379975f770c7a30d1939456925018b20`.
These are local image IDs, not public registry references. A public base that
independently reproduces the complete historical environment remains follow-up
work. Do not substitute stock pip vLLM or claim the current recipe image is
qualified by these measurements.

## What to carry forward

| Mechanism | Source in companion `experiments/dsv41_runtime/` | Why it mattered |
|---|---|---|
| Bounded loader and native tensor stream | `runtime/vllm_disk.py`, `sage_plugin.py`, `native_stream.py` | Avoid materializing every routed expert; require complete native parameters, including image-marker parameters retained by the text-only wrapper |
| Grouped expert execution | `runtime/grouped_plan.py`, `grouped_cache.py` | Batch compatible physical K triplets while preserving routing and ownership |
| Two-record asynchronous reads | `runtime/async_store.py`, `prefetch_cache.py` | Overlap verified reads with compute without unbounded staging |
| Engram raw-row cache/prefetch | `runtime/engram_rows.py`, `cached_engram_table.py`, `history/engram-image-r1/` | Deduplicate rows, preserve original FP8 weights/scales and TP ownership |
| Mixed-K fused execution | `native/mixed_k/`, `runtime/heterogeneous_cache.py` | Execute heterogeneous physical K values in one packed path |
| Larger prefill chunk | Launch `--max-num-batched-tokens 512`; `history/decoder-tail-image-r4/` | Reduce repeated dispatch/read overhead; keep approximate decoder tail off |
| Native miss service | `native/service/`, `runtime/native_shared_cache.py` | Avoid the Python GIL during the entire miss transaction |
| C1 full decode graphs | `vllm_overlay/`, `runtime/graph_runtime.py`, `graph_lifetime.py`, `null_kv_page.py` | Use the active V2 runner, retain graph storage, fence owners and clear reserved KV page zero |
| Graph-safe Engram and dynamic routing | `native/engram/`, `native/dynamic/`, `runtime/graph_rows.py` | Service original rows and changed expert routes during replay |

The final snapshot contains the graph/native path. The `history/` directories
preserve the earlier measured overlays for review; do not load all versions at
once. Mainline mixed-K support and disk Engram were prerequisites, with original
contributor attribution retained in the companion notices.

## Construct rank packages without changing quantization

Use an intact local download of the exact checkpoint revision. The serving
builder reads tensor offsets directly, including gapped original safetensors,
and verifies original shard hashes. It copies the original MUL1 K2–K8 expert
bytes and native MXFP8/FP8 tensors; it does not requantize.

The `sources` argument is the pinned metadata directory used by `catalog_layout`:

- `model-api.json`: Hugging Face model metadata for the exact model/revision,
  including `id`, `sha`, and every sibling's `rfilename`, `size`, `lfs.sha256`;
- `model.safetensors.index.json`: the original index;
- `headers/<shard-name>.json`: `{ "file_size": ..., "header_length": ...,
  "header": ... }` for each original shard. `header_length` is the little-endian
  uint64 in the first eight bytes, and `header` is the following JSON header.

Metadata and original weights must describe the same untouched snapshot. Do not
replace the authoritative shard hashes with hashes of an unverified download.
The builder creates hardlinks for unchanged Engram shards, so build its output
on the same filesystem as the original checkpoint. Use an unused destination;
preserve adequate space for both rank expert banks and their native files.

From the companion plugin checkout, on Linux:

```bash
python3 experiments/dsv41_runtime/runtime/build_packages.py \
  "$ORIGINAL_CHECKPOINT" "$PINNED_METADATA" "$NEW_PACKAGE_ROOT"
python3 experiments/dsv41_runtime/runtime/verify_package.py \
  "$NEW_PACKAGE_ROOT/rank0" --rank 0 --receipt "$RANK0_RECEIPT"
python3 experiments/dsv41_runtime/runtime/verify_package.py \
  "$NEW_PACKAGE_ROOT/rank1" --rank 1 --receipt "$RANK1_RECEIPT"
```

Each rank package must contain 7,680 experts (40 layers × 192 whole experts),
92,160 expert tensors and 1,257 native tensors. Keep each package local to its
own Spark. Verify the copied package on that Spark before launch. The source
config is preserved as `config.original.json`; the serving config adds the
native FP8 delegation metadata and records the source MTP experts policy.
DSpark was not enabled in these runs.

## Build the historical overlay

With a compatible retained V4.1 parent image on each Spark:

```bash
python3 -B experiments/dsv41_runtime/run_cpu_tests.py
docker build --build-arg BASE_IMAGE="$RETAINED_V41_IMAGE" \
  -t sage330-runtime:review experiments/dsv41_runtime
```

The new build wrapper checks the recorded graph/runner/bootstrap preimages,
requires the base's `sm120_page.py`, compiles four local extensions for SM121,
and then applies the overlay. A preimage mismatch requires an explicit port;
changing the expected hashes is not a qualification procedure. The wrapper
has CPU validation only in this contribution; a fresh CUDA build and hardware
qualification remain required. Historical native source/build receipts are
identified in the companion `provenance.json`.

## Rank configuration and launch

Create one `runtime-config.json`, mounted read-only at the same path on both
nodes. Set the two hashes to SHA256 of each rank's verified `package.json`:

```json
{
  "version": 1,
  "model": "vcruz305/DSV4.1-Flash-SAGE-EXL3-3.30bpw",
  "revision": "e831e9e4d6bfeafa6d630848296417b1393404a3",
  "model_path": "/model",
  "cache_bytes": 85899345920,
  "packages": [
    {"package_sha256": "REPLACE_WITH_RANK0_PACKAGE_SHA256"},
    {"package_sha256": "REPLACE_WITH_RANK1_PACKAGE_SHA256"}
  ]
}
```

The recorded Docker resource envelope on each node was host networking,
`--gpus all --device /dev/infiniband --ulimit memlock=-1 --shm-size 1g`,
`--memory 104g --memory-swap 104g --cpus 16 --pids-limit 2048`.
Mount that node's rank package at `/model:ro`, the configuration at
`/runtime-config.json:ro`, and a new writable telemetry directory at `/telemetry`.

Require both nodes idle before starting (at least 112 GiB available each in the
historical admission check). Throughout startup and serving, supervise both
containers together: zero host swap, at least 8 GiB available host memory,
no OOM or failed rank. A cgroup ceiling does not replace the host memory check.
Stop the paired experiment if either rank fails its gate, retaining its logs.

Set these environment variables in both containers; select the actual RoCE
interface and the reachable rank-0 rendezvous address for your own machines:

```bash
NCCL_DEBUG=INFO
NCCL_SOCKET_IFNAME="$ROCE_INTERFACE"
GLOO_SOCKET_IFNAME="$ROCE_INTERFACE"
TORCH_NCCL_ASYNC_ERROR_HANDLING=1
SAGE_DISK_CONFIG=/runtime-config.json
SAGE_TELEMETRY_DIR=/telemetry
DSV41_ENGRAM_DISK=1
DSV41_ENGRAM_DISK_THREADS=4
DSV41_ENGRAM_DISK_CHUNK=16
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
SAGE_DECODER_TAIL=0
SAGE_FULL_GRAPH=1
VLLM_USE_BREAKABLE_CUDAGRAPH=0
```

The final graph-native launch inside each container used the following CLI
settings. Supply `RANK=0` or `RANK=1`; append `--headless` only on rank 1. The
environment variables above must be passed through Docker's `-e` options.

```bash
python3 -B -m vllm.entrypoints.cli.main serve /model \
  --quantization exl3 --dtype bfloat16 \
  --tokenizer-mode deepseek_v41 --tool-call-parser deepseek_v41 \
  --enable-auto-tool-choice --reasoning-parser deepseek_v41 \
  --tensor-parallel-size 2 --enable-expert-parallel --enable-ep-weight-filter \
  --disable-custom-all-reduce --no-enable-prefix-caching \
  --distributed-executor-backend mp --nnodes 2 --node-rank "$RANK" \
  --master-addr "$RANK0_ADDRESS" --master-port "$RENDEZVOUS_PORT" \
  --language-model-only --max-model-len 32768 --max-num-seqs 1 \
  --max-num-batched-tokens 512 --gpu-memory-utilization 0.85 \
  --kv-cache-memory-bytes 4294967296 --kv-cache-dtype fp8 --block-size 64 \
  --kernel-config '{"moe_backend":"triton"}' \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1]}' \
  --served-model-name sage330-offload --host 127.0.0.1 --port 8093 \
  --no-enable-log-requests
```

The routed EXL3 backend was the custom packed heterogeneous extension despite
the native runtime's `moe_backend` CLI value. For an eager comparison within
this candidate, set `SAGE_FULL_GRAPH=0` and replace `--compilation-config` and its
argument with `--enforce-eager`. That compares against the final native-service
candidate in eager mode; the older CUDA-arena baseline is a different historical
overlay. Do not label these two eager baselines interchangeably.

## Results and measurement procedure

The earlier mechanism sweep ran one fresh deployment per mode, the same request
order, TP2+EP2, original 3.30 bytes, 80 GiB expert cache/rank, 4 GiB KV/rank,
32,768 context capacity and C1. Engram remained on local disk; prefix reuse,
DSpark and decoder tail were disabled for the table below.

| Cumulative stage | Coding tokens/s | Writing tokens/s | 32,631-token prompt TTFT (s) |
|---|---:|---:|---:|
| Original eager, chunk 128 | 2.731 | 4.893 | 181.65 |
| Grouped K triplets | 2.657 | 5.019 | 172.47 |
| Two bounded async reads | 2.545 | 4.896 | 157.25 |
| Engram row cache/prefetch | 2.625 | 4.957 | 137.99 |
| Mixed-K fused | 2.654 | 5.365 | 139.19 |
| Chunk 512, tail off | 2.629 | 5.243 | 73.46 |

The context probe placed three archive keys at the beginning, middle and end
of repeated topic text. It calibrated the exact chat-template token count through
the server tokenizer to between `context - 256` and `context - 128`, requested
at most 64 output tokens, and checked all keys and actual returned usage.
Use a fresh deployment and disable prefix reuse for a comparable TTFT probe.

The separate sustained-decode experiment used eight quality requests as warmup,
then three alternating coding/writing pairs. Exact synthetic decode requests and
numeric receipts are included in the companion's `evidence/` directory.

| Runtime | Median coding tokens/s | Median writing tokens/s |
|---|---:|---:|
| CUDA arena, eager | 4.400203 | 5.680964 |
| System arena, eager | 3.967238 | 5.010387 |
| Native service + full decode graphs | 4.648543 | 6.117337 |

Record returned token IDs and arrival timestamps. Only compute decode rate when
all completion token IDs are observed, one token per event, with strictly
increasing finite timestamps: `(N - 1) / (last - first)`. Report TTFT separately.
The coding outputs contained 135 tokens; writing was capped at 512. The final
run had 2,513 full graph replays per rank, passed its quality gates and maintained
zero swap with at least 12.99/12.84 GiB available host memory.

The system-arena eager result was a regression. The approximate tail experiment
also changed semantics and is not part of the accepted configuration. Keep both
findings visible when choosing patches to port. These are historical results
from one deployment per mode, not repeated-deployment confirmation.

No result here qualifies 30 tokens/s, 600K context, DSpark, TP4, or concurrent
serving. The 32K prompt result was measured on the earlier eager path; the final
graph run used short prompts with 32K configured capacity. Porting to current
main requires a new build, component parity, real-model quality/ownership checks,
live KV accounting, graph capture/replay/retirement evidence, paired zero-swap
memory supervision and fresh timing records.
