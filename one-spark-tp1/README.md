# One Spark, TP1: DeepSeek-V4.1-Flash EXL3 on native ExLlamaV3

Single DGX Spark (GB10), no cluster, no vLLM.

> **This is a different stack from the rest of this repo.** TP2 and TP4 here run the pinned vLLM
> image with `vllm-exl3`. TP1 runs **native ExLlamaV3** from a fork branch and does not use
> `runtime.lock.json`'s vLLM pin, `Dockerfile.spark`, the overlays or the cluster scripts. Nothing
> in this folder advances or replaces the locked vLLM runtime.

## Why a separate path

A 1.59 bpw DeepSeek-V4.1-Flash pack is roughly 107 GiB resident. One Spark has 128 GB of unified
LPDDR5X shared between CPU and GPU. That fits, but only if the loader does not keep a second copy
of the weights, and only if the drafter is not made resident alongside the main model.

Native ExLlamaV3 on a GB10 can do this because the GPU runs in **ATS addressing mode**: it shares
the process page tables and can read host virtual addresses directly, so weight tensors can be
aliased straight out of a `mmap` of the safetensors files instead of being copied into CUDA
allocations.

Confirm the mode before anything else:

```bash
nvidia-smi -q | grep -i "addressing mode"
    Addressing Mode                   : ATS
```

## Runtime identity

| Component | Value |
|---|---|
| Hardware | 1 × NVIDIA DGX Spark (GB10), 128 GB unified LPDDR5X, ATS addressing mode |
| Engine | **native ExLlamaV3** (not vLLM, not `vllm-exl3`) |
| ExLlamaV3 source | `https://github.com/vcruz305/exllamav3.git` |
| Branch | `feat/gb10-ats-load` |
| Commit | `954a8ca6e59d` |
| CUDA | 13.0, `TORCH_CUDA_ARCH_LIST=12.1a` |
| Architecture class | `DeepseekV41ForCausalLM` (`exllamav3/architecture/deepseek_v41.py`) |
| Drafter class | `exllamav3/architecture/deepseek_v41_mtp.py` (DSpark / MTP) |
| Topology | TP1, one local CUDA device |

The branch is required. Upstream ExLlamaV3 registers `DeepseekV4ForCausalLM`; **`DeepseekV41ForCausalLM`
and the ATS zero-copy loader live on this branch.**

### Build

```bash
git clone https://github.com/vcruz305/exllamav3.git
cd exllamav3
git checkout 954a8ca6e59d

python3 -m venv ~/venvs/exl3_v41
source ~/venvs/exl3_v41/bin/activate

# aarch64 (Grace) needs the x86-only CPU paths disabled before building the extension
python3 util/patch_exllamav3_aarch64.py exllamav3/exllamav3_ext 2>/dev/null || true

CUDA_HOME=/usr/local/cuda-13.0 \
TORCH_CUDA_ARCH_LIST=12.1a \
MAX_JOBS=8 \
  pip install --no-build-isolation --no-deps .

python3 -c "import exllamav3, exllamav3_ext; print('ok')"
```

> If `util/patch_exllamav3_aarch64.py` is not present on the branch, use the copy in
> `vcruz305/vllm-exl3` at `tools/patch_exllamav3_aarch64.py`. It replaces `__builtin_ia32_pause` /
> `_mm_pause` with `std::this_thread::yield()` and stubs the AVX2/AVX-512 target functions so the
> extension compiles on aarch64.

## Pack preparation: 64-byte re-lay

Zero-copy aliasing can only alias a tensor whose bytes already sit where the kernels expect them.
EXL3 trellis kernels need their `int16` data on a 16-byte boundary. Safetensors writers pack
tensors back to back, so most shards of an existing EXL3 pack have tensors on odd offsets and the
loader has to copy them instead, which defeats the purpose.

Measured on this model: **without re-laying, only 48.6 GiB aliased and 67.4 GiB were copied.**
After re-laying at 64 bytes, all text-model tensors aliased and nothing was copied.

```bash
python util/align_safetensors.py \
  /models/DSV4.1-Flash-SAGE-EXL3-1.59bpw \
  /models/DSV4.1-Flash-SAGE-EXL3-1.59bpw-a64 \
  --align 64 --min-bytes 1048576 --skip .engram.embed. --jobs 4
```

- Shards already on the grid are symlinked, not copied. Keep the source directory.
- Gaps are filled with small `__align_pad__.<shard>.N` U8 tensors so each file stays a standard,
  contiguous safetensors buffer. The loader on this branch skips that prefix.
- Free disk equal to the rewritten shards is needed; ~20 minutes on NVMe for this model.
- Full details: `doc/gb10_ats_loading.md` on the ExLlamaV3 fork branch.

> Re-laid packs are **not** drop-in for other loaders. vLLM's `AutoWeightsLoader` rejects the pad
> tensors unless it is told to ignore that prefix.

## The configuration that actually fits

The main model and the drafter cannot both be resident: 107 GiB + ~14 GiB exceeds the box.
The measured-best configuration copies **everything except the drafter** into CUDA memory and
leaves the drafter aliased in page cache:

```bash
export EXL3_ATS_MMAP=1
export EXL3_ATS_COPY='^(?!mtp\.)'     # copy all tensors whose names do NOT start with "mtp."
export EXL3_DSPARK_CONF=0.7           # DSpark confidence gate
export CHUNK=2048                     # prefill chunk
export CTX=6144
```

`EXL3_ATS_COPY` takes a regex matched against tensor names; matching tensors are copied into CUDA
memory, the rest stay aliased. The negative lookahead above is the whole trick.

Load takes ~40 s and drives `MemAvailable` down to roughly 5 GiB, which is expected. Do not run a
second model process alongside it.

See `scripts/run_tp1.sh` for the exact launcher, including a pre-flight `MemAvailable` check.

## Measured results

**Provenance for every number below** (`AGENTS.md` rule 8):

| Field | Value |
|---|---|
| Runtime | native ExLlamaV3, fork `feat/gb10-ats-load` @ `954a8ca` |
| Model | local 64-byte re-laid build of `vcruz305/DSV4.1-Flash-SAGE-EXL3-1.59bpw` (local variant; that exact local build is not published) |
| Topology | TP1, single Spark, one CUDA device |
| Context | `CTX=6144`, prefill chunk as noted per row |
| Batch | `max_batch_size=1`, single sequence |
| Speculative policy | DSpark / MTP block drafting, confidence gate `EXL3_DSPARK_CONF=0.7`, block size 5 |
| EXL3 backend | ExLlamaV3 EXL3 kernels, per-expert mixed K (K1–K6 present in this pack) |
| MoE dispatch | per-expert path; the opt-in grouped CUDA-graph modes were measured **slower** and are off |

Warm and cold are reported separately and are never combined (`AGENTS.md` benchmark discipline).

### Decode, no drafter

| Loading mode | Decode tok/s | Notes |
|---|---:|---|
| Fully aliased (`EXL3_ATS_MMAP=1`, no copy) | 13.96 – 14.19 | lowest memory, weights stay reclaimable |
| Whole model copied into CUDA | **15.13 – 15.22** | load 37.5 s, ~107 GiB resident, +7.3% |

### Decode, DSpark drafter at confidence 0.7

| Loading mode | Fresh prompt | Repeat prompt | Acceptance |
|---|---:|---:|---:|
| Aliased + drafter | 11.46 median | 16 – 17 warm | — |
| **Main copied + drafter aliased** | **17.53 median** (mean 19.82) | **20.11 – 24.67** | 0.889 |
| Interactive chat session | 11.8 cold | 17.4 warm | 0.74 |

### Prefill

| Loading mode | Chunk | Prefill tok/s | Context |
|---|---:|---:|---|
| Warm | 4096 | 254 – 261 | 4k – 6k |
| Model in CUDA | 2048 | 154 – 229 | 2k – 6k |

Chunk 4096 does not fit once the model is resident in CUDA; use 2048 there.

### Measured dead ends

Recorded so they are not re-tried:

- **Grouped MoE CUDA-graph modes.** `EXL3_MOE_GROUP_GRAPH=1` (per quantization-key groups,
  11–22 graphs/layer) gave 9.69 tok/s and `=2` (per projection, 11–16 groups) gave 10.43, against a
  10.97 baseline in the same harness. Both are **off by default**. The exact per-slot mgemm loses to
  the int8 GEMV path.
- **Huge pages** (`EXL3_ATS_HUGEPAGE=1`): ~1–2%, inside run-to-run noise.
- **Forcing a minimum draft length**: hurts.
- **Draft early-exit**: neutral.
- **`EXL3_MOE_MIXED_BSZ1=1`**: ~5% warm decode but greedy output was **not reproducible run to run**.
  Do not use it.

## TP1 vs TP2 / TP4 in this repo

ExLlamaV3's tensor parallelism is **single-host only**. It spawns one
`multiprocessing.Process` per *local* CUDA index and moves payloads through
`multiprocessing.shared_memory`; `EXLLAMA_MASTER_ADDR` defaults to `127.0.0.1` and there is no
multi-host worker. **Native ExLlamaV3 TP does not span two Sparks.**

So:

- **TP1** (this folder) = native ExLlamaV3, one Spark.
- **TP2 / TP4** (rest of this repo) = vLLM + `vllm-exl3`, multi-node.

No native-ExLlamaV3 TP2 or TP4 throughput numbers are published here, because that engine cannot
produce them across nodes. Cross-node work on the fork branch got as far as a TCP tensor channel
(`exllamav3/model/net_transport.py`, 13 passing unit tests) for a future pipeline split; it is
**not** wired into the forward path and is not a TP implementation.

## Serving with TabbyAPI

See [`tabbyapi/README.md`](tabbyapi/README.md) and [`tabbyapi/config.yml`](tabbyapi/config.yml).

**Status: configuration guidance, not a qualified path.** TabbyAPI is the official API server for
ExLlamaV3 and exposes `draft_mode: mtp`, which is the drafting mode this model's DSpark drafter
needs. It has **not** been run end-to-end against this pack on a Spark as part of this recipe, and
no TabbyAPI throughput numbers are published here.

## License

Recipe content in this folder is **AGPL-3.0-only**, matching the rest of the repository. ExLlamaV3,
CUDA components and model weights retain their own licenses.
