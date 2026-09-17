# One Spark, TP1: DeepSeek-V4.1-Flash EXL3 on native ExLlamaV3

Single DGX Spark (GB10), no cluster, no vLLM.

> **This is a different stack from the rest of this repo.** TP2 and TP4 here run the pinned vLLM
> image with `vllm-exl3`. TP1 runs **native ExLlamaV3** from a fork branch and does not use
> `runtime.lock.json`'s vLLM pin, `Dockerfile.spark`, the overlays or the cluster scripts. Nothing
> in this folder advances or replaces the locked vLLM runtime.

## Why a separate path

A 1.59 bpw DeepSeek-V4.1-Flash pack is roughly 107 GiB resident and the DSpark drafter adds about
14 GiB. One Spark has 128 GB of unified LPDDR5X shared between CPU and GPU, so the main model fits
in CUDA memory but the drafter cannot sit beside it.

Native ExLlamaV3 on a GB10 resolves that because the GPU runs in **ATS addressing mode**: it shares
the process page tables and can read host virtual addresses directly. The main model is copied into
CUDA and the drafter is aliased straight out of a `mmap` of the safetensors files, so both are
reachable without a second copy of either.

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

# aarch64 (Grace) needs the x86-only CPU paths disabled before building the extension.
# This script is NOT on the ExLlamaV3 branch; fetch it from vllm-exl3 first.
curl -fsSL -o util/patch_exllamav3_aarch64.py \
  https://raw.githubusercontent.com/vcruz305/vllm-exl3/28c3585620df228de07e6f5115fbdc82877ba888/tools/patch_exllamav3_aarch64.py
python3 util/patch_exllamav3_aarch64.py exllamav3/exllamav3_ext

CUDA_HOME=/usr/local/cuda-13.0 \
TORCH_CUDA_ARCH_LIST=12.1a \
MAX_JOBS=8 \
  pip install --no-build-isolation --no-deps .

python3 -c "import exllamav3, exllamav3_ext; print('ok')"
```

> The patch replaces `__builtin_ia32_pause` / `_mm_pause` with `std::this_thread::yield()` and stubs
> the AVX2 / AVX-512 target functions so the extension compiles on aarch64. Run it before
> `pip install`; skipping it fails the build on x86 intrinsics.

## The EXL3 attention / MTP overlay

**Every measured number in this folder was produced with this overlay in place.** The published pack
stores attention and the MTP drafter as FP8 rows plus e8m0 scales (`layers.N.attn.wkv.weight` /
`.scale`). The overlay adds EXL3 trellis versions of those same projections
(`layers.N.attn.wkv.trellis` / `.suh` / `.svh` / `.mul1`) and of the drafter. Without it you are
running the FP8 attention path and will not reproduce the tables below.

The overlay is published alongside the pack:

```bash
hf download vcruz305/DSV4.1-Flash-SAGE-EXL3-1.59bpw \
  --revision 5dc954019183ab3d994b60433256001a3f1780e7 \
  --include 'exllamav3/*' --local-dir /models/overlay

# the loader globs the pack directory root, non-recursively: the parts must sit
# BESIDE model-*.safetensors, not in a subdirectory
cp /models/overlay/exllamav3/requant_*.safetensors \
   /models/DSV4.1-Flash-SAGE-EXL3-1.59bpw/
```

12 files, 10.61 GiB total. Notes:

- The overlay is **additive**: its 7276 tensor names do not exist in the base shards, so nothing is
  shadowed and the order `glob` returns the files in does not matter. Modules take the trellis path
  when it is present and fall back to FP8 when it is not.
- The overlay parts carry their own `__align_pad__.requant_part<N>.*` filler tensors, which the
  loader skips.
- To rebuild rather than download, the fork branch carries
  `tests/deepseek_v41/requant_attn_exl3.py` and `tests/deepseek_v41/requant_mtp_exl3.py`.

## The configuration that actually fits

The main model and the drafter cannot both be resident in CUDA: 107 GiB + ~14 GiB exceeds the box.
Measured directly — with aliasing disabled entirely (`EXL3_ATS_MMAP=0`) the load reaches
`torch_alloc 114.03 GiB` and leaves 1.52 GiB of `MemAvailable`, and the memory guard kills it before
the first token. So the configuration copies **everything except the drafter** into CUDA memory and
leaves the drafter aliased:

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
| Model | `vcruz305/DSV4.1-Flash-SAGE-EXL3-1.59bpw` **plus the EXL3 attention / MTP overlay** (`exllamav3/` in that repo) |
| Model revision | `5dc954019183ab3d994b60433256001a3f1780e7` — the repo revision holding both the pack and the `exllamav3/` overlay |
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
| Main model in CUDA | **15.13 – 15.22** | load 37.5 s, ~107 GiB resident |

### Decode, DSpark drafter at confidence 0.7

| Loading mode | Fresh prompt | Repeat prompt | Acceptance |
|---|---:|---:|---:|
| **Main in CUDA + drafter aliased** | **17.53 median** (mean 19.82) | **20.11 – 24.67** | 0.889 |
| Interactive chat session | 11.8 cold | 17.4 warm | 0.74 |

### Prefill

| Loading mode | Chunk | Prefill tok/s | Context |
|---|---:|---:|---|
| Warm | 4096 | 254 – 261 | 4k – 6k |
| Model in CUDA | 2048 | 154 – 229 | 2k – 6k |

Chunk 4096 does not fit once the main model is resident in CUDA; use 2048.

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
