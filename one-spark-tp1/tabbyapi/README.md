# TabbyAPI on one Spark (TP1)

**Status: configuration guidance, not a qualified path.**

TabbyAPI is the official API server for ExLlamaV3 and exposes the drafting mode this model needs
(`draft_mode: mtp`). It has **not** been run end-to-end against this pack on a DGX Spark as part of
this recipe, so no TabbyAPI throughput numbers are published here and nothing in this folder should
be read as a pass (`AGENTS.md` rule 7).

The measured numbers in [`../README.md`](../README.md) come from driving ExLlamaV3 directly, not
through TabbyAPI.

## The one thing that will break this

TabbyAPI installs ExLlamaV3 itself and, per its own documentation, **"enforces the latest
Exllamav3 version for compatibility purposes"**. Any upgrade through its GPU-library extra will
overwrite a custom install.

This recipe needs a **fork branch**, not the released wheel:

```
https://github.com/vcruz305/exllamav3.git @ feat/gb10-ats-load (954a8ca6e59d)
```

Upstream ExLlamaV3 registers `DeepseekV4ForCausalLM`. `DeepseekV41ForCausalLM` and the GB10 ATS
zero-copy loader only exist on that branch. If TabbyAPI replaces it, the model stops loading.

So:

1. Build the fork into a venv first (see [`../README.md`](../README.md) → Build).
2. Install TabbyAPI into **that same venv**, without its GPU-library extra.
3. Re-install the fork with `pip install .` after any TabbyAPI update, and re-check:

```bash
python3 -c "from exllamav3.architecture.deepseek_v41 import *; print('v41 arch present')"
python3 -c "import exllamav3, exllamav3_ext; print(exllamav3.__file__)"
```

If `exllamav3.__file__` points somewhere other than your fork build, TabbyAPI has overwritten it.

## ATS environment variables

TabbyAPI has no knowledge of the zero-copy loader. These are read from the process environment by
the fork's loader and **must be exported before TabbyAPI starts**:

```bash
export EXL3_ATS_MMAP=1
export EXL3_ATS_COPY='^(?!mtp\.)'   # copy everything except the drafter into CUDA
export EXL3_DSPARK_CONF=0.7
```

Without `EXL3_ATS_COPY`, either everything is aliased (slower) or everything is copied (does not
fit: ~107 GiB main model plus ~14 GiB drafter against 128 GB of unified memory).

## Launcher

```bash
#!/usr/bin/env bash
set -euo pipefail
VENV="$HOME/venvs/exl3_v41"
export PATH="$VENV/bin:/usr/local/cuda-13.0/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-13.0

export EXL3_ATS_MMAP=1
export EXL3_ATS_COPY='^(?!mtp\.)'
export EXL3_DSPARK_CONF=0.7

avail=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
[ "$avail" -ge 104857600 ] || { echo "MemAvailable ${avail}kB < 100 GiB" >&2; exit 4; }

cd "$HOME/tabbyAPI"
exec "$VENV/bin/python" main.py --config /path/to/one-spark-tp1/tabbyapi/config.yml
```

Load takes ~40 s and drives `MemAvailable` to roughly 5 GiB. That is expected. Never start a second
model process on the same Spark.

## Config

See [`config.yml`](config.yml). Keys are taken from TabbyAPI's `config_sample.yml`; the values match
the measured-best native configuration.

Two settings carry real consequences:

- `chunk_size: 2048` — 4096 measured faster while weights were still aliased, but does not fit once
  the model is resident in CUDA memory.
- `tensor_parallel: false` — ExLlamaV3 tensor parallelism is single-host only (one
  `multiprocessing.Process` per *local* CUDA index, shared-memory payloads, `EXLLAMA_MASTER_ADDR`
  defaulting to `127.0.0.1`). It does not span two Sparks. TP2 and TP4 in this repository are the
  vLLM path.

## Known unknowns

Recorded rather than guessed:

- Whether `draft_mode: mtp` needs `draft_model_name` set when the MTP head lives inside the main
  pack as `mtp.*` tensors.
- Whether TabbyAPI's loader path preserves the `EXL3_ATS_COPY` placement split, or whether it
  forces its own device placement and defeats the aliasing.
- Whether TabbyAPI tolerates the `__align_pad__.*` tensors in a re-laid pack. The fork's loader
  skips that prefix; TabbyAPI calls the same loader, so it should, but this is untested.
- Quantized KV (`cache_mode: "8,8"`) behaviour for this pack on GB10.

Each of these is a real test, not a formality. Until they are run, treat this folder as a starting
point rather than a supported configuration.
