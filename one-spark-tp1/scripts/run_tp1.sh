#!/usr/bin/env bash
# One Spark (GB10) TP1 launcher for DeepSeek-V4.1-Flash EXL3 on native ExLlamaV3.
#
# Measured-best configuration: the main model is copied into CUDA memory and the DSpark drafter is
# left aliased from page cache, because 107 GiB + ~14 GiB resident does not fit in 128 GB.
#
# Requires: ExLlamaV3 from vcruz305/exllamav3 @ feat/gb10-ats-load (954a8ca), built for aarch64.
# See ../README.md for build, pack re-lay and measured numbers.
set -euo pipefail

VENV="${VENV:-$HOME/venvs/exl3_v41}"
MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to the 64-byte re-laid pack directory}"
ENTRY="${ENTRY:?set ENTRY to your ExLlamaV3 driver script}"

# --- Zero-copy / placement -------------------------------------------------
# Alias weights out of the safetensors mapping instead of copying them.
export EXL3_ATS_MMAP="${EXL3_ATS_MMAP:-1}"
# Copy every tensor whose name does NOT start with "mtp." into CUDA memory; the drafter
# (mtp.*) stays aliased in page cache. This is the configuration that fits and is fastest.
export EXL3_ATS_COPY="${EXL3_ATS_COPY:-^(?!mtp\.)}"

# --- Speculative decoding (DSpark / MTP) -----------------------------------
export EXL3_DSPARK_CONF="${EXL3_DSPARK_CONF:-0.7}"
export DRAFT="${DRAFT:-1}"

# --- Shapes ----------------------------------------------------------------
export CHUNK="${CHUNK:-2048}"     # prefill chunk; 4096 does not fit with the model resident
export CTX="${CTX:-6144}"
export PREWARM="${PREWARM:-1}"

# --- Toolchain -------------------------------------------------------------
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$HOME/.cache/torch_extensions_v41}"

# --- Guards ----------------------------------------------------------------
# Never run two model processes on one Spark; the second one will take the box down.
if pgrep -f "[e]xllamav3.*${MODEL_DIR##*/}" >/dev/null 2>&1; then
  echo "a model process is already using this pack; refusing to start" >&2
  exit 3
fi

# The load needs headroom. It drives MemAvailable down to roughly 5 GiB by design.
MIN_AVAIL_KB="${MIN_AVAIL_KB:-104857600}"   # 100 GiB
avail=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
if [ "$avail" -lt "$MIN_AVAIL_KB" ]; then
  echo "MemAvailable ${avail}kB < ${MIN_AVAIL_KB}kB; not starting" >&2
  exit 4
fi

# ATS addressing mode is what makes zero-copy aliasing possible at all.
if command -v nvidia-smi >/dev/null 2>&1; then
  mode=$(nvidia-smi -q 2>/dev/null | awk -F: '/Addressing Mode/{gsub(/ /,"",$2); print $2; exit}')
  if [ "${mode:-}" != "ATS" ]; then
    echo "warning: addressing mode is '${mode:-unknown}', expected ATS; aliasing will fall back to copies" >&2
  fi
fi

echo "model   : $MODEL_DIR"
echo "copy re : $EXL3_ATS_COPY"
echo "draft   : $DRAFT (conf $EXL3_DSPARK_CONF)"
echo "ctx/chunk: $CTX / $CHUNK"
echo "load takes ~40 s and will drive MemAvailable to ~5 GiB; this is expected"

exec "$VENV/bin/python" "$ENTRY"
