#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

TP="${1:-}"
if [[ "$TP" != "2" && "$TP" != "4" ]]; then
  echo "Usage: $0 2|4" >&2
  exit 2
fi

MODEL="$(resolve_model_for_tp "$TP")"
MODEL_REVISION_RESOLVED="$(resolve_model_revision_for_tp "$TP")"
export MODEL
require_env MODEL

if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  echo "ERROR: head container '$CONTAINER_NAME' is not running on this host." >&2
  exit 2
fi

GPU_COUNT="$(docker exec "$CONTAINER_NAME" python -c 'import ray; ray.init(address="auto", ignore_reinit_error=True, logging_level="ERROR"); print(int(ray.cluster_resources().get("GPU", 0)))' 2>/dev/null | tail -n1)"
if [[ ! "$GPU_COUNT" =~ ^[0-9]+$ ]] || (( GPU_COUNT < TP )); then
  echo "ERROR: Ray reports ${GPU_COUNT:-unknown} GPUs; TP$TP requires at least $TP." >&2
  docker exec "$CONTAINER_NAME" ray status || true
  exit 2
fi

LOCK_MAX_MODEL_LEN="$(python3 "$LOCK_TOOL" get first_boot.max_model_len)"
LOCK_GPU_MEMORY_UTILIZATION="$(python3 "$LOCK_TOOL" get first_boot.gpu_memory_utilization)"
LOCK_MAX_NUM_SEQS="$(python3 "$LOCK_TOOL" get first_boot.max_num_seqs)"
LOCK_MAX_NUM_BATCHED_TOKENS="$(python3 "$LOCK_TOOL" get first_boot.max_num_batched_tokens)"
LOCK_TEXT_ONLY="$(python3 "$LOCK_TOOL" get first_boot.text_only)"
LOCK_DSPARK="$(python3 "$LOCK_TOOL" get first_boot.dspark)"
LOCK_EAGER="$(python3 "$LOCK_TOOL" get first_boot.eager)"
LOCK_NATIVE_MOE="$(python3 "$LOCK_TOOL" get first_boot.native_moe)"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-$LOCK_MAX_MODEL_LEN}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-$LOCK_GPU_MEMORY_UTILIZATION}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-$LOCK_MAX_NUM_SEQS}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-$LOCK_MAX_NUM_BATCHED_TOKENS}"
TEXT_ONLY="${TEXT_ONLY:-$LOCK_TEXT_ONLY}"
DSPARK="${DSPARK:-$LOCK_DSPARK}"
EAGER="${EAGER:-$LOCK_EAGER}"
DRY_RUN="${DRY_RUN:-0}"
NATIVE_MOE="${NATIVE_MOE:-$LOCK_NATIVE_MOE}"
DISK_ENGRAM="${VLLM_ENGRAM_DISK_BACKED:-0}"
HF_OVERRIDES_JSON="${HF_OVERRIDES_JSON:-}"
MOE_PARALLEL_MODE="${MOE_PARALLEL_MODE:-ep}"
MOE_PARALLEL_MODE="$(printf '%s' "$MOE_PARALLEL_MODE" | tr '[:upper:]' '[:lower:]')"
if [[ "$MOE_PARALLEL_MODE" != "ep" && "$MOE_PARALLEL_MODE" != "tp" ]]; then
  echo "ERROR: MOE_PARALLEL_MODE must be ep or tp (got '$MOE_PARALLEL_MODE')." >&2
  exit 2
fi
if [[ "$MOE_PARALLEL_MODE" == "tp" && "$TP" == "4" ]] && ! is_true "${ALLOW_EXPERIMENTAL_TP4_MOE_TP:-0}"; then
  cat >&2 <<'EOF'
ERROR: pure MoE TP4 is not a default-qualified recipe path.
Its 2304/4=576 local intermediate width is not 128-aligned; SGLang padded
576->640 on GB300. This EXL3 recipe does not yet implement/qualify that padding.
Use TP4+EP4, or set ALLOW_EXPERIMENTAL_TP4_MOE_TP=1 only for development.
EOF
  exit 2
fi

# --- Optional serving knobs (all default to the locked/previous behavior) ----
#
# Manual KV budget. vLLM reserves exactly this many bytes and skips memory
# profiling entirely, so --gpu-memory-utilization is NOT respected on that run
# (vLLM logs this on every worker). Treat the two knobs as mutually exclusive:
# choose a KV budget or a utilization fraction, not both.
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-}"
if [[ -n "$KV_CACHE_MEMORY_BYTES" && ! "$KV_CACHE_MEMORY_BYTES" =~ ^[0-9]+$ ]]; then
  echo "ERROR: KV_CACHE_MEMORY_BYTES must be an integer byte count (got '$KV_CACHE_MEMORY_BYTES')." >&2
  exit 2
fi

BLOCK_SIZE="${BLOCK_SIZE:-}"
if [[ -n "$BLOCK_SIZE" && ! "$BLOCK_SIZE" =~ ^[0-9]+$ ]]; then
  echo "ERROR: BLOCK_SIZE must be an integer token count (got '$BLOCK_SIZE')." >&2
  exit 2
fi

PREFIX_CACHING="${PREFIX_CACHING:-}"
case "$PREFIX_CACHING" in
  ""|0|1|true|false|TRUE|FALSE|yes|no|YES|NO|on|off|ON|OFF) ;;
  *)
    echo "ERROR: PREFIX_CACHING must be a boolean 0/1 value (got '$PREFIX_CACHING')." >&2
    exit 2
    ;;
esac

# DSpark draft width stays 5 unless a profile overrides it; SPECULATIVE_QUANTIZATION
# is empty by default, which keeps the previous no-quantization spec config byte-for-byte.
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-5}"
if [[ ! "$NUM_SPECULATIVE_TOKENS" =~ ^[0-9]+$ ]]; then
  echo "ERROR: NUM_SPECULATIVE_TOKENS must be an integer (got '$NUM_SPECULATIVE_TOKENS')." >&2
  exit 2
fi
SPECULATIVE_QUANTIZATION="${SPECULATIVE_QUANTIZATION:-}"

# vLLM --compilation-config as a JSON object (cudagraph_mode / capture sizes).
COMPILATION_CONFIG="${COMPILATION_CONFIG:-}"

# Correctness-first default for EP2/EP4 is ExLlamaV3's fused/fallback routed
# expert path. Pure MoE TP2 is an explicit A/B candidate: 2304/2=1152, already
# 128-aligned, so it does not need the TP4-style 576->640 padding experiment.
EXEC_ENV=(
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600
  -e VLLM_USE_RUST_FRONTEND=1
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=1
  -e VLLM_EXL3_MOE_PARALLEL_MODE="$MOE_PARALLEL_MODE"
)

if is_true "$DISK_ENGRAM"; then
  if [[ -z "${VLLM_ENGRAM_MODEL_DIR:-}" || "${VLLM_ENGRAM_MODEL_DIR}" != /* ]]; then
    echo "ERROR: disk-backed Engram requires absolute VLLM_ENGRAM_MODEL_DIR inside the container." >&2
    exit 2
  fi
  EXEC_ENV+=(
    -e VLLM_ENGRAM_DISK_BACKED=1
    -e VLLM_ENGRAM_MODEL_DIR="$VLLM_ENGRAM_MODEL_DIR"
    -e VLLM_EXL3_MODEL_DIR="$VLLM_ENGRAM_MODEL_DIR"
  )
fi

if is_true "$NATIVE_MOE"; then
  EXEC_ENV+=(
    -e VLLM_EXL3_V41_NATIVE_MOE=1
    -e VLLM_EXL3_MOE_KERNEL=native
  )
  BACKEND_LABEL="native-abi3-k2-k4-candidate"
else
  EXEC_ENV+=(
    -e VLLM_EXL3_V41_NATIVE_MOE=0
    -e VLLM_EXL3_MOE_KERNEL=exllamav3
  )
  BACKEND_LABEL="exllamav3-control"
fi

# The deployed 1M-context profile runs the native kernel with the ABI-3 V4.1
# native MoE path still OFF, which the NATIVE_MOE branch above cannot express.
# Allow an explicit kernel override; unset preserves the branch default.
if [[ -n "${EXL3_MOE_KERNEL:-}" ]]; then
  EXEC_ENV+=( -e VLLM_EXL3_MOE_KERNEL="$EXL3_MOE_KERNEL" )
  BACKEND_LABEL="$BACKEND_LABEL/kernel-$EXL3_MOE_KERNEL"
fi

# EXL3 execution tuning observed in the deployed profile. Every one is opt-in:
# unset means the pinned image defaults apply exactly as before.
for _exl3_key in \
  VLLM_EXL3_ALLOW_SHAPE_MISMATCH \
  VLLM_EXL3_TRELLIS_ARENA \
  VLLM_EXL3_PADDED_MAX_T \
  VLLM_EXL3_PADDED_MAX_K \
  VLLM_EXL3_NATIVE_MOE_MAX_ROWS \
  VLLM_EXL3_MADV_AFTER_H2D \
  VLLM_EXL3_PREFETCH \
  VLLM_EXL3_PAD_SO_CB \
  VLLM_EXL3_MOE_TP_ALIGN \
; do
  if [[ -n "${!_exl3_key:-}" ]]; then
    EXEC_ENV+=( -e "$_exl3_key=${!_exl3_key}" )
  fi
done
unset _exl3_key

ARGS=(
  vllm serve "$MODEL"
  --quantization exl3
  --tokenizer-mode deepseek_v41
  --tensor-parallel-size "$TP"
  --distributed-executor-backend ray
  --tool-call-parser deepseek_v41
  --enable-auto-tool-choice
  --reasoning-parser deepseek_v41
  --served-model-name "$SERVED_MODEL_NAME"
  --host "$HOST"
  --port "$PORT"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
)

# Manual KV budget replaces profiling-based KV sizing (see the note above).
if [[ -n "$KV_CACHE_MEMORY_BYTES" ]]; then
  ARGS+=( --kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES" )
fi

if [[ -n "$BLOCK_SIZE" ]]; then
  ARGS+=( --block-size "$BLOCK_SIZE" )
fi

if [[ -n "$PREFIX_CACHING" ]]; then
  if is_true "$PREFIX_CACHING"; then
    ARGS+=( --enable-prefix-caching )
  else
    ARGS+=( --no-enable-prefix-caching )
  fi
fi

if [[ "$MOE_PARALLEL_MODE" == "ep" ]]; then
  ARGS+=( --enable-expert-parallel --enable-ep-weight-filter )
  TOPOLOGY_LABEL="TP$TP + EP$TP"
else
  TOPOLOGY_LABEL="TP$TP + MoE-TP$TP (EP1)"
fi

if [[ -n "$MODEL_REVISION_RESOLVED" ]]; then
  ARGS+=( --revision "$MODEL_REVISION_RESOLVED" )
fi

# vLLM supports runtime Hugging Face config overrides. This is used only after
# the TP2 canonical snapshot passes its immutable metadata attestation; it does
# not modify config.json or any model shard on disk.
if [[ -n "$HF_OVERRIDES_JSON" && "$HF_OVERRIDES_JSON" != "{}" ]]; then
  if ! python3 - "$HF_OVERRIDES_JSON" <<'PY'
import json, sys
value = json.loads(sys.argv[1])
if not isinstance(value, dict):
    raise SystemExit(2)
PY
  then
    echo "ERROR: HF_OVERRIDES_JSON is not valid JSON object data." >&2
    exit 2
  fi
  ARGS+=( --hf-overrides "$HF_OVERRIDES_JSON" )
fi

if is_true "$TEXT_ONLY"; then
  ARGS+=( --language-model-only )
else
  ARGS+=( --mm-encoder-tp-mode data )
fi

if is_true "$EAGER"; then
  ARGS+=( --enforce-eager )
fi

if is_true "$DSPARK"; then
  SPEC_JSON="{\"method\":\"dspark\",\"num_speculative_tokens\":$NUM_SPECULATIVE_TOKENS,\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"block\",\"enable_adaptive_verification\":false"
  if [[ -n "$SPECULATIVE_QUANTIZATION" ]]; then
    SPEC_JSON+=",\"quantization\":\"$SPECULATIVE_QUANTIZATION\""
  fi
  SPEC_JSON+="}"
  ARGS+=( --speculative-config "$SPEC_JSON" )
fi

if [[ -n "$COMPILATION_CONFIG" && "$COMPILATION_CONFIG" != "{}" ]]; then
  if ! python3 - "$COMPILATION_CONFIG" <<'PY'
import json, sys
value = json.loads(sys.argv[1])
if not isinstance(value, dict):
    raise SystemExit(2)
PY
  then
    echo "ERROR: COMPILATION_CONFIG is not valid JSON object data." >&2
    exit 2
  fi
  ARGS+=( --compilation-config "$COMPILATION_CONFIG" )
fi

if [[ -n "${EXTRA_VLLM_ARGS:-}" ]]; then
  echo "WARNING: EXTRA_VLLM_ARGS is shell-split; use only trusted local values." >&2
  # shellcheck disable=SC2206
  EXTRA=( $EXTRA_VLLM_ARGS )
  ARGS+=( "${EXTRA[@]}" )
fi

cat <<EOF
=== DeepSeek V4.1 EXL3 launch ===
Topology:               $TOPOLOGY_LABEL
MoE parallel mode:      $MOE_PARALLEL_MODE
Ray GPUs:               $GPU_COUNT
Model:                  $MODEL
Model revision:         ${MODEL_REVISION_RESOLVED:-<local-or-unpinned-override>}
Served name:            $SERVED_MODEL_NAME
EXL3 backend variant:   $BACKEND_LABEL
Runtime HF override:    $( [[ -n "$HF_OVERRIDES_JSON" && "$HF_OVERRIDES_JSON" != "{}" ]] && echo set || echo none )
Native V4.1 MoE:        $NATIVE_MOE
Disk-backed Engram:     $DISK_ENGRAM
Engram model dir:       ${VLLM_ENGRAM_MODEL_DIR:-<resident-or-unset>}
DSpark:                 $DSPARK
Eager:                  $EAGER
Text only:              $TEXT_ONLY
Max model len:          $MAX_MODEL_LEN
Max num seqs:           $MAX_NUM_SEQS
Max batched tokens:     $MAX_NUM_BATCHED_TOKENS
GPU memory utilization: $GPU_MEMORY_UTILIZATION
KV cache memory bytes:  ${KV_CACHE_MEMORY_BYTES:-<profiled from gpu-memory-utilization>}
Block size:             ${BLOCK_SIZE:-<model default>}
Prefix caching:         ${PREFIX_CACHING:-<vllm default>}
Spec draft tokens:      $( is_true "$DSPARK" && echo "$NUM_SPECULATIVE_TOKENS" || echo "<dspark off>" )
Spec draft quantization:${SPECULATIVE_QUANTIZATION:-<none>}
Compilation config:     ${COMPILATION_CONFIG:-<vllm default>}
Dry run:                $DRY_RUN
EOF

echo
printf 'Command:'
printf ' %q' "${ARGS[@]}"
echo

if is_true "$DRY_RUN"; then
  echo "DRY_RUN=1: command verified; exiting before model load."
  exit 0
fi

exec docker exec -i "${EXEC_ENV[@]}" "$CONTAINER_NAME" "${ARGS[@]}"
