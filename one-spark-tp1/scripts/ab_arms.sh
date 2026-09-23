#!/usr/bin/env bash
# Run the MoE-dispatch arms in strict alternation, one process per arm.
#
# The dispatch knobs are read at load, so an arm is a process, not a call. Alternating rather
# than running each arm's repeats together is what keeps session drift (page-cache warmth above
# all) from ordering the results: BENCHMARKS.md's Engram prefetch table is the worked example.
#
# Arms:
#   A  the pinned tree (954a8ca), dense per-expert MoE path             -- the baseline
#   C  the integration tree, EXL3_MIXEDK_LEGACY=1: per-K-group fused dispatch. 75.7% of this
#      pack's routed experts sit in groups whose gate/up/down K agree, so those groups get a
#      compile-time K instance; 21.9% land on the runtime-K kernel and 2.4% stay per-expert.
#   D  the integration tree with a row floor instead: the unified kernel above the floor, the
#      legacy dispatch below it. bsz*top_k is 6 at decode and 36 at MTP verify, so a floor in
#      7..36 splits those two, and a floor above 36 sends both to legacy.
#   B  the integration tree at its default (unified kernel everywhere). Measured -15.7% on this
#      pack on GB10 in the fork's own 13f1c16, so it is off by default here; ARMS can add it.
#   C2 C plus EXL3_MIXEDK_DET_GROUPS=1: the group launches accumulate through the layer's slot
#      table and the ordered gather instead of atomics. C measures the path as upstream ships
#      it; C2 measures what making it reproducible costs.
#   E  C plus EXL3_MTP_HEAD_N: the drafter samples its block through a slice of the shared fp16
#      head instead of all 129280 rows. Needs DRAFT=1 to do anything, and is compared against C.
#
# Usage:
#   PIN_SRC=~/dev/vcruz305/exl3-pin INTEG_SRC=~/dev/vcruz305/exl3-integ \
#   MODEL_DIR=/models/DSV4.1-Flash-SAGE-EXL3-1.59bpw \
#     bash ab_arms.sh
set -euo pipefail

VENV="${VENV:-$HOME/venvs/exl3_v41}"
MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to the pack directory}"
PIN_SRC="${PIN_SRC:?set PIN_SRC to the 954a8ca checkout}"
INTEG_SRC="${INTEG_SRC:?set INTEG_SRC to the integration checkout}"
OUT="${OUT:-$PWD/ab-runs-$(date +%Y%m%d-%H%M%S)}"
ROUNDS="${ROUNDS:-2}"
MODE="${MODE:-repeat}"
REPS="${REPS:-6}"
ARMS="${ARMS:-A C}"

HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUT"

# Shared across every arm: the placement split that fits, and the drafter knobs. Only the MoE
# dispatch may differ between arms, or the comparison measures two things at once.
export MODEL_DIR
export EXL3_ATS_MMAP="${EXL3_ATS_MMAP:-1}"
export EXL3_ATS_COPY="${EXL3_ATS_COPY:-^(?!mtp\.)}"
export EXL3_DSPARK_CONF="${EXL3_DSPARK_CONF:-0.7}"
export CHUNK="${CHUNK:-2048}"
export CTX="${CTX:-6144}"
# DRAFT=0 is the arm to run first: plain greedy is self-deterministic here, so a dispatch
# change shows up without acceptance and draft-block length moving underneath it.
export DRAFT="${DRAFT:-0}"
export MODE REPS
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH"

arm_env() {
  # Per-arm environment on stdout, one VAR=VALUE per line.
  case "$1" in
    A) echo "EXL3_SRC=$PIN_SRC" ;;
    B) echo "EXL3_SRC=$INTEG_SRC"; echo "EXL3_GR_INT8=0" ;;
    C) echo "EXL3_SRC=$INTEG_SRC"; echo "EXL3_MIXEDK_LEGACY=1"; echo "EXL3_GR_INT8=0" ;;
    D) echo "EXL3_SRC=$INTEG_SRC"; echo "EXL3_MOE_MIXEDK_MIN_ROWS=${MIN_ROWS:-37}"; echo "EXL3_GR_INT8=0" ;;
    # C2 is C plus the deterministic slot accumulation, so it prices determinism on its own.
    C2) echo "EXL3_SRC=$INTEG_SRC"; echo "EXL3_MIXEDK_LEGACY=1"; echo "EXL3_GR_INT8=0"; echo "EXL3_MIXEDK_DET_GROUPS=1" ;;
    # E is C plus the pruned draft head, so compare it against C rather than A: it only has an
    # effect with the drafter loaded (DRAFT=1), and acceptance is the number to watch.
    E) echo "EXL3_SRC=$INTEG_SRC"; echo "EXL3_MIXEDK_LEGACY=1"; echo "EXL3_GR_INT8=0"; echo "EXL3_MTP_HEAD_N=${HEAD_N:-65536}" ;;
    # Levers on the pinned tree itself, each compared against A. AP cannot change output (the
    # prefetch is a page-cache hint); AH only changes which tokens are proposed. Both need
    # DRAFT=1 to be meaningful for AH, and AP matters most on repeated prompts.
    AP) echo "EXL3_SRC=$PIN_SRC"; echo "EXL3_ENGRAM_PREFETCH=0" ;;
    AH) echo "EXL3_SRC=${PINHEAD_SRC:?set PINHEAD_SRC}"; echo "EXL3_MTP_HEAD_N=${HEAD_N:-65536}" ;;
    AHP) echo "EXL3_SRC=${PINHEAD_SRC:?set PINHEAD_SRC}"; echo "EXL3_MTP_HEAD_N=${HEAD_N:-65536}"; echo "EXL3_ENGRAM_PREFETCH=0" ;;
    *) echo "unknown arm $1" >&2; exit 2 ;;
  esac
}

for round in $(seq 1 "$ROUNDS"); do
  for arm in $ARMS; do
    f="$OUT/${arm}-round${round}.jsonl"
    echo "== arm $arm round $round -> $f"
    # A model process must never overlap another one on this box.
    if pgrep -f "[a]b_dispatch.py" >/dev/null; then
      echo "another ab_dispatch.py is running; refusing to start" >&2
      exit 3
    fi
    avail=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
    if [ "$avail" -lt 104857600 ]; then
      echo "MemAvailable ${avail}kB below 100 GiB; the pack will not load" >&2
      exit 4
    fi
    env $(arm_env "$arm") ARM="$arm" "$VENV/bin/python" "$HERE/ab_dispatch.py" | tee "$f"
  done
done

echo
echo "compare with:"
first="${ARMS%% *}"
cmp_args=""
for arm in $ARMS; do
  [ "$arm" = "$first" ] && continue
  cmp_args="$cmp_args --arm $arm=$(ls "$OUT"/${arm}-round*.jsonl | paste -sd,)"
done
echo "  $VENV/bin/python $HERE/ab_compare.py --baseline $first=$(ls "$OUT"/${first}-round*.jsonl | paste -sd,)$cmp_args"
