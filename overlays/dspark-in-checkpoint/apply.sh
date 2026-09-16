#!/bin/bash
# Install the DSpark in-checkpoint (mtp.*) draft overlay and keep the reader
# lock hashes in sync. Files land over the frame runtime (bind-mount or copy).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
VLLM="${VLLM_DST:-/usr/local/lib/python3.12/dist-packages/vllm}"
LOCK="${EXL3_READER_LOCK:-$VLLM/model_executor/model_loader/exl3_reader_lock.json}"

install_one() {
  local rel="$1"
  test -f "$HERE/vllm/$rel"
  install -m 0644 "$HERE/vllm/$rel" "$VLLM/$rel"
  echo "installed $rel"
}

install_one model_executor/model_loader/exl3_attested_reader.py
install_one model_executor/model_loader/weight_utils.py
install_one model_executor/models/utils.py
install_one models/deepseek_v4_1/nvidia/dspark.py

python3 - "$VLLM" "$LOCK" <<'PY'
import hashlib, json, sys
from pathlib import Path

vllm, lock_path = Path(sys.argv[1]), Path(sys.argv[2])
keys = [
    "vllm/model_executor/model_loader/exl3_attested_reader.py",
    "vllm/model_executor/model_loader/weight_utils.py",
    "vllm/model_executor/models/utils.py",
]
if not lock_path.is_file():
    print(f"lock not present at {lock_path}; skip hash update")
    raise SystemExit(0)
lock = json.loads(lock_path.read_text())
files = lock.get("files")
if not isinstance(files, dict):
    raise SystemExit("lock has no files map")
for key in keys:
    rel = key.removeprefix("vllm/")
    digest = hashlib.sha256((vllm / rel).read_bytes()).hexdigest()
    if key not in files:
        raise SystemExit(f"lock missing files[{key!r}]")
    old = files[key]
    files[key] = digest
    print(f"updated lock {key}: {old} -> {digest}")
lock_path.write_text(json.dumps(lock, indent=2) + "\n")
PY

echo "dspark-in-checkpoint applied"
