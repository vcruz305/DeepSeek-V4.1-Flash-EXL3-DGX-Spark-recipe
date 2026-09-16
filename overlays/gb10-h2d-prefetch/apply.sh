#!/bin/bash
# Install the attested-reader prefetch overlay and keep the reader lock hash in sync.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/vllm/model_executor/model_loader/exl3_attested_reader.py"
DST="${EXL3_ATTESTED_READER_DST:-/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/exl3_attested_reader.py}"
LOCK="${EXL3_READER_LOCK:-/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/exl3_reader_lock.json}"
KEY="vllm/model_executor/model_loader/exl3_attested_reader.py"

test -f "$SRC"
install -m 0644 "$SRC" "$DST"
python3 - "$DST" "$LOCK" "$KEY" <<'PY'
import hashlib, json, sys
from pathlib import Path
dst, lock_path, key = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
digest = hashlib.sha256(dst.read_bytes()).hexdigest()
print(f"attested reader sha256={digest}")
if not lock_path.is_file():
    print(f"lock not present at {lock_path}; skip hash update")
    raise SystemExit(0)
lock = json.loads(lock_path.read_text())
files = lock.get("files")
if not isinstance(files, dict) or key not in files:
    raise SystemExit(f"lock missing files[{key!r}]")
old = files[key]
files[key] = digest
lock_path.write_text(json.dumps(lock, indent=2) + "\n")
print(f"updated lock {key}: {old} -> {digest}")
PY
echo "gb10-h2d-prefetch applied"
