#!/usr/bin/env python3
"""Fix for the 4.75bpw pack missing its non-expert backbone (issue #13).

The 32-shard pack contains only routed EXL3 experts, engram tables, MTP draft
layers, vision/aligner globals, and embed/head/norm. The 40 main layers'
attention / compressor / indexer / router / shared-experts / hc_* / norm
tensors are absent, so serving runs with those parameters at init and emits
deterministic word salad.

This script pulls the missing tensors from the base model
(`deepseek-ai/DeepSeek-V4.1-Flash`) via HTTP Range requests against the shard
headers (no full-shard downloads), writes them as a supplementary
`model-00033-of-00033.safetensors`, and extends the pack's
`model.safetensors.index.json`. The base's dense tensors are fp8 e4m3 with
[32,32] ue8m0 block scales — byte-compatible with the pack config's
`non_routed_quantization`, so no requantization is needed.

Usage (on a node with the pack and internet access):
    python3 fix_missing_backbone.py --pack /var/tmp/dsv41-orig

Idempotent: skips work when the index already contains non-expert layer keys.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
import time
import urllib.request

BASE = "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/main"
HEADER_BYTES = 5 << 20  # safetensors headers are far smaller than this


def fetch(url: str, start: int | None = None, end: int | None = None,
          timeout: int = 120, retries: int = 6) -> bytes:
    headers = {"User-Agent": "backbone-fix"}
    if start is not None:
        headers["Range"] = f"bytes={start}-{end}"
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"fetch failed {url}: {last}")


def base_index() -> dict:
    raw = fetch(f"{BASE}/model.safetensors.index.json", timeout=60)
    return json.loads(raw)["weight_map"]


def shard_header(shard: str) -> dict[str, tuple]:
    raw = fetch(f"{BASE}/{shard}", 0, HEADER_BYTES - 1)
    n = struct.unpack("<Q", raw[:8])[0]
    if 8 + n > len(raw):
        raise RuntimeError(f"header of {shard} larger than {HEADER_BYTES}; bump HEADER_BYTES")
    hdr = json.loads(raw[8:8 + n])
    out = {}
    for name, info in hdr.items():
        if name == "__metadata__":
            continue
        s, e = info["data_offsets"]
        out[name] = (info["dtype"], info["shape"], s + 8 + n, e + 8 + n)
    return out


def pull_set(base_wm: dict, pack_wm: dict, n_layers: int) -> set[str]:
    need = set()
    for k in base_wm:
        m = re.fullmatch(r"layers\.(\d+)\.(.+)", k)
        if not m or not 0 <= int(m.group(1)) < n_layers:
            continue
        suffix = m.group(2)
        if suffix.startswith(("ffn.experts.", "engram.")):
            continue
        if k not in pack_wm:
            need.add(k)
    return need


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True, help="pack directory (with index + shards)")
    ap.add_argument("--n-layers", type=int, default=40)
    ap.add_argument("--out-shard", default="model-00033-of-00033.safetensors")
    args = ap.parse_args()

    pack_idx = os.path.join(args.pack, "model.safetensors.index.json")
    pack_wm = json.load(open(pack_idx))["weight_map"]
    need = pull_set(base_index(), pack_wm, args.n_layers)
    if not need:
        print("pack already has the backbone; nothing to do")
        return 0
    print(f"missing backbone tensors: {len(need)}")

    bwm = base_index()
    by_shard: dict[str, list[str]] = {}
    for k in need:
        by_shard.setdefault(bwm[k], []).append(k)

    offsets: dict[str, tuple] = {}
    cur = 0
    for shard in sorted(by_shard):
        tbl = shard_header(shard)
        for k in sorted(by_shard[shard]):
            dt, shape, st, en = tbl[k]
            offsets[k] = (cur, cur + en - st, dt, shape, shard, st)
            cur += en - st
    total = cur
    print(f"total bytes: {total / 1e9:.2f} GB across {len(by_shard)} shards")

    body = os.path.join(args.pack, "backbone-body.bin")
    with open(body, "wb") as f:
        f.truncate(total)
    done = 0
    t0 = time.time()
    for k in sorted(offsets):
        o1, o2, dt, shape, shard, st = offsets[k]
        data = fetch(f"{BASE}/{shard}", st, st + (o2 - o1) - 1)
        assert len(data) == o2 - o1, f"short read {k}"
        with open(body, "r+b") as f:
            f.seek(o1)
            f.write(data)
        done += 1
        if done % 200 == 0:
            rate = done / (time.time() - t0)
            print(f"{done}/{len(offsets)} {rate:.1f}/s", flush=True)

    hdr = {k: {"dtype": dt, "shape": shape, "data_offsets": [o1, o2]}
           for k, (o1, o2, dt, shape, _, _) in offsets.items()}
    hjson = json.dumps(hdr).encode()
    hjson += b" " * ((8 + len(hjson) + 7) // 8 * 8 - 8 - len(hjson))
    out_path = os.path.join(args.pack, args.out_shard)
    with open(out_path, "wb") as out, open(body, "rb") as f:
        out.write(struct.pack("<Q", len(hjson)))
        out.write(hjson)
        while chunk := f.read(1 << 24):
            out.write(chunk)
    os.remove(body)

    idx = json.load(open(pack_idx))
    for k in offsets:
        idx["weight_map"][k] = args.out_shard
    json.dump(idx, open(pack_idx, "w"))
    print(f"wrote {out_path}; index now {len(idx['weight_map'])} keys")
    print("reload the model to pick up the backbone")
    return 0


if __name__ == "__main__":
    sys.exit(main())
