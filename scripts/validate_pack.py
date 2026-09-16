#!/usr/bin/env python3
"""Fail-closed physical compatibility validator for DeepSeek-V4.1 EXL3 packs.

The validator reads safetensors headers only. It is intentionally independent of
vLLM model loading so malformed or incompatible ~400 GiB checkpoints fail before
cluster startup or weight allocation.

Generic/local validation checks loader compatibility and structural integrity.
Canonical Hugging Face snapshot validation can additionally require the exact
locked shard count via ``--strict-locked-snapshot``. This distinction matters
because a corrected/repacked local checkpoint may legitimately use a different
number of safetensors shards while remaining fully loader-compatible.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
from typing import Any

from runtime_lock import load_lock

MAX_HEADER_BYTES = 256 * 1024 * 1024
LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
ROUTED_RE = re.compile(r"(?:^|\.)(?:ffn\.)?experts(?:\.|$)")
DTYPE_BYTES: dict[str, int] = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E4M3FN": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}


def gib(value: int) -> float:
    return value / (1024**3)


def physical_trellis_k_capability(lock: dict[str, Any]) -> set[int]:
    """Return physical K widths executable by the baseline ExLlamaV3 backend.

    ``accepted_exl3_config_k`` describes values accepted in the checkpoint's
    top-level EXL3 config. Individual expert trellises can be narrower than that
    base declaration, so physical headers must be checked against the kernel
    capability instead. Older locks fall back to the config list.
    """
    capabilities = lock["capabilities"]
    values = capabilities.get(
        "exllamav3_moe_kernel_k", capabilities["accepted_exl3_config_k"]
    )
    return {int(k) for k in values}


def read_safetensors_header(
    path: Path,
) -> tuple[str, dict[str, Any] | None, str | None, int, str | None]:
    size = path.stat().st_size
    with path.open("rb") as handle:
        first = handle.read(256)
        if first.startswith(b"version https://git-lfs.github.com/spec/v1"):
            return "pointer", None, "Git LFS pointer stub; payload not materialized", 0, None
        if first.lstrip().lower().startswith((b"<html", b"<!doctype", b"<?xml")):
            return "html", None, "HTML/XML response saved instead of safetensors", 0, None
        if size < 16:
            return "truncated", None, f"file is only {size} bytes", 0, None
        handle.seek(0)
        raw = handle.read(8)
        header_len = struct.unpack("<Q", raw)[0]
        if header_len <= 1 or header_len > MAX_HEADER_BYTES:
            return "invalid", None, f"implausible safetensors header length {header_len}", 0, None
        payload_start = 8 + header_len
        if payload_start > size:
            return "truncated", None, f"header ends at {payload_start}, file size is {size}", 0, None
        header_raw = handle.read(header_len)
    try:
        header = json.loads(header_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return "invalid", None, f"header JSON decode failed: {exc}", 0, None
    if not isinstance(header, dict):
        return "invalid", None, "safetensors header is not a JSON object", 0, None
    return "ok", header, None, payload_start, hashlib.sha256(header_raw).hexdigest()


def trellis_k(meta: Any) -> int | None:
    if not isinstance(meta, dict):
        return None
    shape = meta.get("shape")
    if not isinstance(shape, list) or not shape:
        return None
    try:
        words = int(shape[-1])
    except (TypeError, ValueError, OverflowError):
        return None
    if words <= 0 or words % 16:
        return None
    return words // 16


def expected_tensor_bytes(meta: dict[str, Any]) -> int | None:
    dtype = str(meta.get("dtype", ""))
    itemsize = DTYPE_BYTES.get(dtype)
    shape = meta.get("shape")
    if itemsize is None or not isinstance(shape, list):
        return None
    count = 1
    try:
        for dim in shape:
            value = int(dim)
            if value < 0:
                return None
            count *= value
    except (TypeError, ValueError, OverflowError):
        return None
    return count * itemsize


def validate_pack(
    model_dir: Path,
    topology: str,
    reserve_gib: float,
    *,
    strict_locked_snapshot: bool = False,
) -> dict[str, Any]:
    lock = load_lock()
    root = model_dir.expanduser().resolve()
    tp_key = "tp4" if topology == "tp4" else "tp2"
    model_contract = lock["models"][tp_key]
    allowed_k = physical_trellis_k_capability(lock)
    layer_uniform_required = not bool(
        lock["capabilities"].get("tensor_level_mixed_k_within_layer", False)
    )

    result: dict[str, Any] = {
        "schema": "dsv41-exl3-pack-validation.v1",
        "model_dir": str(root),
        "topology": topology,
        "runtime_lock_schema": lock["schema"],
        "required_loader_contract": model_contract["required_loader_contract"],
        "reserve_gib": reserve_gib,
        "strict_locked_snapshot": bool(strict_locked_snapshot),
        "errors": [],
        "warnings": [],
    }
    errors: list[str] = result["errors"]
    warnings: list[str] = result["warnings"]

    if not root.is_dir():
        errors.append("model directory does not exist")
        result["deployable_with_current_pinned_loader"] = False
        return result

    config_path = root / "config.json"
    index_path = root / "model.safetensors.index.json"
    config: dict[str, Any] = {}
    index: dict[str, Any] = {}
    if not config_path.is_file():
        errors.append("config.json is missing")
    else:
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"config.json is invalid: {exc}")
    if not index_path.is_file():
        errors.append("model.safetensors.index.json is missing")
    else:
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"index JSON is invalid: {exc}")

    qcfg = config.get("quantization_config") if isinstance(config, dict) else None
    if not isinstance(qcfg, dict):
        qcfg = {}
        errors.append("config.json has no quantization_config object")
    if str(qcfg.get("quant_method", "")).lower() != "exl3":
        errors.append("quantization_config.quant_method must be 'exl3'")
    source = qcfg.get("non_routed_quantization")
    if not isinstance(source, dict):
        source = {}
        errors.append("non_routed_quantization is missing")
    if str(source.get("quant_method", "")).lower() != "deepseek_v4_fp8":
        errors.append("non_routed_quantization.quant_method must be deepseek_v4_fp8")
    if source.get("weight_block_size") != [32, 32]:
        errors.append("non_routed_quantization.weight_block_size must be [32, 32]")
    if qcfg.get("mtp_experts") != "source":
        errors.append("mtp_experts must be 'source'")

    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        errors.append("index weight_map is missing/empty")
        weight_map = {}

    expected_by_shard: dict[str, set[str]] = defaultdict(set)
    for key, shard in weight_map.items():
        expected_by_shard[str(shard)].add(str(key))
    shards = sorted(expected_by_shard)

    expected_shards = model_contract.get("expected_shards")
    if expected_shards is not None and len(shards) != int(expected_shards):
        message = (
            f"index references {len(shards)} shards; runtime lock expects "
            f"{expected_shards} for the canonical {topology} snapshot"
        )
        if strict_locked_snapshot:
            errors.append(message)
        else:
            warnings.append(message + "; allowed for a local/repacked checkpoint")

    statuses: Counter[str] = Counter()
    bad_shards: list[dict[str, str]] = []
    missing_index_tensors: list[dict[str, str]] = []
    unindexed_tensors: list[dict[str, str]] = []
    invalid_offsets: list[dict[str, Any]] = []
    size_mismatches: list[dict[str, Any]] = []
    unknown_dtypes: Counter[str] = Counter()
    shard_headers: dict[str, str] = {}
    k_hist: Counter[int] = Counter()
    routed_layer_ks: dict[int, set[int]] = defaultdict(set)
    routed_layer_examples: dict[int, dict[int, str]] = defaultdict(dict)
    codebook_markers: Counter[str] = Counter()
    total_bytes = 0

    for shard_name in shards:
        path = root / shard_name
        if not path.is_file():
            statuses["missing"] += 1
            bad_shards.append({"shard": shard_name, "status": "missing", "reason": "not found"})
            continue
        total_bytes += path.stat().st_size
        status, header, reason, payload_start, header_sha = read_safetensors_header(path)
        statuses[status] += 1
        if status != "ok" or header is None:
            bad_shards.append({"shard": shard_name, "status": status, "reason": reason or "unknown"})
            continue
        if header_sha:
            shard_headers[shard_name] = header_sha

        header_keys = {str(name) for name in header if name != "__metadata__"}
        for expected in sorted(expected_by_shard[shard_name] - header_keys):
            missing_index_tensors.append({"shard": shard_name, "tensor": expected})
        for extra in sorted(header_keys - expected_by_shard[shard_name]):
            unindexed_tensors.append({"shard": shard_name, "tensor": extra})

        payload_bytes = path.stat().st_size - payload_start
        for name, meta in header.items():
            if name == "__metadata__" or not isinstance(meta, dict):
                continue
            offsets = meta.get("data_offsets")
            start = end = -1
            offsets_valid = False
            if isinstance(offsets, list) and len(offsets) == 2:
                try:
                    start, end = int(offsets[0]), int(offsets[1])
                    offsets_valid = start >= 0 and end >= start and end <= payload_bytes
                except (TypeError, ValueError, OverflowError):
                    offsets_valid = False
            if not offsets_valid:
                invalid_offsets.append(
                    {
                        "shard": shard_name,
                        "tensor": name,
                        "data_offsets": offsets,
                        "payload_bytes": payload_bytes,
                    }
                )
            else:
                expected_bytes = expected_tensor_bytes(meta)
                if expected_bytes is None:
                    dtype = str(meta.get("dtype", "<missing>"))
                    unknown_dtypes[dtype] += 1
                elif end - start != expected_bytes:
                    size_mismatches.append(
                        {
                            "shard": shard_name,
                            "tensor": name,
                            "dtype": meta.get("dtype"),
                            "shape": meta.get("shape"),
                            "offset_bytes": end - start,
                            "expected_bytes": expected_bytes,
                        }
                    )

            if name.endswith((".mcg", "_mcg")):
                codebook_markers["mcg"] += 1
            elif name.endswith((".mul1", "_mul1")):
                codebook_markers["mul1"] += 1

            if not name.endswith(("trellis", "_trellis")):
                continue
            k = trellis_k(meta)
            if k is None:
                warnings.append(f"could not infer trellis K for {name} in {shard_name}")
                continue
            k_hist[k] += 1
            if ROUTED_RE.search(name):
                match = LAYER_RE.search(name)
                if match:
                    layer = int(match.group(1))
                    routed_layer_ks[layer].add(k)
                    routed_layer_examples[layer].setdefault(k, name)

    mixed_layers = {
        str(layer): {
            "k": sorted(ks),
            "examples": {str(k): routed_layer_examples[layer][k] for k in sorted(ks)},
        }
        for layer, ks in sorted(routed_layer_ks.items())
        if len(ks) > 1
    }
    unsupported_k = sorted(k for k in k_hist if k not in allowed_k)

    result.update(
        {
            "index_tensor_count": len(weight_map),
            "index_shard_count": len(shards),
            "expected_locked_snapshot_shards": expected_shards,
            "shard_status": dict(sorted(statuses.items())),
            "bad_shards": bad_shards,
            "missing_index_tensors": missing_index_tensors,
            "unindexed_tensors": unindexed_tensors,
            "invalid_data_offsets": invalid_offsets,
            "tensor_size_mismatches": size_mismatches,
            "unknown_dtypes": dict(sorted(unknown_dtypes.items())),
            "shard_header_sha256": dict(sorted(shard_headers.items())),
            "materialized_shard_bytes": total_bytes,
            "materialized_shard_gib": gib(total_bytes),
            "trellis_k_histogram": {str(k): n for k, n in sorted(k_hist.items())},
            "routed_layer_k": {str(layer): sorted(ks) for layer, ks in sorted(routed_layer_ks.items())},
            "mixed_k_layers": mixed_layers,
            "codebook_marker_counts": dict(sorted(codebook_markers.items())),
            "unsupported_k": unsupported_k,
        }
    )

    if bad_shards:
        errors.append(f"{len(bad_shards)} of {len(shards)} indexed shards are not materialized valid safetensors")
    if missing_index_tensors:
        errors.append(f"{len(missing_index_tensors)} index entries are absent from their declared shard headers")
    if unindexed_tensors:
        errors.append(f"{len(unindexed_tensors)} shard tensors are absent from model.safetensors.index.json")
    if invalid_offsets:
        errors.append(f"{len(invalid_offsets)} tensors have invalid/out-of-range safetensors data_offsets")
    if size_mismatches:
        errors.append(f"{len(size_mismatches)} tensors have dtype/shape byte counts inconsistent with data_offsets")
    if unknown_dtypes:
        warnings.append(f"unknown safetensors dtypes were not byte-size checked: {dict(unknown_dtypes)}")
    if unsupported_k:
        errors.append(f"trellis K outside locked runtime capability: {unsupported_k}")
    if layer_uniform_required and mixed_layers:
        errors.append(
            f"{len(mixed_layers)} routed transformer layers contain multiple physical K widths; "
            "current vllm-exl3 allocates one K per RoutedExperts layer"
        )

    usage = shutil.disk_usage(root)
    result["filesystem_free_gib"] = gib(usage.free)
    result["filesystem_total_gib"] = gib(usage.total)
    result["disk_reserve_ok"] = gib(usage.free) >= reserve_gib
    if not result["disk_reserve_ok"]:
        errors.append(
            f"filesystem free space {result['filesystem_free_gib']:.1f} GiB is below reserve {reserve_gib:.1f} GiB"
        )

    result["deployable_with_current_pinned_loader"] = not errors
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--topology", choices=("tp2", "tp4"), required=True)
    parser.add_argument("--reserve-gib", type=float, default=32.0)
    parser.add_argument(
        "--strict-locked-snapshot",
        action="store_true",
        help=(
            "require canonical locked-snapshot metadata such as exact shard count; "
            "use for materialized HF releases, not arbitrary local repacks"
        ),
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = validate_pack(
        args.model_dir,
        args.topology,
        args.reserve_gib,
        strict_locked_snapshot=args.strict_locked_snapshot,
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"Pack: {result['model_dir']}  topology={result['topology']}")
        print(f"Shard status: {result.get('shard_status', {})}")
        print(f"Trellis K histogram: {result.get('trellis_k_histogram', {})}")
        print(f"Mixed routed-K layers: {len(result.get('mixed_k_layers', {}))}")
        print(f"Materialized shards: {result.get('materialized_shard_gib', 0):.2f} GiB")
        print(f"Filesystem free: {result.get('filesystem_free_gib', 0):.2f} GiB")
        for warning in result["warnings"]:
            print(f"WARNING: {warning}")
        for error in result["errors"]:
            print(f"ERROR: {error}")
        if result.get("bad_shards"):
            print("Bad shards:")
            for item in result["bad_shards"]:
                print(f"  {item['shard']}: {item['status']} - {item['reason']}")
        print(
            "DEPLOYABLE_CURRENT_LOADER="
            + ("YES" if result["deployable_with_current_pinned_loader"] else "NO")
        )
    return 0 if result["deployable_with_current_pinned_loader"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
