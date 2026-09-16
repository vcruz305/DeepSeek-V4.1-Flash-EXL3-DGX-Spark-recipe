#!/usr/bin/env python3
"""Probe a Hugging Face EXL3 checkpoint without downloading tensor payloads.

The probe downloads only config/index JSON and the safetensors header byte ranges.
It aborts if a shard endpoint ignores HTTP Range, so an audit cannot accidentally
turn into a hundreds-of-GiB model download.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
import re
import struct
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from runtime_lock import load_lock
from validate_pack import (
    DTYPE_BYTES,
    expected_tensor_bytes,
    physical_trellis_k_capability,
    trellis_k,
)

MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_REMOTE_HEADER_BYTES = 256 * 1024 * 1024
LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
ROUTED_RE = re.compile(r"(?:^|\.)(?:ffn\.)?experts(?:\.|$)")


def hf_resolve_url(repo_id: str, revision: str, filename: str) -> str:
    repo = quote(repo_id, safe="/")
    rev = quote(revision, safe="")
    name = quote(filename, safe="/")
    return f"https://huggingface.co/{repo}/resolve/{rev}/{name}?download=true"


def _headers(token: str | None, byte_range: str | None = None) -> dict[str, str]:
    headers = {"User-Agent": "dsv41-exl3-recipe/remote-pack-probe"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if byte_range:
        headers["Range"] = byte_range
    return headers


def fetch_small(url: str, token: str | None, limit: int = MAX_JSON_BYTES) -> bytes:
    request = Request(url, headers=_headers(token))
    with urlopen(request, timeout=60) as response:
        length = response.headers.get("Content-Length")
        if length is not None and int(length) > limit:
            raise ValueError(f"refusing {int(length)}-byte non-shard download from {url}")
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"response exceeded {limit} bytes: {url}")
    return data


def fetch_range(url: str, token: str | None, start: int, end: int) -> tuple[bytes, int]:
    if end < start:
        raise ValueError("invalid byte range")
    requested = end - start + 1
    request = Request(url, headers=_headers(token, f"bytes={start}-{end}"))
    with urlopen(request, timeout=120) as response:
        status = getattr(response, "status", response.getcode())
        if status != 206:
            raise ValueError(
                f"server returned HTTP {status}, not 206, for Range {start}-{end}; "
                "aborting to avoid an accidental full-shard download"
            )
        content_range = response.headers.get("Content-Range", "")
        match = re.fullmatch(r"bytes\s+(\d+)-(\d+)/(\d+|\*)", content_range.strip())
        if not match:
            raise ValueError(f"invalid/missing Content-Range: {content_range!r}")
        got_start, got_end = int(match.group(1)), int(match.group(2))
        if (got_start, got_end) != (start, end):
            raise ValueError(
                f"server returned unexpected byte range {got_start}-{got_end}; expected {start}-{end}"
            )
        if match.group(3) == "*":
            raise ValueError("remote shard size is unknown in Content-Range")
        total = int(match.group(3))
        data = response.read(requested + 1)
    if len(data) != requested:
        raise ValueError(f"range returned {len(data)} bytes; expected {requested}")
    return data, total


def fetch_safetensors_header(
    repo_id: str, revision: str, filename: str, token: str | None
) -> tuple[dict[str, Any], int, int]:
    url = hf_resolve_url(repo_id, revision, filename)
    prefix, total = fetch_range(url, token, 0, 7)
    header_len = struct.unpack("<Q", prefix)[0]
    if header_len <= 1 or header_len > MAX_REMOTE_HEADER_BYTES:
        raise ValueError(f"implausible safetensors header length {header_len}")
    header_raw, total_again = fetch_range(url, token, 8, 8 + header_len - 1)
    if total_again != total:
        raise ValueError(f"remote shard size changed during probe: {total} -> {total_again}")
    try:
        header = json.loads(header_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"safetensors header JSON is invalid: {exc}") from exc
    if not isinstance(header, dict):
        raise ValueError("safetensors header is not a JSON object")
    return header, total, 8 + header_len


def probe(repo_id: str, revision: str, topology: str, token: str | None) -> dict[str, Any]:
    lock = load_lock()
    model_contract = lock["models"][topology]
    allowed_k = physical_trellis_k_capability(lock)
    layer_uniform_required = not bool(
        lock["capabilities"].get("tensor_level_mixed_k_within_layer", False)
    )
    result: dict[str, Any] = {
        "schema": "dsv41-exl3-remote-pack-probe.v1",
        "repo_id": repo_id,
        "revision": revision,
        "topology": topology,
        "errors": [],
        "warnings": [],
    }
    errors: list[str] = result["errors"]
    warnings: list[str] = result["warnings"]

    config_url = hf_resolve_url(repo_id, revision, "config.json")
    index_url = hf_resolve_url(repo_id, revision, "model.safetensors.index.json")
    try:
        config = json.loads(fetch_small(config_url, token))
        index = json.loads(fetch_small(index_url, token))
    except (HTTPError, URLError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"could not fetch config/index at locked revision: {type(exc).__name__}: {exc}")
        result["remote_layout_compatible"] = False
        return result

    qcfg = config.get("quantization_config") if isinstance(config, dict) else None
    if not isinstance(qcfg, dict):
        qcfg = {}
        errors.append("config.json has no quantization_config object")
    if str(qcfg.get("quant_method", "")).lower() != "exl3":
        errors.append("quantization_config.quant_method must be exl3")
    source = qcfg.get("non_routed_quantization")
    if not isinstance(source, dict):
        source = {}
        errors.append("non_routed_quantization is missing")
    if str(source.get("quant_method", "")).lower() != "deepseek_v4_fp8":
        errors.append("non_routed_quantization.quant_method must be deepseek_v4_fp8")
    if source.get("weight_block_size") != [32, 32]:
        errors.append("non_routed_quantization.weight_block_size must be [32, 32]")
    if qcfg.get("mtp_experts") != "source":
        errors.append("mtp_experts must be source")

    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        errors.append("model.safetensors.index.json has no weight_map")
        result["remote_layout_compatible"] = False
        return result

    expected_by_shard: dict[str, set[str]] = defaultdict(set)
    for tensor, shard in weight_map.items():
        expected_by_shard[str(shard)].add(str(tensor))
    shards = sorted(expected_by_shard)
    expected_shards = model_contract.get("expected_shards")
    if expected_shards is not None and len(shards) != int(expected_shards):
        errors.append(f"index references {len(shards)} shards; lock expects {expected_shards}")

    statuses: Counter[str] = Counter()
    missing_index_tensors: list[dict[str, str]] = []
    unindexed_tensors: list[dict[str, str]] = []
    invalid_offsets: list[dict[str, Any]] = []
    size_mismatches: list[dict[str, Any]] = []
    unknown_dtypes: Counter[str] = Counter()
    shard_sizes: dict[str, int] = {}
    k_hist: Counter[int] = Counter()
    routed_layer_ks: dict[int, set[int]] = defaultdict(set)
    routed_examples: dict[int, dict[int, str]] = defaultdict(dict)

    for shard_name in shards:
        try:
            header, total_bytes, payload_start = fetch_safetensors_header(
                repo_id, revision, shard_name, token
            )
            statuses["ok"] += 1
            shard_sizes[shard_name] = total_bytes
        except Exception as exc:
            statuses["error"] += 1
            errors.append(f"{shard_name}: {type(exc).__name__}: {exc}")
            continue

        header_keys = {str(name) for name in header if name != "__metadata__"}
        for expected in sorted(expected_by_shard[shard_name] - header_keys):
            missing_index_tensors.append({"shard": shard_name, "tensor": expected})
        for extra in sorted(header_keys - expected_by_shard[shard_name]):
            unindexed_tensors.append({"shard": shard_name, "tensor": extra})

        payload_bytes = total_bytes - payload_start
        for name, meta in header.items():
            if name == "__metadata__" or not isinstance(meta, dict):
                continue
            offsets = meta.get("data_offsets")
            valid = False
            start = end = -1
            if isinstance(offsets, list) and len(offsets) == 2:
                try:
                    start, end = int(offsets[0]), int(offsets[1])
                    valid = start >= 0 and end >= start and end <= payload_bytes
                except (TypeError, ValueError, OverflowError):
                    valid = False
            if not valid:
                invalid_offsets.append({
                    "shard": shard_name,
                    "tensor": name,
                    "data_offsets": offsets,
                    "payload_bytes": payload_bytes,
                })
            else:
                expected_bytes = expected_tensor_bytes(meta)
                if expected_bytes is None:
                    unknown_dtypes[str(meta.get("dtype", "<missing>"))] += 1
                elif end - start != expected_bytes:
                    size_mismatches.append({
                        "shard": shard_name,
                        "tensor": name,
                        "dtype": meta.get("dtype"),
                        "shape": meta.get("shape"),
                        "offset_bytes": end - start,
                        "expected_bytes": expected_bytes,
                    })

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
                    routed_examples[layer].setdefault(k, name)

    mixed_layers = {
        str(layer): {
            "k": sorted(ks),
            "examples": {str(k): routed_examples[layer][k] for k in sorted(ks)},
        }
        for layer, ks in sorted(routed_layer_ks.items())
        if len(ks) > 1
    }
    unsupported_k = sorted(k for k in k_hist if k not in allowed_k)

    if missing_index_tensors:
        errors.append(f"{len(missing_index_tensors)} index entries are missing from shard headers")
    if unindexed_tensors:
        errors.append(f"{len(unindexed_tensors)} shard tensors are absent from the index")
    if invalid_offsets:
        errors.append(f"{len(invalid_offsets)} tensors have invalid/out-of-range data_offsets")
    if size_mismatches:
        errors.append(f"{len(size_mismatches)} tensors have dtype/shape byte-count mismatches")
    if unknown_dtypes:
        warnings.append(f"unknown dtypes were not byte-size checked: {dict(unknown_dtypes)}")
    if unsupported_k:
        errors.append(f"trellis K outside locked runtime capability: {unsupported_k}")
    if layer_uniform_required and mixed_layers:
        errors.append(
            f"{len(mixed_layers)} routed transformer layers contain multiple physical K widths; "
            "current vllm-exl3 allocates one K per RoutedExperts layer"
        )

    result.update({
        "index_tensor_count": len(weight_map),
        "index_shard_count": len(shards),
        "shard_status": dict(statuses),
        "shard_sizes": shard_sizes,
        "remote_total_shard_gib": sum(shard_sizes.values()) / (1024**3),
        "missing_index_tensors": missing_index_tensors,
        "unindexed_tensors": unindexed_tensors,
        "invalid_data_offsets": invalid_offsets,
        "tensor_size_mismatches": size_mismatches,
        "unknown_dtypes": dict(unknown_dtypes),
        "trellis_k_histogram": {str(k): count for k, count in sorted(k_hist.items())},
        "mixed_k_layers": mixed_layers,
        "unsupported_k": unsupported_k,
        "remote_layout_compatible": not errors,
    })
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, choices=(2, 4), required=True)
    parser.add_argument("--repo", default="")
    parser.add_argument("--revision", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    lock = load_lock()
    topology = f"tp{args.tp}"
    model = lock["models"][topology]
    repo = args.repo or str(model["repo_id"])
    revision = args.revision or str(model.get("revision") or "main")
    token = os.environ.get("HF_TOKEN") or None

    result = probe(repo, revision, topology, token)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"Remote pack: {repo}@{revision} ({topology})")
        print(f"Shards: {result.get('index_shard_count', 0)}")
        print(f"Header-probed size: {result.get('remote_total_shard_gib', 0):.2f} GiB")
        print(f"Trellis K histogram: {result.get('trellis_k_histogram', {})}")
        print(f"Mixed routed-K layers: {len(result.get('mixed_k_layers', {}))}")
        for warning in result["warnings"]:
            print(f"WARNING: {warning}")
        for error in result["errors"]:
            print(f"ERROR: {error}")
        print("REMOTE_LAYOUT_COMPATIBLE=" + ("YES" if result.get("remote_layout_compatible") else "NO"))
    return 0 if result.get("remote_layout_compatible") else 2


if __name__ == "__main__":
    raise SystemExit(main())
