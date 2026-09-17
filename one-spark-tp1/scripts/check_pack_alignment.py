#!/usr/bin/env python3
"""Report how much of an EXL3 pack can be zero-copy aliased on a GB10, and whether
a 64-byte re-lay would help.

    python check_pack_alignment.py /models/DSV4.1-Flash-SAGE-EXL3-1.59bpw

Reads safetensors headers only; it never loads weights and never writes anything.

This mirrors the rule the ExLlamaV3 ATS loader actually applies
(exllamav3/loader/safetensors.py): a tensor is aliased out of the mmap when its
absolute file offset is a multiple of the alignment its dtype needs, and is
copied into CUDA memory otherwise. int16 (EXL3 trellis data) needs
EXL3_ATS_MMAP_ALIGN, 16 by default; every other dtype needs only its item size.
Tensors below EXL3_ATS_MMAP_MIN (1 MiB) are copied regardless.

A misaligned tensor is not an error. It is copied into memory that cannot be
reclaimed, which is what makes a large pack stop fitting in 128 GB.

Exit status:
    0  nothing meaningful would be copied; no re-lay needed
    1  a re-lay would move a significant amount out of CUDA memory
    2  the directory could not be read
"""
import argparse
import json
import os
import struct
import sys
from collections import defaultdict

ITEMSIZE = {
    "F64": 8, "I64": 8, "U64": 8,
    "F32": 4, "I32": 4, "U32": 4,
    "F16": 2, "BF16": 2, "I16": 2, "U16": 2,
    "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1,
    "I8": 1, "U8": 1, "BOOL": 1,
}

GIB = 1 << 30


def required_align(dtype, ats_align):
    # int16 carries EXL3 trellis data and is the only dtype needing the wider grid
    if dtype in ("I16", "U16"):
        return ats_align
    return ITEMSIZE.get(dtype, 1)


def read_header(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return 8 + n, json.loads(fh.read(n))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pack_dir", help="directory holding model-*.safetensors")
    ap.add_argument("--align", type=int, default=16,
                    help="EXL3_ATS_MMAP_ALIGN (default 16)")
    ap.add_argument("--min-bytes", type=int, default=1 << 20,
                    help="EXL3_ATS_MMAP_MIN (default 1048576)")
    ap.add_argument("--threshold-gib", type=float, default=1.0,
                    help="exit 1 when more than this many GiB would be copied")
    args = ap.parse_args()

    if not os.path.isdir(args.pack_dir):
        print("not a directory: %s" % args.pack_dir, file=sys.stderr)
        return 2

    shards = sorted(f for f in os.listdir(args.pack_dir)
                    if f.endswith(".safetensors"))
    if not shards:
        print("no .safetensors files in %s" % args.pack_dir, file=sys.stderr)
        return 2

    aliased = misaligned = small = 0
    n_alias = n_mis = n_small = 0
    by_dtype = defaultdict(lambda: [0, 0])
    overlay_files = []
    worst = []

    for name in shards:
        path = os.path.join(args.pack_dir, name)
        try:
            data0, hdr = read_header(path)
        except Exception as e:
            print("  ! could not read %s (%s)" % (name, type(e).__name__))
            continue
        if not name.startswith("model-"):
            overlay_files.append(name)
        for key, e in hdr.items():
            if key in ("__metadata__", "_header_offset") or key.startswith("__align_pad__."):
                continue
            if not isinstance(e, dict) or "data_offsets" not in e:
                continue
            beg, end = e["data_offsets"]
            nbytes = end - beg
            dtype = e["dtype"]
            if nbytes < args.min_bytes:
                small += nbytes
                n_small += 1
                continue
            need = required_align(dtype, args.align)
            if (data0 + beg) % need == 0:
                aliased += nbytes
                n_alias += 1
                by_dtype[dtype][0] += nbytes
            else:
                misaligned += nbytes
                n_mis += 1
                by_dtype[dtype][1] += nbytes
                if len(worst) < 5:
                    worst.append((key, dtype, need, nbytes))

    total = aliased + misaligned + small
    print("pack            : %s" % args.pack_dir)
    print("safetensors     : %d files (%d look like overlay parts)"
          % (len(shards), len(overlay_files)))
    print("total tensor data: %9.2f GiB" % (total / GIB))
    print()
    print("  aliased (page cache, reclaimable) : %9.2f GiB  %7d tensors"
          % (aliased / GIB, n_alias))
    print("  COPIED, off the alignment grid    : %9.2f GiB  %7d tensors"
          % (misaligned / GIB, n_mis))
    print("  copied, under %d bytes          : %9.2f GiB  %7d tensors"
          % (args.min_bytes, small / GIB, n_small))
    print()
    print("  per dtype (aliased | misaligned), GiB:")
    for dtype in sorted(by_dtype, key=lambda d: -sum(by_dtype[d])):
        al, mi = by_dtype[dtype]
        print("    %-10s need=%-3d %9.2f | %9.2f"
              % (dtype, required_align(dtype, args.align), al / GIB, mi / GIB))

    if worst:
        print()
        print("  examples of tensors that would be copied:")
        for key, dtype, need, nbytes in worst:
            print("    %-52s %-8s need %2d  %.3f GiB"
                  % (key[:52], dtype, need, nbytes / GIB))

    print()
    if misaligned / GIB > args.threshold_gib:
        print("VERDICT: a 64-byte re-lay would move %.2f GiB out of unevictable CUDA"
              % (misaligned / GIB))
        print("         memory and into reclaimable page cache. See the re-lay command in")
        print("         one-spark-tp1/README.md. Weight values are unchanged by it.")
        return 1

    print("VERDICT: this pack is already laid out for zero-copy aliasing; no re-lay needed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
