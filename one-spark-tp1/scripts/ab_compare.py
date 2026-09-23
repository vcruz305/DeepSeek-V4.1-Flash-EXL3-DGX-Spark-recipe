"""Read ab_dispatch.py output files and report the paired result.

Two questions, answered separately:

  quality     do the arms emit the same token ids on the same prompt? A lever that changes the
              output is not a speedup, it is a different model (see "Speculation is not
              output-exact" in BENCHMARKS.md for why this is not hypothetical here).
  throughput  per subject, never pooled: decode speed spans 2-3x across subjects, so a pooled
              mean reports which subjects a run happened to draw.

Determinism is checked first and on its own: an arm run twice must produce identical hashes.
The legacy per-K-group dispatch accumulates its experts' outputs with atomics when it is
called without a slot table, and atomic fp32 addition is order-dependent, so an arm can fail
this check while being perfectly fast. EXL3_MOE_MIXED_BSZ1 was rejected for exactly that.

Usage:
    python ab_compare.py --baseline A=runs/a1.jsonl,runs/a2.jsonl \\
                         --arm C=runs/c1.jsonl,runs/c2.jsonl
"""
import argparse
import json
import statistics
import sys
from collections import defaultdict


def load(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                r = json.loads(line)
                if r.get("phase") == "gen":
                    r["_file"] = p
                    rows.append(r)
    return rows


def by_subject(rows):
    d = defaultdict(list)
    for r in rows:
        d[r["subject"]].append(r)
    return d


def determinism(label, rows):
    """Same arm, same subject, repeated: the ids must be identical every time."""
    bad = []
    for subj, rs in by_subject(rows).items():
        shas = {r["ids_sha"] for r in rs if r.get("n_out")}
        if len(shas) > 1:
            bad.append((subj, sorted(shas)))
    if bad:
        print(f"  [FAIL] {label}: same prompt gave different ids")
        for subj, shas in bad:
            print(f"         {subj}: {' '.join(shas)}")
    else:
        print(f"  [ok]   {label}: self-deterministic across repeats")
    return not bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="LABEL=file[,file...]")
    ap.add_argument("--arm", action="append", required=True, help="LABEL=file[,file...]")
    args = ap.parse_args()

    def parse(spec):
        label, _, files = spec.partition("=")
        return label, load(files.split(","))

    base_label, base_rows = parse(args.baseline)
    arms = [parse(a) for a in args.arm]
    if not base_rows:
        sys.exit("baseline has no gen rows")

    print("Determinism (an arm against itself)")
    ok = determinism(base_label, base_rows)
    for label, rows in arms:
        ok &= determinism(label, rows)

    print("\nQuality gate (ids against the baseline arm)")
    base_sha = {}
    for subj, rs in by_subject(base_rows).items():
        shas = {r["ids_sha"] for r in rs}
        base_sha[subj] = next(iter(shas)) if len(shas) == 1 else None
    for label, rows in arms:
        diffs = []
        for subj, rs in by_subject(rows).items():
            b = base_sha.get(subj)
            if b is None:
                continue
            for r in rs:
                if r["ids_sha"] != b:
                    diffs.append(subj)
                    break
        if diffs:
            print(f"  [DIFF] {label}: output differs from {base_label} on {sorted(set(diffs))}")
            print(f"         a throughput win here is a different generation, not a faster one;"
                  f" run spec_exactness.py before reading the numbers below")
        else:
            print(f"  [ok]   {label}: ids identical to {base_label} on every shared subject")

    print("\nThroughput, per subject (median ms/token over repeats)")
    subjects = sorted(by_subject(base_rows))
    head = f"{'subject':<16}{base_label:>12}"
    for label, _ in arms:
        head += f"{label:>12}{'delta':>9}"
    print(head)
    for subj in subjects:
        b = [r["ms_per_token"] for r in by_subject(base_rows)[subj] if r["ms_per_token"]]
        if not b:
            continue
        bm = statistics.median(b)
        line = f"{subj:<16}{bm:>12.2f}"
        for label, rows in arms:
            a = [r["ms_per_token"] for r in by_subject(rows).get(subj, []) if r["ms_per_token"]]
            if not a:
                line += f"{'-':>12}{'-':>9}"
                continue
            am = statistics.median(a)
            line += f"{am:>12.2f}{(bm - am) / bm * 100:>8.1f}%"
        print(line)

    print("\nDraft accounting (median)")
    for label, rows in [(base_label, base_rows)] + arms:
        acc = [r["acceptance"] for r in rows if r.get("acceptance")]
        tpr = [r["tokens_per_round"] for r in rows if r.get("tokens_per_round")]
        mpr = [r["ms_per_round"] for r in rows if r.get("ms_per_round")]
        print(f"  {label:<10} acceptance {statistics.median(acc):.3f}  "
              f"tokens/round {statistics.median(tpr):.3f}  ms/round {statistics.median(mpr):.1f}"
              if acc and tpr and mpr else f"  {label:<10} (no draft rows)")

    print("\nA lever passes only if it is deterministic, matches the baseline ids, and is faster"
          "\non the same subject in the same session. Anything else is a finding, not a win.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
