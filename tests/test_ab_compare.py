from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPARE = ROOT / "one-spark-tp1" / "scripts" / "ab_compare.py"


def row(arm, subject, ms, sha, rep=0):
    return json.dumps({
        "arm": arm, "mode": "repeat", "phase": "gen", "rep": rep, "subject": subject,
        "ms_per_token": ms, "tok_s": round(1000.0 / ms, 2), "accepted": 900, "rejected": 100,
        "acceptance": 0.9, "tokens_per_round": 4.8, "ms_per_round": ms * 4.8,
        "n_out": 256, "ids_sha": sha, "first_ids": [1, 2, 3],
    })


def write(tmp, name, rows):
    p = Path(tmp) / name
    p.write_text("\n".join(rows) + "\n")
    return str(p)


def run(*args):
    """The comparer exits non-zero when an arm fails the determinism check, so the caller gets
    both the report and the verdict."""
    r = subprocess.run([sys.executable, str(COMPARE), *args], capture_output=True, text=True)
    return r.returncode, r.stdout


class AbCompareTests(unittest.TestCase):
    """The comparer is the thing that decides whether a dispatch lever ships, so its three
    verdicts (non-deterministic, output-changing, faster) each need to survive a case that
    would produce the wrong answer if the checks were pooled or reordered."""

    def test_expected_use_faster_arm_with_identical_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            a = write(tmp, "a.jsonl", [row("A", "computing", 40.0, "aaa"),
                                       row("A", "bread", 32.0, "bbb")])
            c = write(tmp, "c.jsonl", [row("C", "computing", 36.0, "aaa"),
                                       row("C", "bread", 30.0, "bbb")])
            code, out = run("--baseline", f"A={a}", "--arm", f"C={c}")
            self.assertEqual(code, 0)
            self.assertIn("ids identical to A", out)
            self.assertIn("10.0%", out)   # 40.0 -> 36.0 on computing
            self.assertIn("6.2%", out)    # 32.0 -> 30.0 on bread

    def test_edge_case_same_arm_different_ids_is_a_determinism_failure(self) -> None:
        # The legacy per-K-group dispatch accumulates with atomics when it gets no slot table,
        # and atomic fp32 addition is order-dependent: the arm can be fast and still fail here.
        with tempfile.TemporaryDirectory() as tmp:
            a = write(tmp, "a.jsonl", [row("A", "computing", 40.0, "aaa")])
            c = write(tmp, "c.jsonl", [row("C", "computing", 36.0, "ccc", rep=0),
                                       row("C", "computing", 36.1, "ddd", rep=1)])
            code, out = run("--baseline", f"A={a}", "--arm", f"C={c}")
            self.assertEqual(code, 1)
            self.assertIn("[FAIL] C", out)
            self.assertIn("same prompt gave different ids", out)

    def test_failure_case_output_changed_is_flagged_even_when_faster(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            a = write(tmp, "a.jsonl", [row("A", "computing", 40.0, "aaa")])
            c = write(tmp, "c.jsonl", [row("C", "computing", 20.0, "zzz")])
            code, out = run("--baseline", f"A={a}", "--arm", f"C={c}")
            self.assertEqual(code, 0)          # deterministic, but
            self.assertIn("[DIFF] C", out)     # not the same generation
            self.assertIn("different generation", out)

    def test_subjects_are_compared_pairwise_not_pooled(self) -> None:
        # Decode speed spans 2-3x across subjects. An arm that only ran the fast subject must
        # not look like a win against a baseline that also ran the slow one.
        with tempfile.TemporaryDirectory() as tmp:
            a = write(tmp, "a.jsonl", [row("A", "orbital", 32.0, "aaa"),
                                       row("A", "medieval_trade", 90.0, "bbb")])
            c = write(tmp, "c.jsonl", [row("C", "orbital", 33.0, "aaa")])
            code, out = run("--baseline", f"A={a}", "--arm", f"C={c}")
            self.assertEqual(code, 0)
            line = [l for l in out.splitlines() if l.startswith("orbital")][0]
            self.assertIn("-3.1%", line)
            trade = [l for l in out.splitlines() if l.startswith("medieval_trade")][0]
            self.assertIn("-", trade.split()[-1])   # no C figure, reported as missing


if __name__ == "__main__":
    unittest.main()
