from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
QUALIFICATION_PATH = ROOT / "qualifications" / "sage159-tp2-a78949a.json"


class Sage159QualificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.qualification = json.loads(QUALIFICATION_PATH.read_text(encoding="utf-8"))
        cls.attestation_path = ROOT / cls.qualification["model"]["metadata_attestation"]
        cls.attestation = json.loads(cls.attestation_path.read_text(encoding="utf-8"))

    def test_receipt_is_bound_to_model_and_plugin_revisions(self) -> None:
        receipt = self.qualification
        attestation = self.attestation
        self.assertEqual(receipt["schema"], "dsv41-exl3-hardware-qualification.v1")
        self.assertEqual(receipt["status"], "pass")
        self.assertEqual(receipt["model"]["repo_id"], attestation["model_repo"])
        self.assertEqual(receipt["model"]["revision"], attestation["revision"])
        self.assertEqual(receipt["model"]["expected_shards"], 17)
        self.assertEqual(
            hashlib.sha256(self.attestation_path.read_bytes()).hexdigest(),
            receipt["model"]["metadata_attestation_sha256"],
        )
        self.assertEqual(
            receipt["model"]["tensor_shard_bytes"],
            attestation["materialized_shard_bytes"],
        )
        expected_names = {
            f"model-{index:05d}-of-00017.safetensors" for index in range(1, 18)
        }
        self.assertEqual(set(attestation["shard_header_sha256"]), expected_names)
        for digest in attestation["shard_header_sha256"].values():
            self.assertEqual(len(digest), 64)
            int(digest, 16)
        self.assertEqual(attestation["trellis_k_histogram"]["1"], 27210)
        self.assertEqual(
            sum(attestation["trellis_k_histogram"].values()),
            attestation["codebook_marker_counts"]["mul1"],
        )
        self.assertEqual(
            receipt["runtime"]["vllm_exl3_commit"],
            "a78949ac82e071f2c3ca7477910d5d9641df6d80",
        )

    def test_context_claim_is_limited_to_observed_server_tokens(self) -> None:
        serving = self.qualification["serving"]
        probes = self.qualification["measurements"]["context_probes"]
        self.assertEqual(serving["max_model_len"], 131072)
        self.assertEqual([probe["prompt_tokens"] for probe in probes], [30835, 63603, 126067])
        self.assertTrue(all(probe["passed"] for probe in probes))
        self.assertTrue(all(probe["keys_found"] == 3 for probe in probes))
        self.assertLess(max(probe["prompt_tokens"] for probe in probes), serving["max_model_len"])

    def test_reference_profile_matches_receipt(self) -> None:
        profile_path = ROOT / self.qualification["reference_profile"]
        values = {}
        for raw in profile_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            key, value = line.split("=", 1)
            values[key] = value
        self.assertEqual(values["MODEL_ID"], self.qualification["model"]["repo_id"])
        self.assertEqual(values["MODEL_REV"], self.qualification["model"]["revision"])
        self.assertEqual(values["VLLM_EXL3_REF"], self.qualification["runtime"]["vllm_exl3_commit"])
        self.assertEqual(int(values["MAX_MODEL_LEN"]), self.qualification["serving"]["max_model_len"])
        self.assertEqual(values["SPECULATIVE_POLICY"], "none_requested")
        self.assertEqual(values["TP"], "2")
        self.assertEqual(values["NNODES"], "2")

    def test_canonical_runtime_lock_is_explicitly_out_of_scope(self) -> None:
        lock = json.loads((ROOT / "runtime.lock.json").read_text(encoding="utf-8"))
        scope = self.qualification["scope"]
        self.assertTrue(scope["canonical_runtime_lock_unchanged"])
        self.assertEqual(scope["runtime_derivation"], "external_reference")
        self.assertNotEqual(
            self.qualification["runtime"]["vllm_exl3_commit"],
            lock["vllm_exl3"]["commit"],
        )

    def test_readme_links_the_qualification(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("docs/SAGE159_TP2_QUALIFICATION.md", readme)
        self.assertIn("126,067", readme)


if __name__ == "__main__":
    unittest.main()
