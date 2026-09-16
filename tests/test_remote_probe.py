from __future__ import annotations

import json
from pathlib import Path
import struct
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import probe_remote_pack as probe  # noqa: E402


class FakeResponse:
    def __init__(self, status: int, body: bytes, headers: dict[str, str]):
        self.status = status
        self._body = body
        self.headers = headers

    def getcode(self) -> int:
        return self.status

    def read(self, _n: int = -1) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class RemoteProbeTests(unittest.TestCase):
    def _probe_physical_k(self, k: int) -> dict:
        tensor = "model.layers.0.ffn.experts.0.w1_trellis"
        config = {
            "quantization_config": {
                "quant_method": "exl3",
                "mtp_experts": "source",
                "non_routed_quantization": {
                    "quant_method": "deepseek_v4_fp8",
                    "weight_block_size": [32, 32],
                },
            }
        }
        index = {"weight_map": {tensor: "model-00001-of-00001.safetensors"}}
        header = {
            tensor: {
                "dtype": "I16",
                "shape": [1, 1, k * 16],
                "data_offsets": [0, k * 32],
            }
        }
        lock = {
            "models": {"tp2": {"expected_shards": 1}},
            "capabilities": {
                "accepted_exl3_config_k": [2, 3, 4, 5, 6, 7, 8],
                "exllamav3_moe_kernel_k": [1, 2, 3, 4, 5, 6, 7, 8],
                "tensor_level_mixed_k_within_layer": True,
            },
        }
        with (
            patch.object(probe, "load_lock", return_value=lock),
            patch.object(
                probe,
                "fetch_small",
                side_effect=[json.dumps(config).encode(), json.dumps(index).encode()],
            ),
            patch.object(
                probe,
                "fetch_safetensors_header",
                return_value=(header, 128 + k * 32, 128),
            ),
        ):
            return probe.probe("owner/model", "deadbeef", "tp2", None)

    def test_hf_resolve_url_pins_revision(self) -> None:
        url = probe.hf_resolve_url("owner/model", "deadbeef", "model-00001.safetensors")
        self.assertIn("/owner/model/resolve/deadbeef/model-00001.safetensors", url)

    def test_range_probe_requires_http_206(self) -> None:
        response = FakeResponse(200, b"12345678", {"Content-Length": "8"})
        with patch.object(probe, "urlopen", return_value=response):
            with self.assertRaisesRegex(ValueError, "not 206"):
                probe.fetch_range("https://example.invalid/shard", None, 0, 7)

    def test_range_probe_checks_content_range(self) -> None:
        response = FakeResponse(
            206,
            b"12345678",
            {"Content-Range": "bytes 0-7/1234", "Content-Length": "8"},
        )
        with patch.object(probe, "urlopen", return_value=response):
            data, total = probe.fetch_range("https://example.invalid/shard", None, 0, 7)
        self.assertEqual(data, b"12345678")
        self.assertEqual(total, 1234)

    def test_remote_safetensors_header_is_range_only(self) -> None:
        header = {
            "model.layers.0.ffn.experts.0.w1_trellis": {
                "dtype": "I16",
                "shape": [1, 1, 32],
                "data_offsets": [0, 64],
            }
        }
        raw = json.dumps(header, separators=(",", ":")).encode()
        total = 8 + len(raw) + 64
        responses = [
            FakeResponse(
                206,
                struct.pack("<Q", len(raw)),
                {"Content-Range": f"bytes 0-7/{total}"},
            ),
            FakeResponse(
                206,
                raw,
                {"Content-Range": f"bytes 8-{7 + len(raw)}/{total}"},
            ),
        ]
        with patch.object(probe, "urlopen", side_effect=responses) as mocked:
            parsed, got_total, payload_start = probe.fetch_safetensors_header(
                "owner/model", "deadbeef", "model-00001.safetensors", None
            )
        self.assertEqual(parsed, header)
        self.assertEqual(got_total, total)
        self.assertEqual(payload_start, 8 + len(raw))
        self.assertEqual(mocked.call_count, 2)

    def test_remote_probe_accepts_physical_k1_kernel_capability(self) -> None:
        report = self._probe_physical_k(1)
        self.assertTrue(report["remote_layout_compatible"], report)
        self.assertEqual(report["unsupported_k"], [])

    def test_remote_probe_rejects_k_outside_kernel_capability(self) -> None:
        report = self._probe_physical_k(9)
        self.assertFalse(report["remote_layout_compatible"])
        self.assertEqual(report["unsupported_k"], [9])

    def test_collective_is_part_of_normal_preflight(self) -> None:
        preflight = (ROOT / "scripts" / "preflight.sh").read_text()
        self.assertIn("cluster_collective.py", preflight)
        self.assertIn("SKIP_NCCL_COLLECTIVE", preflight)
        self.assertTrue((ROOT / "scripts" / "cluster_collective.py").is_file())
        self.assertTrue((ROOT / "scripts" / "cluster_collective.sh").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
