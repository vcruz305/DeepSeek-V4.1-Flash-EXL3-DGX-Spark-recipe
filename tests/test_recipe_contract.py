#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import re
import struct
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

import sys
sys.path.insert(0, str(SCRIPTS))
from runtime_lock import load_lock  # noqa: E402
from validate_pack import validate_pack  # noqa: E402


class RecipeContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lock = load_lock(ROOT / "runtime.lock.json")

    def test_dockerfile_defaults_match_lock(self) -> None:
        dockerfile = (ROOT / "Dockerfile.spark").read_text()

        def arg(name: str) -> str:
            match = re.search(rf"^ARG {re.escape(name)}=(.+)$", dockerfile, re.M)
            self.assertIsNotNone(match, f"missing Docker ARG {name}")
            return match.group(1).strip()

        self.assertEqual(arg("BASE_IMAGE"), self.lock["base_image"]["ref"])
        self.assertEqual(arg("VLLM_EXL3_REPO"), self.lock["vllm_exl3"]["repo"])
        self.assertEqual(arg("VLLM_EXL3_REF"), self.lock["vllm_exl3"]["commit"])
        self.assertEqual(arg("EXLLAMAV3_REF"), self.lock["exllamav3"]["commit"])
        self.assertEqual(arg("CUDA_ARCH_LIST"), self.lock["torch_cuda_arch_list"])
        self.assertIn("physical_fused_k_guard_installed", dockerfile)
        self.assertIn('fused_k_source"] == "physical_trellis_geometry"', dockerfile)

    def test_first_boot_and_mixed_k_contract(self) -> None:
        self.assertEqual(self.lock["first_boot"]["max_model_len"], 8192)
        self.assertEqual(self.lock["first_boot"]["max_num_seqs"], 1)
        self.assertTrue(self.lock["first_boot"]["text_only"])
        self.assertTrue(self.lock["first_boot"]["eager"])
        self.assertFalse(self.lock["first_boot"]["dspark"])
        self.assertFalse(self.lock["first_boot"]["native_moe"])
        caps = self.lock["capabilities"]
        self.assertEqual(caps["accepted_exl3_config_k"], [2, 3, 4, 5, 6, 7, 8])
        self.assertTrue(caps["tensor_level_mixed_k_within_layer"])
        self.assertEqual(caps["routed_allocation_scope"], "per_expert_exact_trellis_shapes")
        self.assertEqual(caps["heterogeneous_mixed_k_dispatch"], "linear_exl3_python_loop")
        self.assertEqual(caps["uniform_k_fused_k_source"], "physical_trellis_geometry")
        self.assertFalse(caps["heterogeneous_mixed_k_cudagraph_qualified"])
        self.assertEqual(caps["mixed_k_first_boot"], "eager")

    @staticmethod
    def _base_config() -> dict:
        return {
            "architectures": ["DeepseekV41ForCausalLM"],
            "model_type": "deepseek_v41",
            "quantization_config": {
                "quant_method": "exl3",
                "codebook": "mcg",
                "mtp_experts": "source",
                "non_routed_quantization": {
                    "quant_method": "deepseek_v4_fp8",
                    "weight_block_size": [32, 32],
                    "activation_scheme": "dynamic",
                },
            },
        }

    @classmethod
    def _write_pack(cls, root: Path, tensors: dict[str, tuple[str, list[int]]]) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        (root / "config.json").write_text(json.dumps(cls._base_config()))
        shard = root / "model-00001-of-00001.safetensors"
        offset = 0
        header: dict[str, dict] = {}
        payload = bytearray()
        sizes = {"I16": 2, "F16": 2, "F32": 4}
        for name, (dtype, shape) in tensors.items():
            count = 1
            for dim in shape:
                count *= dim
            size = count * sizes[dtype]
            header[name] = {
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [offset, offset + size],
            }
            payload.extend(bytes(size))
            offset += size
        raw = json.dumps(header, separators=(",", ":")).encode()
        raw += b" " * ((-len(raw)) % 8)
        shard.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {name: shard.name for name in tensors}})
        )
        return shard

    def test_validator_accepts_uniform_k7(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "pack"
            self._write_pack(
                root,
                {
                    "model.layers.0.ffn.experts.0.w1_trellis": ("I16", [1, 1, 112]),
                    "model.layers.0.ffn.experts.0.w2_trellis": ("I16", [1, 1, 112]),
                },
            )
            report = validate_pack(root, "tp2", 0)
            self.assertTrue(report["deployable_with_current_pinned_loader"], report)
            self.assertEqual(report["trellis_k_histogram"], {"7": 2})

    def test_validator_accepts_physical_k1_supported_by_exllamav3(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "pack"
            self._write_pack(
                root,
                {
                    "model.layers.0.ffn.experts.0.w1_trellis": ("I16", [1, 1, 16]),
                    "model.layers.0.ffn.experts.0.w2_trellis": ("I16", [1, 1, 16]),
                },
            )
            report = validate_pack(root, "tp2", 0)
            self.assertTrue(report["deployable_with_current_pinned_loader"], report)
            self.assertEqual(report["trellis_k_histogram"], {"1": 2})
            self.assertEqual(report["unsupported_k"], [])

    def test_validator_still_rejects_k_outside_kernel_capability(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "pack"
            self._write_pack(
                root,
                {"model.layers.0.ffn.experts.0.w1_trellis": ("I16", [1, 1, 144])},
            )
            report = validate_pack(root, "tp2", 0)
            self.assertFalse(report["deployable_with_current_pinned_loader"])
            self.assertEqual(report["unsupported_k"], [9])

    def test_validator_accepts_mixed_k_within_layer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "pack"
            self._write_pack(
                root,
                {
                    "model.layers.0.ffn.experts.0.w1_trellis": ("I16", [1, 1, 48]),
                    "model.layers.0.ffn.experts.0.w2_trellis": ("I16", [1, 1, 64]),
                    "model.layers.0.ffn.experts.1.w1_trellis": ("I16", [1, 1, 80]),
                },
            )
            report = validate_pack(root, "tp2", 0)
            self.assertTrue(report["deployable_with_current_pinned_loader"], report)
            self.assertTrue(report["mixed_k_layers"])
            self.assertEqual(report["unsupported_k"], [])

    def test_validator_rejects_pointer_stub(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "pack"
            shard = self._write_pack(
                root,
                {"model.layers.0.ffn.experts.0.w1_trellis": ("I16", [1, 1, 32])},
            )
            shard.write_text(
                "version https://git-lfs.github.com/spec/v1\n"
                "oid sha256:" + "0" * 64 + "\nsize 999\n"
            )
            report = validate_pack(root, "tp2", 0)
            self.assertFalse(report["deployable_with_current_pinned_loader"])
            self.assertEqual(report["bad_shards"][0]["status"], "pointer")

    def test_validator_rejects_dtype_shape_byte_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "pack"
            shard = self._write_pack(
                root,
                {"model.layers.0.ffn.experts.0.w1_trellis": ("I16", [1, 1, 32])},
            )
            raw = shard.read_bytes()
            header_len = struct.unpack("<Q", raw[:8])[0]
            header = json.loads(raw[8 : 8 + header_len])
            key = next(iter(header))
            header[key]["data_offsets"] = [0, 32]
            new_header = json.dumps(header, separators=(",", ":")).encode()
            new_header += b" " * ((-len(new_header)) % 8)
            shard.write_bytes(struct.pack("<Q", len(new_header)) + new_header + bytes(64))
            report = validate_pack(root, "tp2", 0)
            self.assertFalse(report["deployable_with_current_pinned_loader"])
            self.assertTrue(report["tensor_size_mismatches"])

    def test_disk_engram_contract_files_and_profile(self) -> None:
        required = [
            "Dockerfile.disk-engram",
            "docs/DISK_ENGRAM.md",
            "profiles/tp4-disk-engram.env",
            "scripts/build_disk_engram_runtime.sh",
            "scripts/start_disk_engram_cluster.sh",
            "scripts/check_disk_engram_cluster.py",
            "scripts/check_oom_guards.sh",
            "scripts/tp4_disk_engram_min_fit.sh",
            "scripts/oom_guard.sh",
            "scripts/watch_oom_guard.sh",
        ]
        missing = [path for path in required if not (ROOT / path).is_file()]
        self.assertEqual(missing, [])
        profile = (ROOT / "profiles/tp4-disk-engram.env").read_text()
        self.assertIn("IMAGE=deepseek-v41-exl3:disk-engram", profile)
        self.assertIn("VLLM_ENGRAM_DISK_BACKED=1", profile)
        self.assertIn("EAGER=1", profile)
        launcher = (ROOT / "scripts/tp4_disk_engram_min_fit.sh").read_text()
        self.assertIn("check_disk_engram_cluster.py 4", launcher)
        self.assertIn("check_oom_guards.sh", launcher)
        serve = (ROOT / "scripts/serve.sh").read_text()
        self.assertIn("VLLM_EXL3_MODEL_DIR", serve)

    def test_no_stale_runtime_pins(self) -> None:
        stale = [
            "8f4517e80416466fa4a3ad2eb28685021" + "d39e95f",
            "21fa627a3933d80de2d1030e732354d8" + "c3cd761e",
            "ee8c2c171bbe0d036a3accb24a76af5" + "a95506748",
            "5666d1b4a55ef2237797eaee60cbb042" + "e933f375",
        ]
        offenders: list[str] = []
        for path in ROOT.rglob("*"):
            if not path.is_file() or ".git" in path.parts:
                continue
            if path.suffix not in {".md", ".py", ".sh", ".json", ".yml", ".yaml", ""}:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if path == Path(__file__):
                continue
            if path.relative_to(ROOT).as_posix() == "docs/evidence/sage330/decode-baseline.json":
                # This one receipt describes an old measured deployment, not a
                # selectable runtime pin. Exempt only its historical identity
                # field; continue checking the rest of this file and all others.
                receipt = json.loads(text)
                self.assertTrue(receipt["scope"].startswith("Historical full-model decode;"))
                self.assertEqual(receipt["source_run_id"], "decode-baseline-r1")
                self.assertEqual(receipt["runtime"].pop("plugin_commit"), stale[0])
                text = json.dumps(receipt)
            for value in stale:
                if value in text:
                    offenders.append(f"{path.relative_to(ROOT)}: stale plugin SHA {value}")
        self.assertEqual(offenders, [], "\n".join(offenders))


if __name__ == "__main__":
    unittest.main(verbosity=2)


class Sm120PreflightContractTests(unittest.TestCase):
    def test_v41_model_import_matches_vllm_registry_package(self) -> None:
        text = (ROOT / "scripts/preflight_sm120_uva.py").read_text()
        self.assertIn("from vllm.models.deepseek_v4_1 import DeepseekV41ForCausalLM", text)
        self.assertNotIn("deepseek_v4_1.nvidia.model import DeepseekV41ForCausalLM", text)
