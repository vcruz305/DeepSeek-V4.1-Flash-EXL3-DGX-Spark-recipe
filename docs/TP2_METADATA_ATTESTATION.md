# TP2 runtime-metadata attestation

The canonical TP2 snapshot predates the explicit mixed-format metadata contract used by `vllm-exl3`.

Locked artifact:

- repo: `vcruz305/DSV4.1-Flash-SAGE-EXL3-3.30bpw`
- revision: `e831e9e4d6bfeafa6d630848296417b1393404a3`
- canonical `config.json` SHA256: `25882944c203b1970a9187dc47334ebafbb7e95ffda888fd86c8181af9e15867`
- canonical config declares `quantization_config.quant_method=exl3` but does not carry `non_routed_quantization` / `mtp_experts` fields.

Do **not** edit the model's `config.json` to make validation pass.

## Why runtime metadata is required

The checkpoint is mixed-format:

- routed main experts are EXL3;
- protected non-routed DeepSeek V4.1 weights retain source FP8 formatting;
- DSpark/MTP experts retain source expert formatting;
- Engram remains native and is handled separately by the disk-backed path.

DeepSeek V4.1's source model declares dynamic FP8 linear quantization with a `[32, 32]` weight block. vLLM's V4.1-specific `DeepseekV4FP8Config` exposes the quantization name `deepseek_v4_fp8`; its 32x32 linear path uses the V4.1 blocked ModelOpt loader, while source FP4 routed experts dispatch through MXFP4.

References:

- DeepSeek V4.1 source model: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash
- vLLM V4.1 quant config: https://github.com/vllm-project/vllm/blob/410f6da5c4bb62010728502035bee1b5f0eab2ac/vllm/models/deepseek_v4_1/quant_config.py
- vLLM `--hf-overrides`: https://docs.vllm.ai/en/latest/cli/serve/#--hf-overrides

The required runtime overlay is therefore:

```json
{
  "quantization_config": {
    "quant_method": "exl3",
    "non_routed_quantization": {
      "quant_method": "deepseek_v4_fp8",
      "activation_scheme": "dynamic",
      "weight_block_size": [32, 32]
    },
    "mtp_experts": "source",
    "mtp_experts_start_layer": 40
  }
}
```

This is supplied to vLLM through `--hf-overrides`; it is never written into the checkpoint directory.

## Fail-closed header-level identity binding

The override is accepted only for `--strict-locked-snapshot` validation and only when all values in:

`attestations/tp2-e831e9e4-metadata.json`

match the observed snapshot, including:

- config SHA256;
- 31 individual **raw safetensors header** SHA256 values;
- index tensor count;
- total shard file bytes;
- EXL3 K histogram;
- codebook-marker counts;
- locked model repo + immutable HF revision in the recipe contract.

This is a **metadata/header attestation**, not a 446 GB payload checksum. It proves the runtime-format declarations are bound to the observed tensor names/dtypes/shapes/offsets in the safetensors headers. It does not claim that every tensor payload byte has been rehashed.

Any attestation mismatch keeps the original metadata errors and produces `DEPLOYABLE_CURRENT_LOADER=NO`.

Structural failures are never attestable. Missing/unindexed tensors, bad offsets, byte-size inconsistencies, unsupported K, malformed shards, or disk-reserve failure remain hard errors even when the metadata attestation matches.

## Remote provenance check

Before a hardware load, compare the checked-in header receipt directly with the immutable HF revision using Range requests only:

```bash
python3 scripts/probe_tp2_attestation.py
```

Require:

```text
REMOTE_METADATA_ATTESTATION=PASS
```

This downloads config/index JSON and only the header byte ranges of the 31 safetensors files. It refuses to use tensor payload downloads for this provenance check.

## Local qualification commands

Use the TP2-specific validator for the canonical local snapshot:

```bash
python3 scripts/check_tp2_pack.py /path/to/DSV4.1-Flash-SAGE-EXL3-3.30bpw \
  --reserve-gib 32 \
  --strict-locked-snapshot
```

Expected metadata lines for the attested snapshot:

```text
Runtime metadata source: locked_snapshot_attestation
Metadata attestation match: True
DEPLOYABLE_CURRENT_LOADER=YES
```

To inspect the exact runtime-only override that will be supplied to vLLM:

```bash
python3 scripts/check_tp2_pack.py /path/to/DSV4.1-Flash-SAGE-EXL3-3.30bpw \
  --reserve-gib 32 \
  --strict-locked-snapshot \
  --print-hf-overrides
```

`serve_tp2.sh` runs the same strict validation inside the serving container and exports the returned JSON to `serve.sh`, which passes it to vLLM as one quoted `--hf-overrides` argument.

## Scope

Passing this gate means the exact observed header/layout identity has a documented runtime metadata contract compatible with the pinned loader. It does **not** prove full payload-byte integrity, two-Spark capacity, disk-Engram performance, output correctness, long-context viability, or throughput. Those remain separate qualification gates.
