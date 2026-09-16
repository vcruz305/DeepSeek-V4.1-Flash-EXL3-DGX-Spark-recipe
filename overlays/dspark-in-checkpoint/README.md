# DSpark in-checkpoint draft (mtp.{0,1,2}) wiring — SAGE 3.30bpw TP2/EP2

DeepSeek-V4.1 checkpoints ship DSpark as three in-checkpoint draft stages
(`mtp.0`–`mtp.2`, ~7.4 GiB total, experts in source MXFP4). Serving them with
speculative decoding needs four runtime overlays on top of the image. Each
file below is bind-mounted over site-packages; `apply.sh` copies the files in
and keeps `exl3_reader_lock.json` hashes in sync.

| File | What it changes |
|---|---|
| `vllm/model_executor/model_loader/exl3_attested_reader.py` | Draft-only filter: while the DSpark draft loads, only `mtp.*` tensors are materialized (the target's tensors must not be read a second time under the post-load RAM ceiling). Also carries the sealed-contract / same-dir revision handling used by the two-Spark job. |
| `vllm/model_executor/model_loader/weight_utils.py` | The lazy safetensors iterator honors the same draft-only filter before materialization. |
| `vllm/model_executor/models/utils.py` | `get_draft_quant_config()`: the draft reuses the target config, but its own `config.json` carries only the bare exl3 declaration. Copy the attested runtime-override fields (`non_routed_quantization`, `mtp_experts`, `mtp_experts_start_layer`, `non_routed_dtype_policy`) from the target's quant config. Without this the draft attention builds unquantized (`weight_scale_inv` KeyError) and the MTP experts get re-quantized to EXL3 instead of staying in their source format. |
| `vllm/models/deepseek_v4_1/nvidia/dspark.py` | Skip `gate.bias_vl`, which the draft loader has no home for (mirrors the V4 Vision recipe). |

## Launch flags that go with it

- `--hf-overrides`: `mtp_experts=source` **plus `mtp_experts_start_layer=40`**
  (the draft stages are `model.layers.40–42` on a 40-layer V4.1 backbone).
- `--speculative-config '{"method":"dspark","num_speculative_tokens":5,...}'`
  (k must divide the pack's `dspark_block_size=5`).
- `DSV41_EXPERIMENT_READER_FLOOR_GIB=2` with a live `experiment-guard`, and
  `VLLM_EXL3_H2D_PREFETCH=4` on the UMA pair.

## Companion plugin change

The plugin tree (`vllm-exl3`) must carry the tensor-metadata provider hooks and
the draft/MTP constructor-plan tolerance (`tensor_metadata.py`,
`tensor_mixed_k.py`, and the `exl3.py` updates); see the companion plugin
commit for the matching revision.

## Status

First verification pass against `vcruz305/DSV4.1-Flash-SAGE-EXL3-3.30bpw`
revision `e831e9e4` on the two-Spark EP2 pair: target loads 40/40 layer
arenas; the draft path is exercised through construction and weight loading.
End-to-end decode qualification of the speculative path is in progress — treat
this overlay as the working configuration, not a finished benchmark receipt.
