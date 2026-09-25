# Compatibility

| Path | V4.1 graph | Routed experts | Engram | Status |
|---|---|---|---|---|
| **4× Spark TP4 + EP4** (`:tp4` image) | vLLM | EXL3 per-expert mixed K3–K8, multi-K / padded MoE kernels | disk-backed | **Serving** |
| 2× Spark TP2 + EP2 | vLLM | EXL3 per-expert mixed K2–K8 | disk-backed | Loader ready; not benchmarked |
| 1× Spark TP1 | native ExLlamaV3 (fork) | EXL3 per-expert mixed K1–K6 | ATS mmap | **Serving** |
| Resident Engram on Spark | vLLM | any | resident | Does not fit (unified-memory cliff) |
| Upstream ExLlamaV3 alone | not registered | kernels only | — | Cannot serve V4.1 |

`vllm-exl3` does not replace the DeepSeek-V4.1 graph. vLLM owns attention, compressed sparse
attention, Engram, routing and DSpark. The plugin supplies routed-expert storage and execution.

## Mixed K

`vllm-exl3` keeps the exact trellis shape of every expert projection, so K can differ between
experts and between `w1/w2/w3` inside one expert:

```text
expert A: w1=K3 w3=K4 w2=K5
expert B: w1=K8 w3=K8 w2=K7
```

On the `:tp4` image, the multi-K fused kernel (`p2b_fused_moe_mk`) handles heterogeneous layers in
one cooperative launch. The fixed-shape padded kernel makes that launch capturable in CUDA graphs.
The base image falls back to a per-expert `LinearEXL3` loop, which is correct but eager-only.

## Quality

The released source experts are already low precision, and EXL3 re-encoding cannot recover what
the source lost. To measure quality, compare **additional** transcode loss against the original
`deepseek-ai/DeepSeek-V4.1-Flash` checkpoint on fixed prompts with deterministic settings.

## TP-MoE (EP1)

TP4's alternative to EP4 routes MoE through tensor parallelism instead of
expert parallelism (`MOE_PARALLEL_MODE=tp`, see
[`four-spark-tp4/README.md`](../four-spark-tp4/README.md#tp-moe-ep1-current-best-serving-configs)).
It requires three `vllm-exl3` PRs still pending upstream:

- [PR #36](https://github.com/vcruz305/vllm-exl3/pull/36): Hadamard-aligned uneven TP MoE (`VLLM_EXL3_MOE_TP_ALIGN=128`)
- [PR #37](https://github.com/vcruz305/vllm-exl3/pull/37): padded-MoE loops bounded by `n_valid`
- [PR #39](https://github.com/vcruz305/vllm-exl3/pull/39): padded-MoE expert-grouped stage1/5 (decode each trellis tile once per expert, not once per slot; required for the recommended k=3 default)

## Upstream

- ExLlamaV3: https://github.com/turboderp-org/exllamav3
- vLLM: https://github.com/vllm-project/vllm
- `vllm-exl3`: https://github.com/vcruz305/vllm-exl3
- DeepSeek-V4.1-Flash: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash
