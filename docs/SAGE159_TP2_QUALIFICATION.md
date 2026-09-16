# SAGE 1.59 two-Spark qualification

This is a bounded hardware receipt for
`vcruz305/DSV4.1-Flash-SAGE-EXL3-1.59bpw` on two DGX Sparks. The exact model
revision reached an OpenAI-compatible endpoint with a configured 131,072-token
window and passed exact-recall probes through 126,067 server-counted prompt
tokens.

It does **not** replace the canonical TP2/3.30 contract in `runtime.lock.json`.
The run used a separate streaming runtime derivation and current `vllm-exl3`
`main`. Treat this as hardware qualification evidence for a future lock
promotion, not as proof that the repository's default image/profile has been
qualified.

## Immutable inputs

| Input | Qualified value |
|---|---|
| Model | `vcruz305/DSV4.1-Flash-SAGE-EXL3-1.59bpw` |
| Model revision | `0c29707b4fbcdd2e7ce61bc336dd355d6a9e1994` |
| Tensor shards | 17; 330,385,931,208 bytes |
| Snapshot | 23 files; 330,406,735,253 bytes |
| `vllm-exl3` | `a78949ac82e071f2c3ca7477910d5d9641df6d80` |
| vLLM | `0.1.dev20904+g179dd0fa9` |
| `vllm-exl3` package | `0.4.2` |
| ExLlamaV3 | `1.4.9` |
| PyTorch | `2.13.0+cu130` |
| Base image | `dsv41-flash-exl3-sm121:streaming-6d6d9fc` |
| Base image ID | `sha256:19034008ed831ea886c548bd637fdb74169ba0fed1909f26109961fb8c37580c` |

The base image is an external reference runtime and is not produced by
`Dockerfile.spark`; the local tag and image ID identify the tested bytes but are
not enough to reconstruct that image from this repository. The candidate layer
installed the exact plugin commit above and passed `runtime_diagnostics()`
checks for tensor-level mixed K, physical-K fused dispatch, and arena prescan
before the hardware run.

## Serving shape

- two nodes, TP2 + EP2, 192 whole routed experts per rank;
- disk-backed Engram on each node's local NVMe;
- eager execution, CUDA graphs off, DSpark off, prefix caching off;
- no speculative decoding configuration requested;
- ExLlamaV3 MoE backend with exact per-expert trellis arenas;
- one sequence, 512 batched tokens, 4 GiB KV cache per rank, FP8 KV;
- `--max-model-len 131072`, block size 64, GPU memory utilization 0.75.

The complete compact setting receipt is
[`qualifications/sage159-tp2-131k.env`](../qualifications/sage159-tp2-131k.env).
It is deliberately outside `profiles/` because the repository's default
launcher and runtime lock describe a different runtime derivation.

## Results

| Gate | Result |
|---|---:|
| True cold orchestration to `/v1/models` | 628.185 s |
| Deterministic smoke (`17 × 19`) | exact `323` |
| Prose median TTFT, 3 warmed runs | 0.357 s |
| Prose median decode, 128 tokens | 11.05 tok/s |
| Counting median TTFT, 3 warmed runs | 0.294 s |
| Counting median decode, 128 tokens | 10.38 tok/s |
| 32K target | 30,835 prompt tokens; 3/3 keys; 49.893 s |
| 64K target | 63,603 prompt tokens; 3/3 keys; 93.175 s |
| 128K target | 126,067 prompt tokens; 3/3 keys; 184.656 s |

Both ranks remained running with zero restarts. No traceback, exception, OOM,
NCCL-error, or worker-died signature was found in either container log after the
probe sequence. Swap remained zero. Available host memory after the probes was
about 43.7 GiB on the head and 44.3 GiB on the worker.

The exact machine-readable receipt is
[`qualifications/sage159-tp2-a78949a.json`](../qualifications/sage159-tp2-a78949a.json).

## Checkpoint metadata boundary

The published snapshot declares only `{"quant_method":"exl3"}`. It does not
carry the mixed-format delegation fields required by the loader. The checkpoint
files were not edited. A 96 KiB symlink serving view supplied a runtime-only
`quantization_config` override after verifying the immutable model revision,
shard count, total bytes, and safetensors header identity.

The checked-in attestation records that identity. It is a header/layout receipt,
not a full tensor-payload checksum. Any config, header, tensor-count, shard-byte,
K-histogram, or codebook-marker change invalidates it.

The physical trellis histogram spans K1–K6. The recipe previously compared
physical headers with `accepted_exl3_config_k` (K2–K8), even though the locked
ExLlamaV3 execution capability is K1–K8. That produced a false K1 rejection for
this served pack. The validator now uses `exllamav3_moe_kernel_k` for physical
trellis widths while leaving the top-level config capability unchanged. A K9
regression fixture remains fail-closed.

For this receipt, the corrected validator was evaluated against an explicitly
supplied external 17-shard contract and returned no errors or warnings for the
serving view. That external contract is not selectable through the canonical
recipe. The default generic TP2 validator still reports the expected 31-shard
canonical-contract warning, and the strict TP2 wrapper rejects the mismatch.

## Claim boundary

This run verifies the 1.59 bpw artifact at the identity and settings above. It
does not verify:

- the canonical TP2/3.30 artifact;
- the repository's locked `Dockerfile.spark` image;
- 262K context (the configured window was 131,072 and the largest observed
  prompt was 126,067 tokens);
- multi-sequence concurrency, CUDA graphs, prefix caching, or DSpark;
- long-duration soak stability or output-quality equivalence to another
  quantization.

Promoting this path into the default recipe should be a separate change that
rebuilds the repository image, advances every affected pin explicitly, reruns
distributed preflight, and repeats the hardware gates.
