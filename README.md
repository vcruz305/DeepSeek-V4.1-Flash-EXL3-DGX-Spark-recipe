# DeepSeek-V4.1-Flash EXL3 on DGX Spark

Serving recipes for **DeepSeek-V4.1-Flash** in EXL3 on NVIDIA DGX Spark (GB10). There is one
folder per cluster size:

| Sparks | Folder | Model | Engine | Status |
|---|---|---|---|---|
| **4** | [`four-spark-tp4/`](four-spark-tp4/README.md) | [`DSV4.1-Flash-EXL3-4.75bpw`](https://huggingface.co/vcruz305/DSV4.1-Flash-EXL3-4.75bpw) | vLLM + `vllm-exl3` | **Serving** |
| **2** | [`two-spark-tp2/`](two-spark-tp2/README.md) | [`DSV4.1-Flash-SAGE-EXL3-3.30bpw`](https://huggingface.co/vcruz305/DSV4.1-Flash-SAGE-EXL3-3.30bpw) | vLLM + `vllm-exl3` | Loader ready; not benchmarked |
| **1** | [`one-spark-tp1/`](one-spark-tp1/README.md) | [`DSV4.1-Flash-SAGE-EXL3-1.59bpw`](https://huggingface.co/vcruz305/DSV4.1-Flash-SAGE-EXL3-1.59bpw) | native ExLlamaV3 | **Serving** |

## Performance

### Four Sparks (TP4 + EP4)

Verified by **@fattchris** with `llm-inference-bench`. Every figure is a 30-second sustained cell:

| Measurement | Result |
|---|---:|
| Single-stream decode, prose | **30.2 tok/s** |
| Single-stream decode, code | **33.4 tok/s** |
| Aggregate decode, c = 1 / 2 / 4 / 8 | 27.8 / 43.9 / 57.7 / 61.5 tok/s |
| KV budget | 3.97M tokens |
| Max context | **1,048,576 tokens** (full 1M) |

Settings: DSpark k=2 with MXFP4 draft experts, decode-only CUDA graphs and disk-backed Engram.
Profile: [`profiles/tp4.env`](profiles/tp4.env). Runtime: `vllm-exl3` `814d4fe` plus the multi-K /
padded MoE kernels from `4c95648`, ExLlamaV3 `be57335`, CUDA 13.0.1, `sm_121a`. The per-domain
matrix and full runtime identity are in [`four-spark-tp4/`](four-spark-tp4/README.md).

### One Spark (TP1)

Native ExLlamaV3 from the `vcruz305/exllamav3` fork, with the EXL3 attention/MTP overlay:

Decode figures lead with the DSpark drafter, which is the configuration this recipe ships.

| Measurement | Result |
|---|---:|
| **Decode, drafter, warm repeat prompt, max** | **33.63 – 33.66 tok/s** |
| Decode, drafter, warm repeat prompt, default prefetch | 32.55 – 33.37 tok/s |
| Decode, drafter, fresh prompt | **17.53 median**, 19.82 mean |
| Decode, no drafter, fresh prompt | 15.13 – 15.22 tok/s |
| Draft acceptance | 0.981 warm repeat, 0.889 fresh prompt |
| Prefill, chunk 4096 | 254 – 261 tok/s |
| Load time, resident size | 37.5 s, ~107 GiB |

**Warm repeat and fresh prompt are different workloads, and the two groups are never comparable.**
The warm rows repeat one prompt in a single process, a best-case prefix-cache hit that isolates the
cost of decode. The fresh rows serve a different prompt every generation. Quote the fresh-prompt
figures for anything resembling real traffic, and the warm ones only as a ceiling.

The max row is the explicit Engram variant `EXL3_ENGRAM_PREFETCH=0` (`AGENTS.md` rule 9), measured
in strict alternation against the default: 33.63 and 33.66 with it off against 32.55 and 32.67 with
it on, a 3.2% gain with no overlap between the two sets. Every other figure on this page was
produced with the prefetch at its default.

Context is nearly free: 131,072 measures the same warm decode as 6,144, and 262,144 costs about
1 tok/s, and the run-time settings are in [`one-spark-tp1/README.md`](one-spark-tp1/README.md).

Methodology, runtime identity and negative results:
[`one-spark-tp1/BENCHMARKS.md`](one-spark-tp1/BENCHMARKS.md) – which also carries the rest of the
fresh-prompt session's row, including its repeat-prompt column (20.11 – 24.67 tok/s at 0.889
acceptance).

## Quick start: four Sparks

On every Spark:

```bash
git clone https://github.com/vcruz305/DeepSeek-V4.1-Flash-EXL3-DGX-Spark-recipe.git
cd DeepSeek-V4.1-Flash-EXL3-DGX-Spark-recipe

bash scripts/build_tp4_runtime.sh
bash scripts/materialize_model.sh 4 /large/models/DSV4.1-Flash-EXL3-4.75bpw

set -a; . profiles/tp4.env; set +a
export MODEL_DIR=/large/models
HEAD_IP=<head-ip> NODE_IP=<this-ip> bash scripts/start_disk_engram_cluster.sh head   # `worker` on the other three
```

On the head only:

```bash
bash scripts/preflight.sh 4
bash scripts/serve_tp4.sh
bash scripts/smoke_test.sh
```

This serves an OpenAI-compatible API on port 8000 under the model name `deepseek-v41-exl3`.
Details and tuning are in [`four-spark-tp4/README.md`](four-spark-tp4/README.md).

For one Spark, follow [`one-spark-tp1/README.md`](one-spark-tp1/README.md). It runs a different
engine: native ExLlamaV3, with no Docker and no Ray.

## How the pieces fit

- **vLLM** owns the DeepSeek-V4.1 graph: attention, compressed sparse attention, Engram, routing
  and the DSpark drafter.
- **[`vllm-exl3`](https://github.com/vcruz305/vllm-exl3)** plugs EXL3 routed-expert storage and
  kernels into vLLM. It supports per-expert mixed K (K2–K8), including different K across
  `w1/w2/w3` of one expert.
- **ExLlamaV3** supplies the EXL3 kernels. On one Spark it also serves the whole model.
- **Engram** is disk-backed on Spark. The embedding table stays on local NVMe and each step stages
  only the rows it needs. A resident Engram table does not fit in the 128 GB unified pool.

| Layout | Local experts | Expert matrix |
|---|---:|---:|
| TP4 + EP4 | 96 | 5120 × 2304 |
| TP2 + EP2 | 192 | 5120 × 2304 |
| pure MoE TP2 (A/B) | 384 | 5120 × 1152 |
| TP1 | 384 | 5120 × 2304 |

## Runtime

[`runtime.lock.json`](runtime.lock.json) pins the vLLM base image, `vllm-exl3`, ExLlamaV3, the
TP4 MoE-kernel ref, the CUDA arch and the model revisions. The build scripts read it, so you do
not pass any of these by hand.

| Image | Built by | Adds |
|---|---|---|
| `deepseek-v41-exl3:spark` | `scripts/build_runtime.sh` | vLLM DeepSeek-V4.1 image + `vllm-exl3` + ExLlamaV3 |
| `deepseek-v41-exl3:disk-engram` | `scripts/build_disk_engram_runtime.sh` | disk-backed Engram overlay |
| `deepseek-v41-exl3:tp4` | `scripts/build_tp4_runtime.sh` | multi-K/padded MoE kernels, DSpark and Engram fixes, SM121 sparse-attention fix |

## Repository layout

```text
four-spark-tp4/     TP4 guide and measured numbers
two-spark-tp2/      TP2 guide, metadata override, earlier offload findings
one-spark-tp1/      native ExLlamaV3 guide, benchmarks, TabbyAPI config, launcher
profiles/           tp4.env (measured), tp2.env
configs/            serve config captured from the measured TP4 deployment
scripts/            build, download, cluster start/stop, preflight, serve, smoke test, OOM guard
tools/engram/       Engram fixes: CUDA-graph hoist, replay keying, cold I/O (baked into :tp4)
tools/dspark/       DSpark drafter fixes and MoE-kernel patches (baked into :tp4)
overlays/           vLLM source overlays: disk Engram, SM121 sparse attention, DSpark-in-checkpoint, H2D prefetch
docs/               disk Engram, compatibility, troubleshooting, SGLang notes
tests/              CPU-only contract tests (run in CI)
```

## Credits

- **@fattchris**: TP4 serving, including the Engram replay/graph/cold-I/O fixes, the SM121
  sparse-attention fix, the DSpark config and the deployed profile. Also the `vllm-exl3` multi-K,
  codebook, padded-MoE and draft-loader PRs and the GB10 build fixes.
- **@Blackwellboy**: per-expert mixed-K loading in `vllm-exl3` and the disk-backed Engram path.
- **@joeynyc**: the SAGE 3.30 two-Spark work.
- **@tiggerite**: the single-layer ExLlamaV3 / `vllm-exl3` Docker build.
- **turboderp**: [ExLlamaV3](https://github.com/turboderp-org/exllamav3) and the EXL3 format.
- **DeepSeek**: [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash).

Third-party code and licenses: [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## License

Recipe code is **AGPL-3.0-only**. Model weights, vLLM, ExLlamaV3, CUDA components and container
layers keep their own licenses.
