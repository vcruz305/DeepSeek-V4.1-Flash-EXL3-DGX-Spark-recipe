"""Where does a verify round actually spend its time? Per-module CUDA-event timing.

`BENCHMARKS.md` establishes that the decode is GPU-bound and that a round costs 130-175 ms.
It does not say which modules that is. `torch.profiler` cannot answer it here: the methodology
note records CUPTI overhead swamping a workload that issues 620+ kernel launches per token and
reporting impossible occupancy. CUDA events can, because they add two launches per wrapped
module and are read once per generation rather than per kernel.

The split decides which candidate in OPTIMIZATION_CANDIDATES.md is worth building:

  routed MoE dominant   -> the fused mixed-K dispatch (candidates 1, 2, 4, 5)
  attention dominant    -> the 38 DSV41Attention layers with no CUDA-graph path
  head dominant         -> the pruned drafter head (candidate 6); the fp16 head is 1.23 GiB and
                           is read once for the draft block and once for the verify

Modules are wrapped per instance, so the main model and the drafter are reported separately
even though they share classes.

Environment:
  MODEL_DIR   pack directory (required)
  EXL3_SRC    ExLlamaV3 checkout to import from (optional)
  REPS        generations, default 3
  TOKENS      new tokens per generation, default 128

Timing includes the events' own launches, so read the shares, not the absolute totals, and
compare a candidate against this baseline with the same harness.
"""
import json
import os
import sys
import time
from collections import defaultdict

_src = os.environ.get("EXL3_SRC")
if _src:
    sys.path.insert(0, _src)

import torch
from exllamav3 import Config, Model, Tokenizer
from exllamav3.cache.cache import Cache
from exllamav3.generator import Generator, Job
from exllamav3.generator.sampler import GreedySampler

REPS = int(os.environ.get("REPS", "3"))
TOKENS = int(os.environ.get("TOKENS", "128"))
CTX = int(os.environ.get("CTX", "6144"))
CHUNK = int(os.environ.get("CHUNK", "2048"))

totals = defaultdict(float)
counts = defaultdict(int)
pending = []        # (tag, start_event, end_event), drained at each generation boundary


def drain():
    """Events are only readable once the work behind them has run, and a pair cannot be reused
    until then, so the whole batch is read at a generation boundary rather than per forward."""
    if not pending:
        return
    torch.cuda.synchronize()
    for tag, a, b in pending:
        totals[tag] += a.elapsed_time(b)
        counts[tag] += 1
    pending.clear()


def wrap(module, tag):
    inner = module.forward

    def timed(*args, **kwargs):
        a = torch.cuda.Event(enable_timing = True)
        b = torch.cuda.Event(enable_timing = True)
        a.record()
        out = inner(*args, **kwargs)
        b.record()
        pending.append((tag, a, b))
        return out

    module.forward = timed


def wrap_model(model, prefix):
    """Wrap the leaves that carry the work: attention, the routed MLP and its shared expert,
    Engram, the hyper-connection mixers and the head. Blocks themselves are left alone, so the
    shares sum to the model's forward rather than double-counting it."""
    for m in model.modules:
        cls = type(m).__name__
        for attr in ("attn", "mlp", "attn_hc", "mlp_hc"):
            child = getattr(m, attr, None)
            if child is not None and hasattr(child, "forward"):
                wrap(child, f"{prefix}.{type(child).__name__}")
                if attr == "mlp":
                    sh = getattr(child, "shared_experts", None)
                    if sh is not None and hasattr(sh, "forward"):
                        wrap(sh, f"{prefix}.shared_experts")
        if not any(hasattr(m, a) for a in ("attn", "mlp")):
            wrap(m, f"{prefix}.{cls}")


M = os.environ["MODEL_DIR"]
config = Config.from_directory(M)
model = Model.from_config(config)
draft = Model.from_config(config, component = "mtp")
mh = draft.caps.get("default_draft_size", 4)
cache = Cache(model, max_num_tokens = CTX, max_history = mh, max_batch_size = 1)
draft_cache = Cache(draft, max_num_tokens = CTX)
t0 = time.time()
model.load("cuda:0", progressbar = False, verbose = False)
draft.load("cuda:0", progressbar = False, verbose = False)
tokenizer = Tokenizer.from_config(config)
print(json.dumps({"phase": "loaded", "seconds": round(time.time() - t0, 1)}), flush = True)

wrap_model(model, "main")
wrap_model(draft, "draft")

gen = Generator(model = model, cache = cache, tokenizer = tokenizer, draft_model = draft,
                draft_cache = draft_cache, max_chunk_size = CHUNK)

para = ("The history of computing is a history of abstraction. Each generation of engineers "
        "built a layer that hid the one beneath it, and in hiding it made possible a kind of "
        "work that would otherwise have been unthinkable. ")
base = tokenizer.encode(para, add_bos = True).shape[-1]
ids = tokenizer.encode(para * max(1, 2048 // base), add_bos = True)

for rep in range(REPS):
    if rep == 1:
        # Drop the first generation: it carries prefill and a cold page cache.
        totals.clear()
        counts.clear()
    first = last = None
    n = 0
    gen.enqueue(Job(input_ids = ids, max_new_tokens = TOKENS, stop_conditions = [],
                    sampler = GreedySampler()))
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            t = r.get("token_ids")
            if t is not None and t.numel():
                now = time.perf_counter()
                if first is None:
                    first = now
                else:
                    n += t.numel()
                last = now
    drain()
    ms = (last - first) / n * 1000.0 if n and last and first else 0.0
    print(json.dumps({"phase": "gen", "rep": rep, "ms_per_token": round(ms, 3)}), flush = True)

wall = sum(totals.values())
print(f"\n{'module':<34}{'ms total':>12}{'calls':>9}{'ms/call':>10}{'share':>8}")
for tag, ms in sorted(totals.items(), key = lambda kv: -kv[1]):
    print(f"{tag:<34}{ms:>12.1f}{counts[tag]:>9}{ms / counts[tag]:>10.3f}{ms / wall * 100:>7.1f}%")
print(f"{'(sum of wrapped modules)':<34}{wall:>12.1f}")
print("MODULE_TIMING_OK", flush = True)
