"""Paired A/B for a MoE dispatch change, with output token ids as the quality gate.

Every throughput lever on this pack has to clear two bars: it must be faster on the same
prompt in the same session, and it must emit the same tokens. `EXL3_MOE_MIXED_BSZ1` cleared
the first and failed the second, and the confidence-gate table in BENCHMARKS.md is what an
A/B without output hashes looks like. This harness records both.

One process = one arm, because the dispatch knobs (EXL3_MIXEDK_LEGACY, EXL3_GR_INT8) are read
at load. `ab_arms.sh` alternates arms strictly so session drift cannot order the results.

Run `DRAFT=0` first. Plain greedy is self-deterministic on this pack (the control in
`spec_exactness.py` establishes that), and without the drafter a round is one forward of the
main model at one row, so a MoE dispatch change shows up undiluted by acceptance, draft block
length or the two kernels the verify window switches between. The drafted configuration is what
ships, so confirm the win carries over to `DRAFT=1` afterwards; it is the noisier measurement,
not the more truthful one.

Two prompt modes, reported separately and never averaged (AGENTS.md benchmark discipline):

  MODE=repeat   one paragraph, REPS generations, warm after the first. This is the regime the
                ~33 tok/s headline figure comes from, and the one a dispatch change moves most
                visibly.
  MODE=varied   the six subjects from prompt_variance.py, one per generation, no reuse of the
                previous generation's Engram rows. Decode speed spans 2-3x across these, so
                compare per subject, never pooled.

Emits one JSON line per generation and a SUMMARY line. `ab_compare.py` reads those.

Environment:
  MODEL_DIR   pack directory (required)
  EXL3_SRC    ExLlamaV3 checkout to import from (required for the A/B: it is what selects the
              tree, since the two arms are different trees)
  ARM         label recorded in every row (default "unlabeled")
  DRAFT       1 to load the DSpark/MTP drafter (default 1); 0 for the plain-decode arm
  MODE        repeat | varied (default repeat)
  REPS        generations, default 6
  TOKENS      new tokens per generation, default 256

Run it the way run_tp1.sh sets the environment up, and never alongside a second model process.
"""
import hashlib
import json
import os
import sys
import threading
import time

_src = os.environ.get("EXL3_SRC")
if _src:
    sys.path.insert(0, _src)

import torch
from exllamav3 import Config, Model, Tokenizer
from exllamav3.cache.cache import Cache
from exllamav3.generator import Generator, Job
from exllamav3.generator.sampler import GreedySampler

ARM = os.environ.get("ARM", "unlabeled")
MODE = os.environ.get("MODE", "repeat")
REPS = int(os.environ.get("REPS", "6"))
TOKENS = int(os.environ.get("TOKENS", "256"))
CTX = int(os.environ.get("CTX", "6144"))
CHUNK = int(os.environ.get("CHUNK", "2048"))

# The six subjects from prompt_variance.py, verbatim, so the per-subject grouping recorded in
# BENCHMARKS.md carries over.
SUBJECTS = [
    "The history of computing is a history of abstraction. Each generation of engineers "
    "built a layer that hid the one beneath it, and in hiding it made possible a kind of "
    "work that would otherwise have been unthinkable. ",

    "Marine biologists studying hydrothermal vents found ecosystems that derive their "
    "entire energy budget from chemosynthesis rather than sunlight, overturning a long "
    "assumption about where life can persist on this planet. ",

    "The reconstruction of medieval trade routes relies on unglamorous evidence: customs "
    "ledgers, shipwreck cargo manifests, and the distribution of coin hoards across river "
    "valleys and mountain passes. ",

    "In orbital mechanics a transfer between two circular orbits costs the least delta-v "
    "when the burns are made at apoapsis and periapsis, which is why mission planners "
    "care so much about launch windows and phase angles. ",

    "Modern bread fermentation is a negotiation between yeast and lactic acid bacteria, "
    "and the baker controls it almost entirely through temperature, hydration and the "
    "timing of each fold rather than through the recipe itself. ",

    "Legal systems that evolved from common law treat precedent as binding in a way that "
    "civil code jurisdictions do not, which produces strikingly different appellate "
    "cultures and very different judicial opinions. ",
]
NAMES = ["computing", "marine_biology", "medieval_trade", "orbital", "bread", "law"]


def meminfo():
    d = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":", 1)
        if k in ("MemAvailable", "AnonPages", "SUnreclaim"):
            d[k] = round(int(v.split()[0]) / 1048576, 2)
    return d


def emit(**kw):
    print(json.dumps(dict(arm=ARM, mode=MODE, **kw)), flush=True)


def knobs():
    """The environment that distinguishes one arm from another, recorded with the results so a
    row can never be read without the configuration that produced it (AGENTS.md rule 8)."""
    keys = [k for k in os.environ if k.startswith("EXL3_")]
    return {k: os.environ[k] for k in sorted(keys)} | {"EXL3_SRC": _src or ""}


pre = meminfo()
emit(phase="preflight", reps=REPS, tokens=TOKENS, mem=pre, knobs=knobs())
if pre.get("MemAvailable", 0) < 115:
    emit(phase="ABORT_PREFLIGHT")
    sys.exit(10)


def watchdog():
    # The pack leaves only a few GiB free once resident. Bail out rather than let the box
    # start reclaiming into swap.
    while True:
        m = meminfo()
        anon = m.get("AnonPages", 0) + m.get("SUnreclaim", 0)
        if m.get("MemAvailable", 999) < 2.5 and anon > 100:
            emit(phase="WATCHDOG_KILL", mem=m)
            os._exit(9)
        time.sleep(3)


threading.Thread(target=watchdog, daemon=True).start()

M = os.environ["MODEL_DIR"]
DRAFT = os.environ.get("DRAFT", "1") == "1"
config = Config.from_directory(M)
model = Model.from_config(config)
draft = draft_cache = None
max_history = 1
if DRAFT:
    draft = Model.from_config(config, component="mtp")
    max_history = draft.caps.get("default_draft_size", 4)
cache = Cache(model, max_num_tokens=CTX, max_history=max_history, max_batch_size=1)
t0 = time.time()
model.load("cuda:0", progressbar=False, verbose=False)
if DRAFT:
    draft_cache = Cache(draft, max_num_tokens=CTX)
    draft.load("cuda:0", progressbar=False, verbose=False)
tokenizer = Tokenizer.from_config(config)
emit(phase="loaded", seconds=round(time.time() - t0, 1), mem=meminfo(), draft=DRAFT,
     torch_alloc=round(torch.cuda.memory_allocated() / 2**30, 2))

gen = Generator(model=model, cache=cache, tokenizer=tokenizer,
                draft_model=draft, draft_cache=draft_cache, max_chunk_size=CHUNK)

rows = []
for rep in range(REPS):
    idx = 0 if MODE == "repeat" else rep % len(SUBJECTS)
    para = SUBJECTS[idx]
    base = tokenizer.encode(para, add_bos=True).shape[-1]
    ids = tokenizer.encode(para * max(1, 2048 // base), add_bos=True)
    first = last = None
    n = 0
    acc = rej = None
    out_ids = []
    gen.enqueue(Job(input_ids=ids, max_new_tokens=TOKENS, stop_conditions=[],
                    sampler=GreedySampler()))
    try:
        while gen.num_remaining_jobs():
            for r in gen.iterate():
                if r.get("stage") == "error":
                    emit(phase="ERROR", rep=rep, detail=str(r)[:300])
                    sys.exit(3)
                t = r.get("token_ids")
                if t is not None and t.numel():
                    out_ids += t.flatten().tolist()
                    now = time.perf_counter()
                    # The first token carries prefill; start the clock after it.
                    if first is None:
                        first = now
                    else:
                        n += t.numel()
                    last = now
                if "accepted_draft_tokens" in r:
                    acc = r["accepted_draft_tokens"]
                    rej = r["rejected_draft_tokens"]
    except Exception as e:
        emit(phase="EXCEPTION", rep=rep, type=type(e).__name__, detail=str(e)[:300])
        sys.exit(4)

    span = (last - first) if first and last and last > first else 0.0
    ms = (span / n) * 1000.0 if n and span else None
    total_new = n + 1
    rounds = (total_new - acc) if acc is not None and total_new > acc else None
    tpr = (total_new / rounds) if rounds else None
    row = dict(rep=rep, subject=NAMES[idx],
               ms_per_token=round(ms, 3) if ms else None,
               tok_s=round(1000.0 / ms, 2) if ms else None,
               accepted=acc, rejected=rej,
               acceptance=round(acc / (acc + rej), 4) if acc is not None and (acc + rej) else None,
               tokens_per_round=round(tpr, 3) if tpr else None,
               ms_per_round=round(ms * tpr, 3) if ms and tpr else None,
               n_out=len(out_ids),
               # The quality gate. Same prompt, same arm-independent model: the ids must match
               # the baseline arm's ids exactly, or the lever changed the output.
               ids_sha=hashlib.sha256(
                   json.dumps(out_ids).encode()).hexdigest()[:16],
               first_ids=out_ids[:8])
    rows.append(row)
    emit(phase="gen", **row)

emit(phase="SUMMARY", rows=rows, knobs=knobs())
print("AB_DISPATCH_OK", flush=True)
