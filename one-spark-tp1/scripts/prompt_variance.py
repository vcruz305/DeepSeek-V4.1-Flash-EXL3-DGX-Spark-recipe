"""Measure decode speed and draft accounting across prompt subjects.

Decode speed on this pack varies 2x to 3x with the subject of the prompt, at
identical prompt length and settings. This harness is what measured that, and
what separates the two candidate explanations for it.

It reports, per subject:

  ms_per_token     wall-clock decode, excluding the first token
  acceptance       accepted / (accepted + rejected) draft tokens
  tokens_per_round total new tokens / verify rounds, so tokens yielded per forward
  ms_per_round     ms_per_token * tokens_per_round

The discriminator is tokens_per_round, not acceptance. Acceptance stays high on
every subject; what changes is how many draft tokens the confidence gate lets
through per round. See BENCHMARKS.md, "Decode speed varies 2x to 3x with prompt
subject".

Each generation uses a different subject, so no generation reuses the previous
one's Engram rows. Reported per rep rather than best-of: with varied prompts each
rep is a different workload, so a best-of would report the easiest subject rather
than the configuration.

Absolute values shift with page-cache warmth between sessions. The grouping of
subjects is what reproduces, not the exact milliseconds.

Environment:
  MODEL_DIR   pack directory (required)
  EXL3_SRC    ExLlamaV3 checkout to import from (optional)
  REPS        number of generations, default 6 (one per subject)

Run it the way run_tp1.sh sets up the environment, with EXL3_ATS_MMAP=1,
EXL3_ATS_COPY='^(?!mtp\\.)' and EXL3_DSPARK_CONF=0.7, and never alongside a
second model process.
"""
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


def meminfo():
    d = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":", 1)
        if k in ("MemAvailable", "AnonPages", "SUnreclaim"):
            d[k] = round(int(v.split()[0]) / 1048576, 2)
    return d


def emit(**kw):
    print(json.dumps(kw), flush=True)


REPS = int(os.environ.get("REPS", "6"))
pre = meminfo()
emit(phase="preflight", reps=REPS, mem=pre)
if pre.get("MemAvailable", 0) < 115:
    emit(phase="ABORT_PREFLIGHT")
    sys.exit(10)


def watchdog():
    # The pack leaves only a few GiB free once resident. Bail out rather than
    # let the box start reclaiming into swap.
    while True:
        m = meminfo()
        anon = m.get("AnonPages", 0) + m.get("SUnreclaim", 0)
        if m.get("MemAvailable", 999) < 2.5 and anon > 100:
            emit(phase="WATCHDOG_KILL", mem=m)
            os._exit(9)
        time.sleep(3)


threading.Thread(target=watchdog, daemon=True).start()

M = os.environ["MODEL_DIR"]
CTX = 6144
config = Config.from_directory(M)
model = Model.from_config(config)
draft = Model.from_config(config, component="mtp")
mh = draft.caps.get("default_draft_size", 4)
cache = Cache(model, max_num_tokens=CTX, max_history=mh, max_batch_size=1)
draft_cache = Cache(draft, max_num_tokens=CTX)
t0 = time.time()
model.load("cuda:0", progressbar=False, verbose=False)
draft.load("cuda:0", progressbar=False, verbose=False)
tokenizer = Tokenizer.from_config(config)
emit(phase="loaded", seconds=round(time.time() - t0, 1), mem=meminfo())

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

gen = Generator(model=model, cache=cache, tokenizer=tokenizer,
                draft_model=draft, draft_cache=draft_cache, max_chunk_size=2048)

rows = []
for rep in range(REPS):
    para = SUBJECTS[rep % len(SUBJECTS)]
    base = tokenizer.encode(para, add_bos=True).shape[-1]
    ids = tokenizer.encode(para * max(1, 2048 // base), add_bos=True)
    first = last = None
    n = 0
    acc = rej = None
    gen.enqueue(Job(input_ids=ids, max_new_tokens=256, stop_conditions=[],
                    sampler=GreedySampler()))
    try:
        while gen.num_remaining_jobs():
            for r in gen.iterate():
                if r.get("stage") == "error":
                    emit(phase="ERROR", rep=rep)
                    sys.exit(3)
                t = r.get("token_ids")
                if t is not None and t.numel():
                    now = time.perf_counter()
                    # The first token carries prefill; start the clock after it.
                    if first is None:
                        first = now
                    else:
                        n += t.numel()
                    last = now
                # Emitted once per job, only when a drafter is attached.
                if "accepted_draft_tokens" in r:
                    acc = r["accepted_draft_tokens"]
                    rej = r["rejected_draft_tokens"]
    except Exception as e:
        emit(phase="EXCEPTION", rep=rep, type=type(e).__name__, detail=str(e)[:300])
        sys.exit(4)

    span = (last - first) if first and last and last > first else 0.0
    ms = (span / n) * 1000.0 if n and span else None
    total_new = n + 1
    # Every verify round emits one guaranteed token plus whichever drafts it accepted.
    rounds = (total_new - acc) if acc is not None and total_new > acc else None
    tpr = (total_new / rounds) if rounds else None
    row = dict(rep=rep, subject=NAMES[rep % len(NAMES)],
               ms_per_token=round(ms, 3) if ms else None,
               accepted=acc, rejected=rej,
               acceptance=round(acc / (acc + rej), 4) if acc is not None and (acc + rej) else None,
               tokens_per_round=round(tpr, 3) if tpr else None,
               ms_per_round=round(ms * tpr, 3) if ms and tpr else None)
    rows.append(row)
    emit(phase="gen", **row)

emit(phase="SUMMARY", rows=rows)
print("PROMPT_VARIANCE_OK", flush=True)
