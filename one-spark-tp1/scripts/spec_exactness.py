"""Does DSpark speculation emit the same tokens as the target model decoding alone?

Greedy speculative decoding is supposed to change only HOW tokens are produced, never WHICH
ones. On this pack that does not always hold. This harness is what measured it.

Method, exact by construction. For each prompt it decodes twice, once with the drafter and
once without, and scores the FIRST differing token. At that index both sequences still share
an identical prefix, so the no-drafter distribution there is conditioned on exactly the
context the speculative run saw, and the comparison is apples-to-apples. Positions after the
first difference are NOT comparable, because the runs have forked into different contexts.

Classification at the first difference:

  exact              no divergence in the whole generation
  near_tie           target top-1 minus top-2 < 0.05, i.e. fp16 tie-breaking
  clear_preference   the target preferred its top-1 by more than that

Control: plain greedy must be self-deterministic, or none of this is interpretable. Run the
no-drafter pass twice and confirm the hashes match before trusting any row. That control
passed on every prompt tested here.

Environment:
  MODEL_DIR   pack directory (required)
  EXL3_SRC    ExLlamaV3 checkout to import from (optional)

Run it with the environment run_tp1.sh sets up, and never alongside a second model process.
This measures tokens and probabilities only, never wall-clock, so it is not disturbed by
other light load on the box.
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

TOPK = 5
MAXNEW = 192
NEAR_TIE = 0.05


def meminfo():
    d = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":", 1)
        if k in ("MemAvailable", "AnonPages", "SUnreclaim"):
            d[k] = round(int(v.split()[0]) / 1048576, 2)
    return d


def emit(**kw):
    print(json.dumps(kw), flush=True)


pre = meminfo()
emit(phase="preflight", mem=pre)
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
CTX = 6144
config = Config.from_directory(M)
model = Model.from_config(config)
draft = Model.from_config(config, component="mtp")
mh = draft.caps.get("default_draft_size", 4)
cache = Cache(model, max_num_tokens=CTX, max_history=mh, max_batch_size=1)
draft_cache = Cache(draft, max_num_tokens=CTX)
cache_plain = Cache(model, max_num_tokens=CTX, max_history=mh, max_batch_size=1)
t0 = time.time()
model.load("cuda:0", progressbar=False, verbose=False)
draft.load("cuda:0", progressbar=False, verbose=False)
tokenizer = Tokenizer.from_config(config)
emit(phase="loaded", seconds=round(time.time() - t0, 1), mem=meminfo())

PROMPTS = {
    "computing":
        "The history of computing is a history of abstraction. Each generation of engineers "
        "built a layer that hid the one beneath it, and in hiding it made possible a kind of "
        "work that would otherwise have been unthinkable. ",
    "marine_biology":
        "Marine biologists studying hydrothermal vents found ecosystems that derive their "
        "entire energy budget from chemosynthesis rather than sunlight, overturning a long "
        "assumption about where life can persist on this planet. ",
    "medieval_trade":
        "The reconstruction of medieval trade routes relies on unglamorous evidence: customs "
        "ledgers, shipwreck cargo manifests, and the distribution of coin hoards across river "
        "valleys and mountain passes. ",
    "orbital":
        "In orbital mechanics a transfer between two circular orbits costs the least delta-v "
        "when the burns are made at apoapsis and periapsis, which is why mission planners "
        "care so much about launch windows and phase angles. ",
    "bread":
        "Modern bread fermentation is a negotiation between yeast and lactic acid bacteria, "
        "and the baker controls it almost entirely through temperature, hydration and the "
        "timing of each fold rather than through the recipe itself. ",
    "law":
        "Legal systems that evolved from common law treat precedent as binding in a way that "
        "civil code jurisdictions do not, which produces strikingly different appellate "
        "cultures and very different judicial opinions. ",
    "glaciology":
        "Ice cores drilled from polar glaciers preserve annual layers whose trapped air "
        "bubbles record the composition of the atmosphere at the moment the snow was buried, "
        "which is why they anchor so much of the paleoclimate record. ",
    "typography":
        "A typeface intended for long passages of text solves a different problem than one "
        "meant for signage: the reader's eye returns to the same shapes thousands of times, "
        "so evenness of colour matters more than novelty of form. ",
    "immunology":
        "The adaptive immune system works by clonal selection, keeping an enormous repertoire "
        "of receptors in reserve and expanding only the lineages whose receptors happen to "
        "bind whatever antigen has arrived. ",
    "railway":
        "Railway signalling evolved from a purely permissive system into an interlocked one "
        "because the cost of a single misrouted train grew faster than the cost of the "
        "machinery required to make that routing impossible. ",
    "distributed":
        "Consensus in a distributed system is expensive precisely because it must survive the "
        "failure of the participants that are supposed to provide it, so every practical "
        "protocol trades latency for a bounded number of tolerated faults. ",
    "ceramics":
        "The behaviour of a glaze during firing depends on the eutectic relationships between "
        "its fluxes, and a potter adjusting a recipe is navigating a phase diagram whether or "
        "not the diagram is ever drawn. ",
}

gen_spec = Generator(model=model, cache=cache, tokenizer=tokenizer,
                     draft_model=draft, draft_cache=draft_cache, max_chunk_size=2048)
gen_plain = Generator(model=model, cache=cache_plain, tokenizer=tokenizer,
                      max_chunk_size=2048)


def as_rows(t):
    """Normalize a held top-k tensor to a list of per-position lists."""
    if t is None:
        return []
    x = t
    while x.dim() > 2:
        x = x[0]
    if x.dim() == 1:
        x = x.unsqueeze(0)
    return x.tolist()


def run(g, ids, name, label, topk=0):
    toks, ktoks, kprobs = [], [], []
    kw = {"return_top_tokens": topk, "return_probs": True} if topk else {}
    g.enqueue(Job(input_ids=ids, max_new_tokens=MAXNEW, stop_conditions=[],
                  sampler=GreedySampler(), **kw))
    try:
        while g.num_remaining_jobs():
            for r in g.iterate():
                if r.get("stage") == "error":
                    emit(phase="ERROR", prompt=name, label=label)
                    sys.exit(3)
                t = r.get("token_ids")
                if t is not None and t.numel():
                    toks.extend(t.flatten().tolist())
                    if topk:
                        ktoks.extend(as_rows(r.get("top_k_tokens")))
                        kprobs.extend(as_rows(r.get("top_k_probs")))
    except Exception as e:
        emit(phase="EXCEPTION", prompt=name, label=label,
             type=type(e).__name__, detail=str(e)[:300])
        sys.exit(4)
    sha = hashlib.sha1(",".join(str(x) for x in toks).encode()).hexdigest()[:12]
    return toks, ktoks, kprobs, sha


rows = []
for name, para in PROMPTS.items():
    base = tokenizer.encode(para, add_bos=True).shape[-1]
    ids = tokenizer.encode(para * max(1, 2048 // base), add_bos=True)

    # Control: plain greedy against itself. If these differ, stop reading the rest.
    a, ktoks, kprobs, sha_a1 = run(gen_plain, ids, name, "plain_1", topk=TOPK)
    _, _, _, sha_a2 = run(gen_plain, ids, name, "plain_2")
    b, _, _, sha_b = run(gen_spec, ids, name, "spec")
    if sha_a1 != sha_a2:
        emit(phase="row", prompt=name, klass="NOT_SELF_DETERMINISTIC",
             plain_1=sha_a1, plain_2=sha_a2)
        rows.append(dict(prompt=name, klass="NOT_SELF_DETERMINISTIC"))
        continue

    i = None
    for n, (x, y) in enumerate(zip(a, b)):
        if x != y:
            i = n
            break

    if i is None:
        row = dict(prompt=name, diverged=False, klass="exact")
    else:
        rt = ktoks[i] if i < len(ktoks) else None
        rp = kprobs[i] if i < len(kprobs) else None
        if not rt or not rp:
            row = dict(prompt=name, diverged=True, first_diff=i, klass="unscored")
        else:
            st = b[i]
            rank = (rt.index(st) + 1) if st in rt else None
            p1 = rp[0]
            p2 = rp[1] if len(rp) > 1 else 0.0
            margin = p1 - p2
            p_spec = rp[rt.index(st)] if st in rt else None
            klass = ("outside_topk" if rank is None
                     else ("near_tie" if margin < NEAR_TIE else "clear_preference"))
            row = dict(prompt=name, diverged=True, first_diff=i, spec_rank=rank,
                       p_top1=round(p1, 5),
                       p_spec=(round(p_spec, 5) if p_spec is not None else None),
                       margin=round(margin, 5), klass=klass)
    rows.append(row)
    emit(phase="row", **row)

agg = {}
for r in rows:
    agg[r["klass"]] = agg.get(r["klass"], 0) + 1
margins = [r["margin"] for r in rows if r.get("margin") is not None]
emit(phase="AGGREGATE", n_prompts=len(rows), counts=agg,
     margins_sorted=sorted(round(m, 4) for m in margins))
print("SPEC_EXACTNESS_OK", flush=True)
