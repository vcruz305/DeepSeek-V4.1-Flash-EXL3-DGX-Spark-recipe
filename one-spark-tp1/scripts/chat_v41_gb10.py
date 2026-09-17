#!/usr/bin/env python3
"""Interactive DeepSeek-V4.1-Flash EXL3 on one DGX Spark (GB10), native ExLlamaV3.

This is the driver script `run_tp1.sh` expects in $ENTRY:

    MODEL_DIR=/models/DSV4.1-Flash-SAGE-EXL3-1.59bpw \
    ENTRY=scripts/chat_v41_gb10.py \
      scripts/run_tp1.sh

Each line you type is sent as a raw completion prompt; the continuation streams back followed by
tok/s and draft acceptance. Commands: /tokens N, /quit.

Environment:
    MODEL_DIR   required, the pack directory (base pack plus the exllamav3/ overlay
                parts copied in beside the shards)
    EXL3_SRC    optional, path to an ExLlamaV3 source checkout to import instead of the
                installed package (only needed if you did not `pip install .` the fork)
    CTX         context length (default 6144)
    DRAFT       1 to load the DSpark/MTP drafter (default 1)
    PREWARM     1 to fadvise(WILLNEED) the weights before the first token (default 1)
    CHUNK       prefill chunk (default 2048)
    TOKENS      max new tokens per turn (default 256)
    EXL3_DSPARK_CONF, EXL3_ATS_MMAP, EXL3_ATS_COPY are read by ExLlamaV3 itself; see ../README.md
"""
import os, sys, time

# Import ExLlamaV3 from a source checkout when asked, otherwise use the installed package.
_src = os.environ.get("EXL3_SRC")
if _src:
    sys.path.insert(0, _src)

import torch
from exllamav3 import Config, Model, Tokenizer
from exllamav3.cache.cache import Cache
from exllamav3.generator import Generator, Job
from exllamav3.generator.sampler import GreedySampler

M = os.environ.get("MODEL_DIR")
if not M:
    sys.exit("MODEL_DIR is not set; point it at the pack directory "
             "(base pack plus the exllamav3/ overlay parts)")
if not os.path.isdir(M):
    sys.exit(f"MODEL_DIR={M} is not a directory")
CTX = int(os.environ.get("CTX", "6144"))
DRAFT = os.environ.get("DRAFT", "1") == "1"
PREWARM = os.environ.get("PREWARM", "1") == "1"
tokens = int(os.environ.get("TOKENS", "256"))

t0 = time.time()
config = Config.from_directory(M)
model = Model.from_config(config)
draft = draft_cache = None
max_history = 1
if DRAFT:
    draft = Model.from_config(config, component = "mtp")
    max_history = draft.caps.get("default_draft_size", 4)
cache = Cache(model, max_num_tokens = CTX, max_history = max_history, max_batch_size = 1)
if DRAFT:
    draft_cache = Cache(draft, max_num_tokens = CTX)
model.load("cuda:0", progressbar = False, verbose = False)
if DRAFT:
    draft.load("cuda:0", progressbar = False, verbose = False)
tokenizer = Tokenizer.from_config(config)

if PREWARM:
    stc = config.stc
    fds = {}
    for key, fn in stc.tensor_file_map.items():
        if ".engram.embed." in key or key.startswith("vision."):
            continue
        b, e = stc.file_headers[fn][key]["data_offsets"]
        fds.setdefault(fn, os.open(fn, os.O_RDONLY))
        os.posix_fadvise(fds[fn], stc.file_headers[fn]["_header_offset"] + b, e - b, os.POSIX_FADV_WILLNEED)
    for fd in fds.values():
        os.close(fd)

# Chunk 2048 with the main model resident in CUDA: 213-229 tok/s at 2-6k tokens. Chunk 4096's
# activations do not fit alongside 107 GiB of resident weights and the run is killed.
CHUNK = int(os.environ.get("CHUNK", "2048"))
if DRAFT:
    generator = Generator(model = model, cache = cache, tokenizer = tokenizer, draft_model = draft,
                          draft_cache = draft_cache, max_chunk_size = CHUNK)
else:
    generator = Generator(model = model, cache = cache, tokenizer = tokenizer, max_chunk_size = CHUNK)
print(f"ready in {time.time() - t0:.1f}s  model={os.path.basename(M)} ctx={CTX} draft={DRAFT} "
      f"conf={os.environ.get('EXL3_DSPARK_CONF', '0.5')} tokens={tokens}", flush = True)

while True:
    try:
        prompt = input("\n> ")
    except EOFError:
        break
    if not prompt.strip():
        continue
    if prompt.strip() == "/quit":
        break
    if prompt.startswith("/tokens"):
        tokens = int(prompt.split()[1])
        print(f"max new tokens = {tokens}")
        continue
    ids = tokenizer.encode(prompt, add_bos = True)
    if ids.shape[-1] + tokens > CTX:
        print(f"prompt of {ids.shape[-1]} tokens plus {tokens} new exceeds ctx {CTX}")
        continue
    generator.enqueue(Job(input_ids = ids, max_new_tokens = tokens, stop_conditions = [tokenizer.eos_token_id],
                          sampler = GreedySampler()))
    first = last = None
    n = 0
    acc = rej = 0
    while generator.num_remaining_jobs():
        for r in generator.iterate():
            if r.get("stage") == "error":
                print("\n[error]", {k: v for k, v in r.items() if not torch.is_tensor(v) and k != "job"})
                break
            if "accepted_draft_tokens" in r:
                acc, rej = r["accepted_draft_tokens"], r.get("rejected_draft_tokens", 0)
            if r.get("stage") != "streaming":
                continue
            if r.get("text"):
                print(r["text"], end = "", flush = True)
            t = r.get("token_ids")
            if t is not None and t.numel():
                now = time.perf_counter()
                if first is None:
                    first = now
                else:
                    n += t.numel()
                last = now
    rate = n / (last - first) if first and last and last > first else 0.0
    extra = f", draft acceptance {acc / (acc + rej):.0%}" if acc + rej else ""
    print(f"\n[{n + 1} tokens, {rate:.1f} tok/s decode{extra}]", flush = True)
