"""Which EXL3 kernel path is most accurate, measured against a dequantized-weight reference.

Decode and speculative verify do not run the same kernel on this pack. Dispatch is by row
count, and there are three regimes:

    rows 1 to 2     int8 GEMV (exl3_gemv_int8, gated at size_m <= 2)
    rows 3 to 144   exl3_gemm / exl3_mgemm
    rows over 144   reconstruct_hgemm          (AUTO_RECONSTRUCT_THRESHOLD)

That matters because exl3_gemv_int8 quantizes ACTIVATIONS to int8 and the other paths do not.
Single-token decode therefore runs the lowest-precision path, while a speculative verify window
of 3+ rows runs a higher-precision one, and prefill chunks run the highest. It is why greedy
output can differ between a run with a drafter and a run without one: the two sides straddle a
kernel boundary. See BENCHMARKS.md, "Speculation is not output-exact".

This script measures which path is actually closer to the truth, as an operator-level A/B:
real activations are captured from live LinearEXL3 modules during a forward, then the SAME
input row is pushed through each kernel and compared against reconstruct_hgemm, which
dequantizes the trellis to a dense weight and does an ordinary hgemm with no activation
quantization. Same weights, same row, same process, one reference, so kernel noise is the only
variable and engine nondeterminism cancels.

Reading the output:
    sqnr_db     higher is better. Reference calibration: tests/deepseek_v41/moe_grouped_equiv.py
                records ordinary fp16 kernel differences at around 1e-3 relative.
    mean_rel    mean relative error against the reference.
    max_rel     REPORTED BUT NOT MEANINGFUL. Near-zero reference elements make it explode
                (values above 90 occur). Judge on sqnr_db and mean_rel.

Caveat on the reference: reconstruct_hgemm is itself an fp16 hgemm, not an fp32 oracle, so
these are relative rankings between paths rather than absolute error figures.

Environment:
    MODEL_DIR   pack directory (required)
    EXL3_SRC    ExLlamaV3 checkout to import from (optional)

Loads the main model only, no drafter, at CTX 2048, and generates four tokens purely to
populate the capture. Cheap compared to the decode benchmarks, and it measures no wall-clock,
so light concurrent load on the box does not invalidate it.
"""
import inspect
import json
import os
import sys
import threading
import time

_src = os.environ.get("EXL3_SRC")
if _src:
    sys.path.insert(0, _src)

import torch


def emit(**kw):
    try:
        print(json.dumps(kw), flush=True)
    except TypeError:
        print(json.dumps({k: str(v) for k, v in kw.items()}), flush=True)


def meminfo():
    d = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":", 1)
        if k in ("MemAvailable", "AnonPages", "SUnreclaim"):
            d[k] = round(int(v.split()[0]) / 1048576, 2)
    return d


# Discover the API and print it before anything expensive, so a failure still returns enough
# to diagnose it rather than needing a blind retry.
from exllamav3.modules.quant import exl3 as exl3mod
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.util.measures import sqnr, cosine_error

inner_cls = None
for nm in dir(exl3mod):
    o = getattr(exl3mod, nm)
    if inspect.isclass(o) and hasattr(o, "reconstruct_hgemm"):
        inner_cls = o
        break

emit(phase="api",
     module_consts={k: getattr(exl3mod, k) for k in dir(exl3mod)
                    if k.isupper() and isinstance(getattr(exl3mod, k), (int, float, bool))},
     inner_class=(inner_cls.__name__ if inner_cls else None),
     forward_sig=(str(inspect.signature(inner_cls.forward)) if inner_cls else None),
     ext_reconstruct=[n for n in dir(ext) if "reconstruct" in n.lower()])

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

from exllamav3 import Config, Model, Tokenizer
from exllamav3.cache.cache import Cache
from exllamav3.generator import Generator, Job
from exllamav3.generator.sampler import GreedySampler

M = os.environ["MODEL_DIR"]
CTX = 2048
config = Config.from_directory(M)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens=CTX, max_history=4, max_batch_size=1)
t0 = time.time()
model.load("cuda:0", progressbar=False, verbose=False)
tokenizer = Tokenizer.from_config(config)
emit(phase="loaded", seconds=round(time.time() - t0, 1), mem=meminfo())

captured = {}
patched = []


def find_targets(mod, path="", out=None, limit=6):
    """Collect LinearEXL3 inners that carry mul1, since exl3_gemm.cu gates the int8 GEMV
    dispatch on it."""
    if out is None:
        out = []
    if len(out) >= limit:
        return out
    inner = getattr(mod, "inner", None)
    if inner is not None and inner_cls is not None and isinstance(inner, inner_cls):
        if getattr(inner, "mul1", False):
            out.append((path, mod, inner))
    for attr in ("modules", "layers", "children"):
        sub = getattr(mod, attr, None)
        if isinstance(sub, (list, tuple)):
            for i, m2 in enumerate(sub):
                if hasattr(m2, "__dict__"):
                    find_targets(m2, f"{path}.{attr}[{i}]", out, limit)
                if len(out) >= limit:
                    return out
    return out


targets = find_targets(model, "model")
emit(phase="targets", n=len(targets), paths=[p for p, _, _ in targets][:6])
if not targets:
    emit(phase="ABORT_NO_TARGETS")
    sys.exit(11)

# Monkeypatch rather than register a hook, so this does not depend on these being nn.Modules.
for path, mod, inner in targets:
    orig = inner.forward

    def make(orig_fn, key):
        def wrapper(*a, **kw):
            if key not in captured and len(a) >= 1 and torch.is_tensor(a[0]):
                x = a[0]
                if x.dim() >= 2 and x.shape[-2] >= 4:
                    captured[key] = (x.detach().clone(), a[1:], dict(kw))
            return orig_fn(*a, **kw)
        return wrapper

    inner.forward = make(orig, path)
    patched.append((inner, orig))

para = ("Marine biologists studying hydrothermal vents found ecosystems that derive their "
        "entire energy budget from chemosynthesis rather than sunlight. ")
ids = tokenizer.encode(para * 8, add_bos=True)
gen = Generator(model=model, cache=cache, tokenizer=tokenizer, max_chunk_size=1024)
gen.enqueue(Job(input_ids=ids, max_new_tokens=4, stop_conditions=[], sampler=GreedySampler()))
while gen.num_remaining_jobs():
    for _ in gen.iterate():
        pass

for inner, orig in patched:
    inner.forward = orig
emit(phase="captured", n=len(captured))
if not captured:
    emit(phase="ABORT_NO_CAPTURE")
    sys.exit(12)


def stats(a, ref):
    a32, r32 = a.float(), ref.float()
    denom = r32.abs().clamp_min(1e-6)
    return dict(
        sqnr_db=round(sqnr(a32, r32), 3),
        cos_err=round(cosine_error(a32, r32), 8),
        max_abs=round((a32 - r32).abs().max().item(), 6),
        max_rel=round(((a32 - r32).abs() / denom).max().item(), 6),
        mean_rel=round(((a32 - r32).abs() / denom).mean().item(), 8),
    )


wins = {"gemm": 0, "gemv": 0}
for path, mod, inner in targets:
    if path not in captured:
        continue
    x, rest, kw = captured[path]
    x = x.reshape(-1, x.shape[-1])
    if x.shape[0] < 3:
        continue
    row1 = x[0:1].contiguous()
    row3 = x[0:3].contiguous()

    res = {"path": path, "K": int(inner.K), "in": int(x.shape[-1])}
    try:
        with torch.inference_mode():
            y_gemv = inner.forward(row1, *rest, **kw)                 # m = 1  -> int8 GEMV
            y_gemm = inner.forward(row3, *rest, **kw)[0:1]            # m = 3  -> exl3_gemm
            y_ref = inner.reconstruct_hgemm(
                row1, getattr(inner, "out_dtype", torch.half))        # dense, no act quant
        res["gemv_vs_ref"] = stats(y_gemv, y_ref)
        res["gemm_vs_ref"] = stats(y_gemm, y_ref)
        g = res["gemv_vs_ref"]["sqnr_db"]
        m_ = res["gemm_vs_ref"]["sqnr_db"]
        res["closer_to_reference"] = ("gemm" if m_ > g else "gemv")
        res["sqnr_gap_db"] = round(m_ - g, 3)
        wins[res["closer_to_reference"]] += 1
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {str(e)[:220]}"
    emit(phase="cmp", **res)

emit(phase="SUMMARY", wins=wins,
     verdict=("GEMM closer in %d of %d modules" % (wins["gemm"], wins["gemm"] + wins["gemv"])))
print("KERNEL_ACCURACY_OK", flush=True)
