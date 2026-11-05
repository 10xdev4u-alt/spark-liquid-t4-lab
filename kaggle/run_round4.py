"""R4 crazy: Spark eager 32K + top_k=0 sampling + int4 + KV eviction proxy.

- Spark: eager ONLY (SDPA unsupported), sparse needle 16K/32K
- Sampling: override top_k=0 (fixes -1 crash)
- int4: bitsandbytes 4bit load, VRAM/speed vs fp16
- LFM: KV eviction proxy = full-cache FULL ctx vs sink+window truncated decode
Sequential pip flip (Spark 4.57 / LFM 5.x).
"""
import subprocess, sys, os
from pathlib import Path

WORK = Path("/kaggle/working")
WORK.mkdir(parents=True, exist_ok=True)

SPARK_WORKER = r'''
import json, time, torch, sys
from pathlib import Path

def sparse_ids(model, tok, dev, full, sink=128, window=512, n_chunks=64, keep=8):
    ids = full["input_ids"][0]; n = ids.shape[0]
    if n <= sink + window: return full
    emb = model.get_input_embeddings(); mid = ids[sink:n-window]
    with torch.no_grad():
        me = emb(mid.unsqueeze(0).to(dev))[0]
        q = emb(ids[-1:].unsqueeze(0).to(dev))[0, 0]
    me = torch.nn.functional.normalize(me.float(), dim=-1)
    q = torch.nn.functional.normalize(q.float(), dim=-1)
    ch = max(1, me.shape[0] // n_chunks)
    scores = sorted([(i, float((me[i*ch:(i+1)*ch] @ q).mean()))
                     for i in range(n_chunks) if me[i*ch:(i+1)*ch].shape[0] > 0],
                    key=lambda x: -x[1])
    keep_idx = sorted(i for i, _ in scores[:keep])
    parts = [ids[:sink]] + [mid[i*ch:(i+1)*ch] for i in keep_idx] + [ids[n-window:]]
    return {"input_ids": torch.cat(parts).unsqueeze(0).to(dev)}

def gen(model, tok, dev, ids, max_new=64, sample=False):
    torch.cuda.synchronize(dev); t0 = time.perf_counter()
    kw = dict(max_new_tokens=max_new, pad_token_id=tok.eos_token_id)
    if sample:
        kw.update(do_sample=True, temperature=0.7, top_p=0.95, top_k=0)
    else:
        kw.update(do_sample=False)
    with torch.no_grad():
        g = model.generate(**kw, **ids)
    torch.cuda.synchronize(dev)
    return g, time.perf_counter() - t0

out = {"scale": {}, "sampling": {}, "int4": {}}
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("XHToken/Spark-X2.6-4B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "XHToken/Spark-X2.6-4B", trust_remote_code=True, torch_dtype=torch.float16,
    attn_implementation="eager").to("cuda:0").eval()
if tok.eos_token_id is not None:
    model.generation_config.eos_token_id = tok.eos_token_id

# scale 16K/32K sparse
for target in (16000, 32000):
    reps = max(1, target // 8)
    needle = ("The secret code is ZEBRA-7391. " + ("Filler history science text. " * reps)
              + "The secret code is ZEBRA-7391. " + ("More filler geography rivers. " * reps)
              + "\nQuestion: What is the secret code?")
    full = tok(needle, return_tensors="pt").to("cuda:0")
    row = {"ctx": int(full["input_ids"].shape[1])}
    for name, v in (("SINK+WINDOW", {"input_ids": torch.cat(
            [full["input_ids"][0][:128], full["input_ids"][0][-512:]]).unsqueeze(0).to("cuda:0")}),
                    ("TOPK", sparse_ids(model, tok, "cuda:0", full))):
        try:
            torch.cuda.reset_peak_memory_stats(0)
            g, dt = gen(model, tok, "cuda:0", v)
            rep = tok.decode(g[0][v["input_ids"].shape[1]:], skip_special_tokens=True)
            row[name] = {"in": int(v["input_ids"].shape[1]), "wall_s": round(dt, 2),
                "vram": round(torch.cuda.max_memory_allocated(0)/1e9, 2),
                "needle": "ZEBRA-7391" in rep}
        except Exception as e:
            row[name] = {"error": f"{type(e).__name__}: {str(e)[:150]}"}
            torch.cuda.empty_cache()
    out["scale"][str(target)] = row

# sampling fix
for name, s in (("greedy", False), ("sample_t07_topk0", True)):
    try:
        ids = tok("Write a Python function to find the nth Fibonacci number.",
                  return_tensors="pt").to("cuda:0")
        g, _ = gen(model, tok, "cuda:0", ids, max_new=120, sample=s)
        out["sampling"][name] = tok.decode(g[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)[:500]
    except Exception as e:
        out["sampling"][name] = f"FAIL {type(e).__name__}: {str(e)[:200]}"
        torch.cuda.empty_cache()
del model; torch.cuda.empty_cache()

# int4
try:
    import bitsandbytes as bnb
    from transformers import BitsAndBytesConfig
    torch.cuda.reset_peak_memory_stats(0)
    bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    m4 = AutoModelForCausalLM.from_pretrained(
        "XHToken/Spark-X2.6-4B", trust_remote_code=True, quantization_config=bnb_cfg,
        attn_implementation="eager").to("cuda:0").eval()
    ids = tok("What is 17 * 24? Show work briefly.", return_tensors="pt").to("cuda:0")
    g, dt = gen(m4, tok, "cuda:0", ids, max_new=48)
    out["int4"] = {"ok": True, "wall_s": round(dt, 2),
        "vram": round(torch.cuda.max_memory_allocated(0)/1e9, 2),
        "reply": tok.decode(g[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)[:300]}
except Exception as e:
    out["int4"] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:250]}"}

Path("/kaggle/working/r4_spark4b.json").write_text(json.dumps(out, indent=2))
print("DONE r4 spark", flush=True)
'''

LFM_WORKER = r'''
import json, time, torch, sys
from pathlib import Path

out = {"kv": {}}
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("LiquidAI/LFM3-1.7B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "LiquidAI/LFM3-1.7B", trust_remote_code=True, torch_dtype=torch.float16,
    attn_implementation="eager").to("cuda:1").eval()
if tok.eos_token_id is not None:
    model.generation_config.eos_token_id = tok.eos_token_id

# KV eviction proxy at 32K: FULL cache (all tokens) vs evicted (sink+window only)
reps = 32000 // 8
needle = ("The secret code is ZEBRA-7391. " + ("Filler history science text. " * reps)
          + "The secret code is ZEBRA-7391. " + ("More filler geography rivers. " * reps)
          + "\nQuestion: What is the secret code?")
full = tok(needle, return_tensors="pt").to("cuda:1")
out["kv"]["ctx_tokens"] = int(full["input_ids"].shape[1])

# FULL with cache (expect OOM or slow)
try:
    torch.cuda.reset_peak_memory_stats(1)
    torch.cuda.synchronize(1); t0 = time.perf_counter()
    with torch.no_grad():
        g = model.generate(**full, max_new_tokens=48, do_sample=False,
                           pad_token_id=tok.eos_token_id, use_cache=True)
    torch.cuda.synchronize(1)
    rep = tok.decode(g[0][full["input_ids"].shape[1]:], skip_special_tokens=True)
    out["kv"]["FULL_cache"] = {"wall_s": round(time.perf_counter()-t0, 2),
        "vram": round(torch.cuda.max_memory_allocated(1)/1e9, 2), "needle": "ZEBRA-7391" in rep}
except Exception as e:
    out["kv"]["FULL_cache"] = {"error": f"{type(e).__name__}: {str(e)[:180]}"}
    torch.cuda.empty_cache()

# EVICTED: sink+window only (KV never grows beyond 640)
trunc = {"input_ids": torch.cat([full["input_ids"][0][:128], full["input_ids"][0][-512:]]).unsqueeze(0).to("cuda:1")}
try:
    torch.cuda.reset_peak_memory_stats(1)
    torch.cuda.synchronize(1); t0 = time.perf_counter()
    with torch.no_grad():
        g = model.generate(**trunc, max_new_tokens=48, do_sample=False,
                           pad_token_id=tok.eos_token_id, use_cache=True)
    torch.cuda.synchronize(1)
    rep = tok.decode(g[0][trunc["input_ids"].shape[1]:], skip_special_tokens=True)
    out["kv"]["EVICTED_sinkwin"] = {"wall_s": round(time.perf_counter()-t0, 2),
        "vram": round(torch.cuda.max_memory_allocated(1)/1e9, 2), "needle": "ZEBRA-7391" in rep,
        "reply": rep[:200]}
except Exception as e:
    out["kv"]["EVICTED_sinkwin"] = {"error": f"{type(e).__name__}: {str(e)[:180]}"}

Path("/kaggle/working/r4_lfm3.json").write_text(json.dumps(out, indent=2))
print("DONE r4 lfm", flush=True)
'''

(WORK / "r4_spark.py").write_text(SPARK_WORKER)
(WORK / "r4_lfm.py").write_text(LFM_WORKER)
env = dict(os.environ)
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
print("pinning Spark 4.57 + bnb...", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.1", "bitsandbytes"], env=env)
subprocess.run([sys.executable, str(WORK / "r4_spark.py")], env=env)
print("flipping to transformers 5.x...", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers>=5.0"], env=env)
subprocess.run([sys.executable, str(WORK / "r4_lfm.py")], env=env)
print("ALL DONE", flush=True)
