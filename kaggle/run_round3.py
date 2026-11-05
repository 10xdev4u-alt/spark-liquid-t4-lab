"""Round 3: SDPA vs eager + sparse scaling to 16K/32K.

Q1: SDPA let FULL fit at 9K? Q2: sparse needle at 16K/32K?
Q3: sampling (temp 0.7) vs greedy quality on thinking models.
Sequential pip flip (Spark 4.57 / LFM 5.x), expandable_segments.
"""
import subprocess, sys, os
from pathlib import Path

WORK = Path("/kaggle/working")
WORK.mkdir(parents=True, exist_ok=True)

WORKER = r'''
import json, time, torch, sys
from pathlib import Path

def sparse_ids(model, tok, dev, full, sink=128, window=512, n_chunks=64, keep=8):
    ids = full["input_ids"][0]
    n = ids.shape[0]
    if n <= sink + window: return full
    emb = model.get_input_embeddings()
    mid = ids[sink:n-window]
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

def gen(model, tok, dev, ids, max_new=64, do_sample=False):
    torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    kw = dict(max_new_tokens=max_new, pad_token_id=tok.eos_token_id)
    if do_sample:
        kw.update(do_sample=True, temperature=0.7, top_p=0.95)
    else:
        kw.update(do_sample=False)
    with torch.no_grad():
        g = model.generate(**ids, **kw)
    torch.cuda.synchronize(dev)
    return g, time.perf_counter() - t0

def bench(tag, model_id, gpu):
    torch.cuda.set_device(gpu)
    dev = f"cuda:{gpu}"
    out = {"tag": tag, "attn": {}, "scale": {}, "sampling": {}}
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # --- Q1: eager vs sdpa at 9K ---
    needle9 = ("The secret code is ZEBRA-7391. " + ("Filler history science text. " * 900)
               + "The secret code is ZEBRA-7391. " + ("More filler geography rivers. " * 900)
               + "\nQuestion: What is the secret code?")
    full9 = tok(needle9, return_tensors="pt").to(dev)
    for impl in ("eager", "sdpa"):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True, torch_dtype=torch.float16,
                attn_implementation=impl).to(dev).eval()
            if tok.eos_token_id is not None:
                model.generation_config.eos_token_id = tok.eos_token_id
            torch.cuda.reset_peak_memory_stats(gpu)
            g, dt = gen(model, tok, dev, full9)
            reply = tok.decode(g[0][full9["input_ids"].shape[1]:], skip_special_tokens=True)
            out["attn"][impl] = {"ok": True, "wall_s": round(dt, 2),
                "peak_vram_gb": round(torch.cuda.max_memory_allocated(gpu)/1e9, 2),
                "needle_pass": "ZEBRA-7391" in reply}
            del model
        except Exception as e:
            out["attn"][impl] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:180]}"}
        torch.cuda.empty_cache()

    # --- Q2: sparse scaling 16K/32K (sdpa model) ---
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=torch.float16,
            attn_implementation="sdpa").to(dev).eval()
        if tok.eos_token_id is not None:
            model.generation_config.eos_token_id = tok.eos_token_id
        for target in (16000, 32000):
            reps = max(1, target // 8)
            needle = ("The secret code is ZEBRA-7391. "
                      + ("Filler history science text. " * reps)
                      + "The secret code is ZEBRA-7391. "
                      + ("More filler geography rivers. " * reps)
                      + "\nQuestion: What is the secret code?")
            full = tok(needle, return_tensors="pt").to(dev)
            row = {"ctx_tokens": int(full["input_ids"].shape[1])}
            for name, v in (("SINK+WINDOW", {"input_ids": torch.cat(
                    [full["input_ids"][0][:128], full["input_ids"][0][-512:]]).unsqueeze(0).to(dev)}),
                            ("TOPK", sparse_ids(model, tok, dev, full))):
                try:
                    torch.cuda.reset_peak_memory_stats(gpu)
                    g, dt = gen(model, tok, dev, v)
                    reply = tok.decode(g[0][v["input_ids"].shape[1]:], skip_special_tokens=True)
                    row[name] = {"in": int(v["input_ids"].shape[1]), "wall_s": round(dt, 2),
                        "vram": round(torch.cuda.max_memory_allocated(gpu)/1e9, 2),
                        "needle": "ZEBRA-7391" in reply}
                except Exception as e:
                    row[name] = {"error": f"{type(e).__name__}: {str(e)[:150]}"}
                    torch.cuda.empty_cache()
            out["scale"][str(target)] = row
    except Exception as e:
        out["scale"]["error"] = f"{type(e).__name__}: {str(e)[:200]}"

    # --- Q3: sampling vs greedy quality ---
    try:
        for name, sample in (("greedy", False), ("sample_t07", True)):
            ids = tok("Write a Python function to find the nth Fibonacci number.",
                      return_tensors="pt").to(dev)
            g, _ = gen(model, tok, dev, ids, max_new=120, do_sample=sample)
            out["sampling"][name] = tok.decode(
                g[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)[:500]
    except Exception as e:
        out["sampling"]["error"] = f"{type(e).__name__}: {str(e)[:200]}"

    Path(f"/kaggle/working/r3_{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"DONE r3 {tag}", flush=True)

bench(sys.argv[1], sys.argv[2], int(sys.argv[3]))
'''

(WORK / "r3_worker.py").write_text(WORKER)
env = dict(os.environ)
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
print("pinning Spark 4.57...", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.1"], env=env)
subprocess.run([sys.executable, str(WORK / "r3_worker.py"), "spark4b", "XHToken/Spark-X2.6-4B", "0"], env=env)
print("flipping to transformers 5.x...", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers>=5.0"], env=env)
subprocess.run([sys.executable, str(WORK / "r3_worker.py"), "lfm3_17b", "LiquidAI/LFM3-1.7B", "1"], env=env)
print("ALL DONE", flush=True)
