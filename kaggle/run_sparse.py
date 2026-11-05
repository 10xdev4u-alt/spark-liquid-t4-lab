"""Sparse attention (LSA-lite) experiment on Spark-4B vs LFM3-1.7B.

Idea: LSA keeps sink + window + k long-range tokens instead of all T.
We test the same trade-off at the context level (model-agnostic, no kernel
surgery): FULL (all tokens) vs SINK+WINDOW (first 128 + last 512) vs
TOPK (sink + window + best middle chunks by query-chunk similarity).

Tasks: needle retrieval (fact planted mid-doc), long perplexity, speed/VRAM.
Context 4K/8K, T4 16GB cap. Sequential pip flip (Spark 4.57 / LFM 5.x).
"""
import subprocess, sys, os
from pathlib import Path

WORK = Path("/kaggle/working")
WORK.mkdir(parents=True, exist_ok=True)

WORKER = r'''
import json, time, torch, sys
from pathlib import Path

def select_topk(tok, model, dev, full_ids, sink=128, window=512, n_chunks=64, keep_chunks=8):
    """LSA-lite: split middle into chunks, score by embedding cosine to last token."""
    ids = full_ids["input_ids"][0]
    n = ids.shape[0]
    if n <= sink + window:
        return full_ids
    mid = ids[sink:n - window]
    emb = model.get_input_embeddings()
    with torch.no_grad():
        mid_emb = emb(mid.unsqueeze(0).to(dev))[0]          # [M, H]
        q = emb(ids[-1:].unsqueeze(0).to(dev))[0]          # [H]
    mid_emb = torch.nn.functional.normalize(mid_emb.float(), dim=-1)
    q = torch.nn.functional.normalize(q.float(), dim=-1)
    chunk = max(1, mid_emb.shape[0] // n_chunks)
    scores = []
    for i in range(n_chunks):
        seg = mid_emb[i*chunk:(i+1)*chunk]
        if seg.shape[0] == 0: continue
        scores.append((i, float((seg @ q).mean())))
    scores.sort(key=lambda x: -x[1])
    keep = sorted(i for i, _ in scores[:keep_chunks])
    parts = [ids[:sink]]
    for i in keep:
        parts.append(mid[i*chunk:(i+1)*chunk])
    parts.append(ids[n - window:])
    sparse = torch.cat(parts).unsqueeze(0)
    return {"input_ids": sparse.to(dev)}

def run_mode(model, tok, dev, ids, max_new=64):
    torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    with torch.no_grad():
        gen = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize(dev)
    dt = time.perf_counter() - t0
    return gen, dt

def bench(tag, model_id, gpu):
    torch.cuda.set_device(gpu)
    dev = f"cuda:{gpu}"
    out = {"tag": tag, "model_id": model_id, "gpu": gpu, "modes": {}, "needle": {}}
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=torch.float16,
        attn_implementation="eager").to(dev).eval()
    if tok.eos_token_id is not None:
        model.generation_config.eos_token_id = tok.eos_token_id

    needle = "The secret code is ZEBRA-7391. " + ("Filler text about history and science. " * 600)
    needle += "The secret code is ZEBRA-7391. " + ("More filler about geography and rivers. " * 600)
    needle += "\nQuestion: What is the secret code?"
    full = tok(needle, return_tensors="pt").to(dev)
    n_tok = full["input_ids"].shape[1]

    variants = {"FULL": full}
    # sink+window
    ids = full["input_ids"][0]
    variants["SINK+WINDOW"] = {"input_ids": torch.cat(
        [ids[:128], ids[-512:]]).unsqueeze(0).to(dev)}
    variants["TOPK"] = select_topk(tok, model, dev, full)

    for name, v in variants.items():
        try:
            torch.cuda.reset_peak_memory_stats(gpu)
            gen, dt = run_mode(model, tok, dev, v)
            reply = tok.decode(gen[0][v["input_ids"].shape[1]:], skip_special_tokens=True)
            out["modes"][name] = {
                "input_tokens": int(v["input_ids"].shape[1]),
                "wall_s": round(dt, 2),
                "peak_vram_gb": round(torch.cuda.max_memory_allocated(gpu) / 1e9, 2),
                "needle_pass": "ZEBRA-7391" in reply,
                "reply_head": reply[:200],
            }
        except Exception as e:
            out["modes"][name] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            torch.cuda.empty_cache()
    out["context_tokens"] = int(n_tok)
    Path(f"/kaggle/working/sparse_{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"DONE sparse {tag} ctx={n_tok} modes={list(out['modes'].keys())}", flush=True)

bench(sys.argv[1], sys.argv[2], int(sys.argv[3]))
'''

(WORK / "sparse_worker.py").write_text(WORKER)
env = dict(os.environ)
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

print("pinning Spark 4.57...", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.1"], env=env)
subprocess.run([sys.executable, str(WORK / "sparse_worker.py"), "spark4b", "XHToken/Spark-X2.6-4B", "0"], env=env)
print("flipping to transformers 5.x...", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers>=5.0"], env=env)
subprocess.run([sys.executable, str(WORK / "sparse_worker.py"), "lfm3_17b", "LiquidAI/LFM3-1.7B", "1"], env=env)
print("ALL DONE", flush=True)
