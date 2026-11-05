"""Spark-X2.6-4B (GPU0) vs LFM3-1.7B (GPU1) — parallel head-to-head on Kaggle 2xT4.

Each worker: load model (fp16, T4 has no bf16), dump arch, measure
prefill TTFT + decode tok/s at 512/4K/16K context, perplexity on fixed
text, and 3 qualitative generations. Writes /kaggle/working/<tag>.json.
"""
import json, time, subprocess, sys
from pathlib import Path

WORK = Path("/kaggle/working")
WORK.mkdir(parents=True, exist_ok=True)

WORKER = r'''
import json, time, torch
from pathlib import Path

def bench(tag, model_id, gpu):
    torch.cuda.set_device(gpu)
    dev = f"cuda:{gpu}"
    out = {"tag": tag, "model_id": model_id, "gpu": gpu}
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    try:
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        out["config"] = {
            "arch": getattr(cfg, "architectures", None),
            "layers": getattr(cfg, "num_hidden_layers", None),
            "hidden": getattr(cfg, "hidden_size", None),
            "heads_q": getattr(cfg, "num_attention_heads", None),
            "heads_kv": getattr(cfg, "num_key_value_heads", None),
            "max_pos": getattr(cfg, "max_position_embeddings", None),
            "layer_types": getattr(cfg, "layer_types", None),
        }
        lt = getattr(cfg, "layer_types", None) or []
        from collections import Counter
        out["layer_type_counts"] = dict(Counter(lt))
    except Exception as e:
        out["config_error"] = str(e)[:300]
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=torch.float16)
    model.to(dev).eval()
    out["load_s"] = round(time.perf_counter() - t0, 1)
    out["params_b"] = round(sum(p.numel() for p in model.parameters()) / 1e9, 2)
    out["peak_vram_gb"] = round(torch.cuda.max_memory_allocated(gpu) / 1e9, 2)
    if hasattr(tok, "eos_token_id") and tok.eos_token_id is not None:
        model.generation_config.eos_token_id = tok.eos_token_id

    # --- perplexity on fixed text ---
    text = ("The quick brown fox jumps over the lazy dog. " * 20
            + "In 2026, small language models run locally on consumer GPUs. "
            + "Attention mechanisms let transformers weigh which tokens matter most. " * 10)
    ids = tok(text, return_tensors="pt").to(dev)
    with torch.no_grad():
        loss = model(**ids, labels=ids["input_ids"]).loss.item()
    out["ppl_fixed_text"] = round(float(torch.exp(torch.tensor(loss))), 2)

    # --- prefill TTFT + decode speed vs context ---
    out["speed"] = {}
    for ctx in (512, 4096, 16384):
        prompt = "Explain photosynthesis in detail. " * max(1, ctx // 8)
        ids = tok(prompt, return_tensors="pt").to(dev)
        torch.cuda.synchronize(gpu)
        t0 = time.perf_counter()
        with torch.no_grad():
            gen = model.generate(**ids, max_new_tokens=64, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        torch.cuda.synchronize(gpu)
        dt = time.perf_counter() - t0
        new_toks = gen.shape[1] - ids["input_ids"].shape[1]
        out["speed"][str(ctx)] = {
            "wall_s_64tok": round(dt, 2),
            "decode_tok_s": round(new_toks / dt, 1) if dt > 0 else None,
        }
        out["peak_vram_gb"] = round(torch.cuda.max_memory_allocated(gpu) / 1e9, 2)

    # --- qualitative prompts ---
    prompts = [
        "Write a Python function to find the nth Fibonacci number.",
        "What is 17 * 24? Show your work briefly.",
        "List three steps to plan a weekend trip to Kyoto.",
    ]
    out["samples"] = []
    for p in prompts:
        ids = tok(p, return_tensors="pt").to(dev)
        with torch.no_grad():
            gen = model.generate(**ids, max_new_tokens=120, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        out["samples"].append({"prompt": p,
            "reply": tok.decode(gen[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)[:600]})
    Path(f"/kaggle/working/{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"DONE {tag} params={out['params_b']}B ppl={out['ppl_fixed_text']} vram={out['peak_vram_gb']}GB", flush=True)

import sys
bench(sys.argv[1], sys.argv[2], int(sys.argv[3]))
'''

(WORK / "worker.py").write_text(WORKER)

import subprocess, sys
jobs = [
    ("spark4b", "XHToken/Spark-X2.6-4B", "0"),
    ("lfm3_17b", "LiquidAI/LFM3-1.7B", "1"),
]
procs = []
for tag, mid, gpu in jobs:
    print(f" launching {tag} on GPU{gpu}", flush=True)
    procs.append(subprocess.Popen(
        [sys.executable, str(WORK / "worker.py"), tag, mid, gpu],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
for p in procs:
    for line in p.stdout:
        print(line, end="", flush=True)
    p.wait()
print("ALL DONE", flush=True)
