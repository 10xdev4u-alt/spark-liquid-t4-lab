"""Spark-X2.6-4B (GPU0) vs LFM3-1.7B (GPU1) v2 — fixes from v1 failures.

v1 failures:
- Spark: AttributeError 'list' object has no attribute 'keys' in modeling_spark.py
  -> transformers version clash (Spark pinned 4.57.1, Kaggle ships 5.x).
  Fix: pip install transformers==4.57.1 in the Spark worker before import.
- LFM: OOM at 16K ctx (SDPA materializes 6.25GiB).
  Fix: cap context at 4096, expandable_segments, per-stage try/except so
  partial results always land in json.
"""
import json, subprocess, sys
from pathlib import Path

WORK = Path("/kaggle/working")
WORK.mkdir(parents=True, exist_ok=True)

WORKER = r'''
import json, time, torch, sys
from pathlib import Path

def bench(tag, model_id, gpu):
    torch.cuda.set_device(gpu)
    dev = f"cuda:{gpu}"
    out = {"tag": tag, "model_id": model_id, "gpu": gpu, "stages": {}}
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    import transformers
    out["transformers"] = transformers.__version__
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
        }
        lt = getattr(cfg, "layer_types", None) or []
        from collections import Counter
        out["layer_type_counts"] = dict(Counter(lt))
    except Exception as e:
        out["config_error"] = f"{type(e).__name__}: {e}"[:300]
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=torch.float16,
        attn_implementation="eager")
    model.to(dev).eval()
    out["load_s"] = round(time.perf_counter() - t0, 1)
    out["params_b"] = round(sum(p.numel() for p in model.parameters()) / 1e9, 2)
    out["peak_vram_gb"] = round(torch.cuda.max_memory_allocated(gpu) / 1e9, 2)
    if tok.eos_token_id is not None:
        model.generation_config.eos_token_id = tok.eos_token_id

    # perplexity
    try:
        text = ("The quick brown fox jumps over the lazy dog. " * 20
                + "In 2026, small language models run locally on consumer GPUs. " * 10)
        ids = tok(text, return_tensors="pt").to(dev)
        with torch.no_grad():
            loss = model(**ids, labels=ids["input_ids"]).loss.item()
        out["ppl_fixed_text"] = round(float(torch.exp(torch.tensor(loss))), 2)
        out["stages"]["ppl"] = "ok"
    except Exception as e:
        out["stages"]["ppl"] = f"FAIL {type(e).__name__}: {str(e)[:200]}"
        torch.cuda.empty_cache()

    # speed vs context (capped 4096 for T4 16GB)
    out["speed"] = {}
    for ctx in (512, 2048, 4096):
        try:
            prompt = "Explain photosynthesis in detail. " * max(1, ctx // 8)
            ids = tok(prompt, return_tensors="pt").to(dev)
            torch.cuda.synchronize(gpu)
            t0 = time.perf_counter()
            with torch.no_grad():
                gen = model.generate(**ids, max_new_tokens=48, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            torch.cuda.synchronize(gpu)
            dt = time.perf_counter() - t0
            new_toks = gen.shape[1] - ids["input_ids"].shape[1]
            out["speed"][str(ctx)] = {"wall_s": round(dt, 2),
                "decode_tok_s": round(new_toks / dt, 1) if dt > 0 else None}
            out["stages"][f"speed_{ctx}"] = "ok"
        except Exception as e:
            out["stages"][f"speed_{ctx}"] = f"FAIL {type(e).__name__}: {str(e)[:200]}"
            torch.cuda.empty_cache()
        out["peak_vram_gb"] = round(torch.cuda.max_memory_allocated(gpu) / 1e9, 2)

    # quality samples
    out["samples"] = []
    for p in ["Write a Python function to find the nth Fibonacci number.",
              "What is 17 * 24? Show your work briefly.",
              "List three steps to plan a weekend trip to Kyoto."]:
        try:
            ids = tok(p, return_tensors="pt").to(dev)
            with torch.no_grad():
                gen = model.generate(**ids, max_new_tokens=100, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            out["samples"].append({"prompt": p,
                "reply": tok.decode(gen[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)[:500]})
        except Exception as e:
            out["samples"].append({"prompt": p, "reply": f"FAIL {type(e).__name__}: {str(e)[:150]}"})
            torch.cuda.empty_cache()
    Path(f"/kaggle/working/{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"DONE {tag} params={out['params_b']}B ppl={out.get('ppl_fixed_text')} vram={out['peak_vram_gb']}GB stages={out['stages']}", flush=True)

bench(sys.argv[1], sys.argv[2], int(sys.argv[3]))
'''

(WORK / "worker.py").write_text(WORKER)
(WORK / "spark_prep.py").write_text(
    "import subprocess, sys\n"
    "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',"
    " 'transformers==4.57.1', 'accelerate'], check=False)\n"
    "print('spark env pinned to transformers 4.57.1', flush=True)\n"
)

import subprocess, sys, os
env = dict(os.environ)
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Spark needs transformers 4.57.1, LFM3 needs 5.x -> run SEQUENTIALLY
# with a pip flip between, still using both GPUs (one at a time).
print("pinning Spark env...", flush=True)
subprocess.run([sys.executable, str(WORK / "spark_prep.py")], env=env)
print(" launching spark4b on GPU0", flush=True)
p = subprocess.run([sys.executable, str(WORK / "worker.py"), "spark4b", "XHToken/Spark-X2.6-4B", "0"], env=env)
print(f" spark rc={p.returncode}", flush=True)

print("flipping to transformers 5.x for LFM3...", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers>=5.0"], env=env)
print(" launching lfm3_17b on GPU1", flush=True)
p = subprocess.run([sys.executable, str(WORK / "worker.py"), "lfm3_17b", "LiquidAI/LFM3-1.7B", "1"], env=env)
print(f" lfm rc={p.returncode}", flush=True)
print("ALL DONE", flush=True)
