"""R5: chat template fix + GGUF llama.cpp smoke.

A) Spark/LFM with apply_chat_template, vendor sampling (temp1.0/top_p0.95)
   vs raw greedy from R1-R4. Does template kill Spark loops?
B) GGUF Q4_K_M via llama.cpp (CPU smoke: quality + VRAM/RAM; speed = CPU bound, labeled)
C) HF fp16 vs GGUF Q4 quality side-by-side on coding + needle
"""
import subprocess, sys, os
from pathlib import Path

WORK = Path("/kaggle/working")
WORK.mkdir(parents=True, exist_ok=True)

TEMPLATE_WORKER = r'''
import json, torch, sys
from pathlib import Path

def test(tag, model_id, gpu):
    torch.cuda.set_device(gpu)
    dev = f"cuda:{gpu}"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=torch.float16,
        attn_implementation="eager").to(dev).eval()
    if tok.eos_token_id is not None:
        model.generation_config.eos_token_id = tok.eos_token_id
    out = {"tag": tag, "has_template": tok.chat_template is not None}
    prompts = [
        {"role": "user", "content": "Write a Python function to find the nth Fibonacci number."},
        {"role": "user", "content": "What is 17 * 24? Show your work briefly."},
    ]
    # A) template + vendor sampling
    try:
        ids = tok.apply_chat_template(prompts, tokenize=True, return_tensors="pt").to(dev)
        with torch.no_grad():
            g = model.generate(**ids, max_new_tokens=150, do_sample=True,
                               temperature=1.0, top_p=0.95, top_k=0,
                               pad_token_id=tok.eos_token_id)
        out["template_vendor_sample"] = tok.decode(g[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)[:600]
    except Exception as e:
        out["template_vendor_sample"] = f"FAIL {type(e).__name__}: {str(e)[:250]}"
        torch.cuda.empty_cache()
    # B) template + greedy
    try:
        ids = tok.apply_chat_template(prompts, tokenize=True, return_tensors="pt").to(dev)
        with torch.no_grad():
            g = model.generate(**ids, max_new_tokens=150, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        out["template_greedy"] = tok.decode(g[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)[:600]
    except Exception as e:
        out["template_greedy"] = f"FAIL {type(e).__name__}: {str(e)[:250]}"
        torch.cuda.empty_cache()
    Path(f"/kaggle/working/r5_tmpl_{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"DONE template {tag}", flush=True)

test(sys.argv[1], sys.argv[2], int(sys.argv[3]))
'''

GGUF_WORKER = r'''
import json, sys
from pathlib import Path
from huggingface_hub import hf_hub_download

out = {"gguf": {}}
# llama.cpp python (CPU wheel - smoke only)
try:
    import llama_cpp
    out["llama_cpp_version"] = llama_cpp.__version__
except Exception as e:
    out["gguf"]["llama_cpp_import"] = f"FAIL {type(e).__name__}: {str(e)[:200]}"

for tag, repo, fname in [
    ("lfm3_17b", "LiquidAI/LFM3-1.7B-GGUF", "LFM3-1.7B-Q4_K_M.gguf"),
    ("spark4b", "XHToken/Spark-X2.6-4B-GGUF", "Spark-X2.6-4B-Q4_K_M.gguf"),
]:
    try:
        p = hf_hub_download(repo, fname)
        out["gguf"][tag] = {"path": p,
            "size_gb": round(Path(p).stat().st_size / 1e9, 2)}
        from llama_cpp import Llama
        llm = Llama(model_path=p, n_gpu_layers=0, n_ctx=8192, verbose=False)
        r = llm.create_chat_completion(
            messages=[{"role": "user", "content": "Write a Python function to find the nth Fibonacci number."}],
            max_tokens=120, temperature=0.7)
        out["gguf"][tag]["reply"] = r["choices"][0]["message"]["content"][:600]
        r2 = llm.create_chat_completion(
            messages=[{"role": "user", "content": "What is 17 * 24? Show work briefly."}],
            max_tokens=80, temperature=0.7)
        out["gguf"][tag]["math"] = r2["choices"][0]["message"]["content"][:300]
    except Exception as e:
        out["gguf"][tag] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}

Path("/kaggle/working/r5_gguf.json").write_text(json.dumps(out, indent=2))
print("DONE gguf", flush=True)
'''

(WORK / "r5_tmpl.py").write_text(TEMPLATE_WORKER)
(WORK / "r5_gguf.py").write_text(GGUF_WORKER)
env = dict(os.environ)
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
print("pinning Spark 4.57...", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.1"], env=env)
subprocess.run([sys.executable, str(WORK / "r5_tmpl.py"), "spark4b", "XHToken/Spark-X2.6-4B", "0"], env=env)
print("flipping to transformers 5.x...", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers>=5.0"], env=env)
subprocess.run([sys.executable, str(WORK / "r5_tmpl.py"), "lfm3_17b", "LiquidAI/LFM3-1.7B", "1"], env=env)
print("installing llama-cpp-python (cpu smoke)...", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "llama-cpp-python"], env=env)
subprocess.run([sys.executable, str(WORK / "r5_gguf.py")], env=env)
print("ALL DONE", flush=True)
