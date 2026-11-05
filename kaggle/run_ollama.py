"""R6: Ollama + GGUF on Kaggle 2xT4.

Install Ollama, serve, create models from local GGUF (Spark Q4_K_M, LFM3 Q4_K_M),
test chat, measure speed/VRAM via `ollama ps` + generate timings.
Ollama is llama.cpp-based -> LFM3 arch likely FAILS (same as R5 llama.cpp).
Spark GGUF should load and run on T4 CUDA via Ollama.
"""
import subprocess, sys, os, time, json
from pathlib import Path

WORK = Path("/kaggle/working")
WORK.mkdir(parents=True, exist_ok=True)

env = dict(os.environ)
env["OLLAMA_HOST"] = "127.0.0.1:11434"
env["OLLAMA_MODELS"] = str(WORK / "ollama_models")

def sh(cmd, **kw):
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env, **kw)
    print((r.stdout or "")[-1500:], flush=True)
    if r.stderr:
        print("STDERR:", r.stderr[-800:], flush=True)
    return r

# 1. install ollama
print("=== install ollama ===", flush=True)
r = sh("which ollama || curl -fsSL https://ollama.com/install.sh | sh")
print("ollama check rc", r.returncode, flush=True)

# 2. start serve in background
print("=== start ollama serve ===", flush=True)
serve = subprocess.Popen(["ollama", "serve"], env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
# wait for ready
for i in range(30):
    time.sleep(2)
    r = subprocess.run(["ollama", "list"], capture_output=True, text=True, env=env)
    if r.returncode == 0:
        print(f"ollama ready after {(i+1)*2}s", flush=True)
        break
else:
    print("ollama never ready, log:", flush=True)
    serve.terminate()
    try:
        out, _ = serve.communicate(timeout=5)
        print(out[-2000:])
    except Exception as e:
        print("log err", e)
    sys.exit(0)

# 3. download GGUFs
from huggingface_hub import hf_hub_download
ggufs = {}
for tag, repo, fname in [
    ("spark4b", "XHToken/Spark-X2.6-4B-GGUF", "Spark-X2.6-4B-Q4_K_M.gguf"),
    ("lfm3_17b", "LiquidAI/LFM3-1.7B-GGUF", "LFM3-1.7B-Q4_K_M.gguf"),
]:
    try:
        p = hf_hub_download(repo, fname)
        ggufs[tag] = p
        print(f"{tag} gguf: {p} ({Path(p).stat().st_size/1e9:.2f}GB)", flush=True)
    except Exception as e:
        print(f"{tag} download FAIL {type(e).__name__}: {str(e)[:200]}", flush=True)

# 4. create + test each
out = {"models": {}}
for tag, path in ggufs.items():
    name = f"{tag}-q4"
    modelfile = WORK / f"Modelfile-{tag}"
    modelfile.write_text(f"FROM {path}\n")
    r = sh(f"ollama create {name} -f {modelfile}")
    res = {"create_rc": r.returncode}
    if r.returncode == 0:
        for prompt in ["Write a Python function to find the nth Fibonacci number.",
                       "What is 17 * 24? Show work briefly."]:
            t0 = time.perf_counter()
            r2 = sh(f"ollama run {name} \"{prompt}\"", timeout=300)
            res.setdefault("replies", []).append(
                {"prompt": prompt, "wall_s": round(time.perf_counter()-t0, 1),
                 "reply": (r2.stdout or "")[:500]})
        r3 = sh("ollama ps")
        res["ps"] = (r3.stdout or "")[:500]
    out["models"][tag] = res

(WORK / "r6_ollama.json").write_text(json.dumps(out, indent=2))
print("=== ollama models ===", flush=True)
sh("ollama list")
print("ALL DONE", flush=True)
serve.terminate()
