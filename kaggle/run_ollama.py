"""R6 v2: Ollama + GGUF on Kaggle 2xT4 — follows Test-LLMs notebook pattern.

Key fixes vs v1 (ERROR: zstd missing):
- apt-get install zstd BEFORE ollama install (notebook cell 1)
- OLLAMA_MODELS=/kaggle/working/ollama-models, FLASH_ATTENTION=1, KV q8_0
- ollama pull hf.co/... GGUF directly (no hf_hub_download needed)
- create via /api/create, generate via /api/generate (notebook cells 2-4)
Models: hf.co/XHToken/Spark-X2.6-4B-GGUF:Q4_K_M and
        hf.co/LiquidAI/LFM3-1.7B-GGUF:Q4_K_M
"""
import subprocess, sys, os, time, json, urllib.request, shutil
from pathlib import Path

def run(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)

print("=== nvidia-smi ===", flush=True)
run(["nvidia-smi"])

if not shutil.which("zstd"):
    print("=== install zstd ===", flush=True)
    run(["apt-get", "update", "-qq"])
    run(["apt-get", "install", "-y", "-qq", "zstd"])

if not shutil.which("ollama"):
    print("=== install ollama ===", flush=True)
    run(["bash", "-lc", "curl -fsSL https://ollama.com/install.sh | sh"])

print(subprocess.check_output(["ollama", "--version"], text=True).strip(), flush=True)

OLLAMA_URL = "http://127.0.0.1:11434"
MODELS = {
    "spark4b": "hf.co/XHToken/Spark-X2.6-4B-GGUF:Q4_K_M",
    "lfm3_17b": "hf.co/LiquidAI/LFM3-1.7B-GGUF:Q4_K_M",
}

def ollama_request(endpoint, payload, timeout=3600):
    req = urllib.request.Request(
        f"{OLLAMA_URL}{endpoint}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def ollama_ready():
    try:
        urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=2)
        return True
    except Exception:
        return False

PERSIST = Path("/kaggle/working/ollama-models")
PERSIST.mkdir(parents=True, exist_ok=True)
os.environ["OLLAMA_MODELS"] = str(PERSIST)
oenv = os.environ.copy()
oenv.update({
    "OLLAMA_HOST": "127.0.0.1:11434", "OLLAMA_KEEP_ALIVE": "-1",
    "OLLAMA_FLASH_ATTENTION": "1", "OLLAMA_KV_CACHE_TYPE": "q8_0",
    "OLLAMA_CONTEXT_LENGTH": "8192", "OLLAMA_NUM_PARALLEL": "1",
    "OLLAMA_MAX_LOADED_MODELS": "1",
})

if not ollama_ready():
    logp = Path("/tmp/ollama-server.log")
    logf = logp.open("a")
    proc = subprocess.Popen(["ollama", "serve"], env=oenv,
                            stdout=logf, stderr=subprocess.STDOUT)
    for _ in range(60):
        if ollama_ready():
            break
        if proc.poll() is not None:
            raise RuntimeError(f"Ollama exited early. Check {logp}")
        time.sleep(1)
    else:
        raise RuntimeError(f"Ollama did not become ready. Check {logp}")
print("Ollama ready", flush=True)

out = {"models": {}}
for tag, src in MODELS.items():
    name = f"{tag}-q4"
    res = {}
    try:
        print(f"=== pull {src} ===", flush=True)
        run(["ollama", "pull", src], env=oenv)
        ollama_request("/api/create", {
            "model": name, "from": src,
            "parameters": {"num_ctx": 8192}, "stream": False})
        # warm load
        ollama_request("/api/generate", {
            "model": name, "prompt": "", "stream": False, "keep_alive": -1,
            "options": {"num_ctx": 8192}})
        res["loaded"] = True
    except Exception as e:
        res["load_error"] = f"{type(e).__name__}: {str(e)[:300]}"
        out["models"][tag] = res
        continue
    res["replies"] = []
    for prompt in ["Write a Python function to find the nth Fibonacci number.",
                   "What is 17 * 24? Show work briefly."]:
        try:
            t0 = time.perf_counter()
            r = ollama_request("/api/generate", {
                "model": name, "prompt": prompt, "think": False,
                "stream": False, "keep_alive": -1,
                "options": {"temperature": 0.7, "num_ctx": 8192}})
            ec = r.get("eval_count", 0)
            es = r.get("eval_duration", 0) / 1e9
            res["replies"].append({
                "prompt": prompt,
                "wall_s": round(time.perf_counter() - t0, 1),
                "tok_s": round(ec / es, 1) if es else None,
                "reply": r.get("response", "")[:500]})
        except Exception as e:
            res["replies"].append({"prompt": prompt,
                "error": f"{type(e).__name__}: {str(e)[:200]}"})
    res["ps"] = subprocess.check_output(["ollama", "ps"], env=oenv, text=True)[:600]
    out["models"][tag] = res

Path("/kaggle/working/r6_ollama.json").write_text(json.dumps(out, indent=2))
print(subprocess.check_output(["ollama", "list"], env=oenv, text=True), flush=True)
print("ALL DONE", flush=True)
