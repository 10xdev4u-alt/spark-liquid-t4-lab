"""R7: MTP speculative decode + 32K stress in Ollama (Test-LLMs pattern).

- draft_num_predict=2 (MTP) vs baseline tok/s for Spark+LFM Q4
- Needle at 32K via Ollama native context (num_ctx 32768)
- /api/chat vs /api/generate comparison
Follows ~/Downloads/test-llms.ipynb cells 2-4 exactly (zstd, env, pull/create/warm).
"""
import subprocess, sys, os, time, json, urllib.request, shutil
from pathlib import Path

def run(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)

print("=== nvidia-smi ===", flush=True)
run(["nvidia-smi"])
if not shutil.which("zstd"):
    run(["apt-get", "update", "-qq"]); run(["apt-get", "install", "-y", "-qq", "zstd"])
if not shutil.which("ollama"):
    run(["bash", "-lc", "curl -fsSL https://ollama.com/install.sh | sh"])
print(subprocess.check_output(["ollama", "--version"], text=True).strip(), flush=True)

OLLAMA_URL = "http://127.0.0.1:11434"
MODELS = {
    "spark4b": "hf.co/XHToken/Spark-X2.6-4B-GGUF:Q4_K_M",
    "lfm3_17b": "hf.co/LiquidAI/LFM3-1.7B-GGUF:Q4_K_M",
}
NUM_CTX = 32768

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

PERSIST = Path("/tmp/ollama-models")  # OUTSIDE /kaggle/working to avoid output bloat (R6 lesson)
PERSIST.mkdir(parents=True, exist_ok=True)
os.environ["OLLAMA_MODELS"] = str(PERSIST)
oenv = os.environ.copy()
oenv.update({
    "OLLAMA_HOST": "127.0.0.1:11434", "OLLAMA_KEEP_ALIVE": "-1",
    "OLLAMA_FLASH_ATTENTION": "1", "OLLAMA_KV_CACHE_TYPE": "q8_0",
    "OLLAMA_CONTEXT_LENGTH": str(NUM_CTX), "OLLAMA_NUM_PARALLEL": "1",
    "OLLAMA_MAX_LOADED_MODELS": "1",
})
if not ollama_ready():
    logp = Path("/tmp/ollama-server.log"); logf = logp.open("a")
    proc = subprocess.Popen(["ollama", "serve"], env=oenv,
                            stdout=logf, stderr=subprocess.STDOUT)
    for _ in range(60):
        if ollama_ready(): break
        if proc.poll() is not None: raise RuntimeError(f"Ollama exited. Check {logp}")
        time.sleep(1)
print("Ollama ready", flush=True)

out = {"models": {}}
for tag, src in MODELS.items():
    name = f"{tag}-q4"
    res = {}
    try:
        print(f"=== pull {src} ===", flush=True)
        run(["ollama", "pull", src], env=oenv)
        # baseline create (no MTP)
        ollama_request("/api/create", {"model": name, "from": src,
            "parameters": {"num_ctx": NUM_CTX}, "stream": False})
        ollama_request("/api/generate", {"model": name, "prompt": "",
            "stream": False, "keep_alive": -1, "options": {"num_ctx": NUM_CTX}})
        res["baseline_load"] = True
    except Exception as e:
        res["load_error"] = f"{type(e).__name__}: {str(e)[:300]}"
        out["models"][tag] = res; continue

    # --- MTP vs baseline speed (same prompt, temp 0) ---
    res["speed"] = {}
    for label, opts in [("baseline", {"temperature": 0, "num_ctx": NUM_CTX}),
                        ("mtp_draft2", {"temperature": 0, "num_ctx": NUM_CTX, "draft_num_predict": 2})]:
        try:
            t0 = time.perf_counter()
            r = ollama_request("/api/generate", {"model": name, "stream": False,
                "keep_alive": -1,
                "prompt": "Write a Python function to find the nth Fibonacci number with docstring.",
                "options": opts})
            ec = r.get("eval_count", 0); es = r.get("eval_duration", 0) / 1e9
            res["speed"][label] = {"tok_s": round(ec / es, 1) if es else None,
                "eval_count": ec, "reply_head": r.get("response", "")[:200]}
        except Exception as e:
            res["speed"][label] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}

    # --- needle at 32K (Ollama native ctx) ---
    try:
        needle = ("The secret code is ZEBRA-7391. " + ("Filler history science text. " * 4000)
                  + "The secret code is ZEBRA-7391. " + ("More filler geography rivers. " * 4000)
                  + "\nQuestion: What is the secret code? Answer briefly.")
        r = ollama_request("/api/generate", {"model": name, "prompt": needle,
            "think": False, "stream": False, "keep_alive": -1,
            "options": {"temperature": 0, "num_ctx": NUM_CTX}})
        res["needle_32k"] = {"pass": "ZEBRA-7391" in r.get("response", ""),
            "reply": r.get("response", "")[:250],
            "prompt_eval_count": r.get("prompt_eval_count"),
            "prompt_eval_duration_s": round(r.get("prompt_eval_duration", 0) / 1e9, 2)}
    except Exception as e:
        res["needle_32k"] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}

    # --- chat vs generate ---
    try:
        r = ollama_request("/api/chat", {"model": name, "stream": False, "keep_alive": -1,
            "messages": [{"role": "user", "content": "What is 17 * 24? Show work briefly."}],
            "options": {"temperature": 0.7, "num_ctx": NUM_CTX}})
        res["chat"] = {"reply": r["message"]["content"][:250],
            "eval_count": r.get("eval_count"),
            "tok_s": round(r.get("eval_count", 0) / (r.get("eval_duration", 1) / 1e9), 1) if r.get("eval_duration") else None}
    except Exception as e:
        res["chat"] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}

    res["ps"] = subprocess.check_output(["ollama", "ps"], env=oenv, text=True)[:600]
    out["models"][tag] = res

Path("/kaggle/working/r7_mtp.json").write_text(json.dumps(out, indent=2))
print(subprocess.check_output(["ollama", "list"], env=oenv, text=True), flush=True)
print("ALL DONE", flush=True)
