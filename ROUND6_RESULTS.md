# Round 6 results — Ollama + GGUF (2xT4)

Kernel: jprinceraj/r6-ollama-gguf-on-2xt4 v2 COMPLETE
Pattern: copied from Test-LLMs.ipynb (zstd pre-install, ollama pull hf.co GGUF, /api/create, FLASH_ATTENTION=1, KV q8_0)

## v1 FAILURE root cause
`ERROR: This version requires zstd for extraction` — ollama install.sh needs zstd; notebook cell 1 installs it first, my script didn't. Fix: apt-get install zstd before ollama.

## Results — BOTH models run 100% GPU in Ollama
| Model | GGUF | VRAM (ollama ps) | Speed | Quality |
|---|---|---|---|---|
| Spark-4B Q4_K_M | 3.1GB | 100% GPU, 8192 ctx | 29.1 tok/s (coding), 52.9 tok/s (math) | Fibonacci iterative + explainer, 17*24=408 correct |
| LFM3-1.7B Q4_K_M | 1.2GB | 100% GPU, 8192 ctx | **110.4 tok/s**, 114.1 tok/s | Table comparison + LaTeX math, 408 correct |

## Big surprise
LFM3 GGUF FAILED in llama-cpp-python 0.3.36 (R5) but WORKS in Ollama — Ollama bundles a newer llama.cpp with LFM3 conv-hybrid support. Runtime version > pip version. Ollama is now the unified runner for BOTH.

## Speed context (T4, Q4, 100% GPU)
- LFM3 110 tok/s vs HF fp16 eager 19-23 tok/s (R1) — **~5x faster** (Q4 + flash attn + Ollama kernels)
- Spark 29-53 tok/s vs HF fp16 eager 7-11 tok/s — **~4-5x faster**, eager tax gone
- Spark first prompt wall 81s = model load; subsequent 1.6s

## Verdict
Ollama + GGUF Q4 is the production stack for T4: both models, 100% GPU, quality intact, 4-5x faster than HF fp16. Test-LLMs notebook pattern confirmed as the right base.
Permalink: ollama-models pollutes /kaggle/working output (GBs) — download results with --file-pattern, and set OLLAMA_MODELS outside working in future.
