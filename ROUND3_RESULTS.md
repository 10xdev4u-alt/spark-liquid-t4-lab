# Round 3 results — SDPA vs eager + sparse scaling (2xT4)

Kernel: jprinceraj/round3-sdpa-vs-sparse-on-2xt4 v1 COMPLETE

## Q1: SDPA save FULL at 9K? NO.
- LFM3 eager OOM (6.99GiB alloc) AND sdpa OOM (same 6.99GiB) — SDPA did not help; conv cache + fragmentation dominate.
- Spark: `ValueError: Spark2_5ForCausalLM does not support SDPA yet` — eager-only architecture. FULL OOMs as before.

## Q2: Sparse scaling (LFM3, sdpa worker)
| Target | Actual ctx | SINK+WINDOW (640t) | TOPK |
|---|---|---|---|
| 16K | 24,031 | 3.28s, 3.58GB, needle PASS | 3560t, 3.72s, 5.63GB, PASS |
| 32K | 48,031 | 2.82s, 3.58GB, PASS | 6560t, 5.53s, 10.38GB, PASS |
Sink+Window VRAM is FLAT (3.58GB at 24K and 48K) — input size fixed. TOPK grows with ctx.
Spark 16K/32K NOT tested: sdpa gate crashed before scale loop (eager fallback needed next round).

## Q3: Greedy vs sampling
- LFM3 greedy Fibonacci: CORRECT iterative code. Sampled: also correct. LFM fine either way.
- Spark greedy: repetition loop ("This is a large and complex task..." xN). Sampled: CRASH `top_k has to be strictly positive, but is -1` — Spark generation_config ships top_k=-1 (their recommended sampling temp1/top_p0.95/topk-1), invalid in transformers 4.57 without override. Fix: top_k=0 or strip config.

## Next round (R4)
1. Spark scale test with eager (skip sdpa gate), sampling with top_k=0
2. True KV-cache sparse: sliding-window cache eviction vs full cache at 32K
3. int4 GGUF via llama.cpp as unified runner (kills transformers-version split)
