# Round 4 results — crazy menu (2xT4)

Kernel: jprinceraj/r4-crazy-spark-eager-int4-kv-on-2xt4 v1 COMPLETE

## 1. Spark eager sparse scaling — PASS at 44K ctx
| Ctx | SINK+WINDOW (640t) | TOPK |
|---|---|---|
| 22,036 | 5.09s, 8.42GB flat, PASS | 3312t, 6.39s, 10.37GB, PASS |
| 44,036 | 4.13s, 8.42GB flat, PASS | 6064t, 10.35s, 14.35GB (near OOM), PASS |
Spark now matches LFM scaling story. TOPK VRAM grows ~1.5GB per 8K ctx — at 64K it would OOM. Sink+Window is the safe 64K+ mode.

## 2. Sampling fix (top_k=0) — crash gone, quality still bad
- Greedy: "This is a large and complex task..." loop
- Sampled t0.7: "The answer is 9." loop (wrong Fibonacci, repetitive)
Root cause candidate: raw prompt WITHOUT chat template on a thinking model. Spark needs its chat template + thinking control. LFM raw works; Spark raw doesn't. R5: use tokenizer.apply_chat_template.

## 3. int4 (bitsandbytes) — FAILED
`NotImplementedError: baddbmm_cuda not implemented for 'Byte'` — bnb doesn't support Spark's custom arch (trust_remote_code). Lesson: custom-arch models need GGUF/llama.cpp for quantization, not bnb. R5: try llama.cpp GGUF Q4 for Spark.

## 4. KV eviction proxy (LFM3, 48K ctx)
- FULL_cache: OOM (704MiB alloc fail, 14.02GB used)
- EVICTED_sinkwin: 2.44s, 3.55GB, needle PASS
True KV eviction = same win as context truncation: bounded cache, flat VRAM, needle survives.

## Leaderboard so far
- Best long-ctx on T4: LFM3 Sink+Window (3.58GB @48K) > Spark Sink+Window (8.42GB @44K)
- Best raw quality: LFM3 (codes greedy-clean)
- Spark: needs template + GGUF for int4, eager-only tax
