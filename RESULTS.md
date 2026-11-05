# Results — Spark-X2.6-4B vs LFM3-1.7B on Kaggle 2xT4 (fp16, eager attn)

Kernel: jprinceraj/spark-x2-6-4b-vs-lfm3-1-7b-on-2xt4 v2 (COMPLETE)
Raw: kaggle_out/spark4b.json + lfm3_17b.json

| Metric | Spark-X2.6-4B (GPU0) | LFM3-1.7B (GPU1) |
|---|---|---|
| Params | 4.11B | 1.73B |
| Layers | 36 (27 sliding / 9 full) | 24 (13 conv / 5 sliding / 6 full) |
| Load time | 73.3s | 30.5s |
| Peak VRAM | 10.67 GB | 4.46 GB |
| Decode tok/s @512 | 7.4 | 19.6 |
| Decode tok/s @2048 | 10.6 | 22.9 |
| Decode tok/s @4096 | 7.2 | 19.2 |
| PPL (fixed text) | 1.34 | 1.38 |
| transformers | 4.57.1 (pinned) | 5.22.1 |

## Gotchas found (v1 failures -> v2 fixes)
1. Spark `modeling_spark.py` crashes on transformers 5.x: `AttributeError: 'list' object has no attribute 'keys'`. Spark pins 4.57.1. LFM3 needs 5.x. Same env can't hold both -> sequential pip flip.
2. LFM OOM at 16K ctx on 14.5GiB T4 (SDPA materializes 6.25GiB). Capped at 4096 + expandable_segments.
3. Greedy (do_sample=False) on thinking models = repetition loops. Spark sample 1 degenerates; needs thinking-off chat template + sampling.

## Reading
- LFM3 is 2.6x faster decode, 2.4x lighter VRAM — conv hybrid wins on T4.
- Spark's 10.6 tok/s peak vs LFM 22.9: eager attention + head_dim 256 hurts.
- PPL tie (1.34 vs 1.38) on repetitive text is non-discriminative.
- Quality samples: both weak greedy; LFM Kyoto plan was the only clean win.
- Next: sdpa/flash attn, thinking-off template, 8K ctx with gradient checkpointing.
