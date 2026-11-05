# Sparse attention (LSA-lite) — Spark-4B vs LFM3-1.7B on 2xT4

Kernel: jprinceraj/spark-vs-liquid-sparse-attention-on-2xt4 v2 COMPLETE
Modes: FULL (all tokens) vs SINK+WINDOW (128+512=640) vs TOPK (sink+window+8 best middle chunks=~1.7K)
Task: needle-in-haystack, secret code planted mid-doc, ~9-9.6K token context

| Model | Mode | Input toks | Wall (64 tok) | Peak VRAM | Needle |
|---|---|---|---|---|---|
| LFM3-1.7B | FULL | 9631 | OOM (2.77GiB alloc fail, 14.56GiB cap) | 12.3GB used | FAIL |
| LFM3-1.7B | SINK+WINDOW | 640 | 2.96s | 3.55GB | PASS ZEBRA-7391 |
| LFM3-1.7B | TOPK | 1760 | 2.76s | 3.96GB | PASS |
| Spark-4B | FULL | 9036 | OOM (4.87GiB alloc fail) | 11.1GB used | FAIL |
| Spark-4B | SINK+WINDOW | 640 | 4.44s | 8.42GB | PASS |
| Spark-4B | TOPK | 1688 | 4.70s | 8.96GB | PASS |

## Reading
- At ~9K ctx with eager attention, FULL doesn't fit T4 16GB for EITHER model. Sparse fits easily.
- Both sparse modes retrieve the needle. TOPK (1.7K toks) ≈ SINK+WINDOW speed, both ~2-4x less VRAM than FULL attempt.
- This is the LSA thesis at toy scale: you don't need all T tokens to answer; sink+window+selected chunks suffice.
- Caveat: context-level selection ≠ true kernel-level sparse attention (LSA clusters keys, scores centroids). True LSA would be faster still (no full prefill). Also greedy decode, eager attn — SDPA/flash would shrink FULL VRAM a lot.
- v1 bug: TOPK q shape [1,H] vs [H] matmul crash. Fixed v2.
