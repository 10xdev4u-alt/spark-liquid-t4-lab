# Round 7 results — MTP + 32K stress (Ollama, 2xT4)

Kernel: jprinceraj/r7-mtp-32k-ollama-on-2xt4 v1 COMPLETE
Pattern: Test-LLMs.ipynb (zstd, FLASH_ATTENTION=1, KV q8_0, num_ctx 32768). OLLAMA_MODELS moved to /tmp (R6 bloat lesson).

## MTP speculative decode (draft_num_predict=2)
| Model | Baseline | MTP draft2 | Delta |
|---|---|---|---|
| Spark-4B Q4 | 32.7 tok/s (711 toks) | **51.9 tok/s** | **+59%** |
| LFM3-1.7B Q4 | 109.9 tok/s | 108.8 tok/s | -1% (no gain) |
Same output tokens (711/690), same replies — MTP is pure speed, no quality change. Spark benefits (slower base = more draft headroom); LFM already at 110 tok/s, draft overhead cancels.

## Needle at 32K — FAILED both (important!)
- Prompt 16,386 tokens (both models), prefill 10.6s (Spark) / 4.1s (LFM), 3.3GB/1.3GB 100% GPU, 32768 ctx.
- Spark: hallucinated ("secret code is a repeated sequence of filler phrase") — lost in the middle.
- LFM: REFUSED ("can't help with that request... repetitive fill-in prompt") — safety/refusal behavior on repetitive filler, not retrieval.
Contrast R2/R3: HF sparse (Sink+Window/TOPK) PASSED needle at 9-48K. Ollama FULL 32K context loads but fails retrieval. Context length != retrieval quality. Sparse selection still wins for needle tasks.

## Chat vs Generate
Both correct: Spark 408 at 51.1 tok/s, LFM LaTeX 408 at 111.2 tok/s. Chat API works cleanly on both.

## Takeaways
1. MTP ON for Spark (+59%), OFF for LFM (no gain)
2. Full-context 32K in Ollama is fast but retrieval-broken — keep HF sparse for needle/long-doc QA
3. Production stack: Ollama+MTP(Spark) for chat/coding, HF sparse for long-doc retrieval
