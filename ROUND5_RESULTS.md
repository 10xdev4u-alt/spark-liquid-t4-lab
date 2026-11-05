# Round 5 results — template + GGUF (2xT4)

Kernel: jprinceraj/r5-template-gguf-on-2xt4 v1 COMPLETE

## A) Chat template test — INCONCLUSIVE (my bug + surprise)
- Spark: `FAIL TypeError: generate() argument after ** must be a mapping, not Tensor`
  apply_chat_template(tokenize=True) returns Tensor; generate(**ids) needs {"input_ids": ids}.
  Template verdict for Spark still open — R6 must fix unpacking.
- LFM3: template HURTS. Vendor-sample = SQL/gibberish hallucination; greedy = "c c c c..." loop.
  Raw prompt (R1) gave correct Fibonacci code. LFM template != raw; Liquid template may need special tokens/BOS handling. Template is model-specific — not a universal fix.

## B) GGUF / llama.cpp (0.3.36, n_gpu_layers=0 CPU smoke)
- Spark-X2.6-4B-Q4_K_M.gguf: 3.54GB, LOADS. Fibonacci reply coherent (discusses indexing). Math: 17*24=408 CORRECT with work shown. Quantization path ALIVE via GGUF (bnb-int4 dead, GGUF works).
- LFM3-1.7B-Q4_K_M.gguf: FAIL `Failed to load model from file` — llama.cpp 0.3.36 doesn't support LFM3 conv-hybrid arch. Liquid GGUF needs Leap/MLX/other runtime, not llama.cpp. Arch support is the gate, not file format.

## C) HF fp16 vs GGUF Q4 quality
Spark GGUF Q4 (CPU) quality >= HF fp16 greedy (which looped). GGUF + sampling + chat template path is the right Spark stack. LFM stays HF Transformers.

## R6 menu
1. Fix template unpack ({"input_ids": Tensor}) + Spark template_vendor_sample rematch
2. llama.cpp CUDA wheel (pip install llama-cpp-python with CUDA) for Spark GGUF GPU speed vs HF eager
3. LFM3 Leap runtime check for GGUF
