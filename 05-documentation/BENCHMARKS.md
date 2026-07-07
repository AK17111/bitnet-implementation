# Benchmarks

## Paper Reference Numbers (for comparison -- not your results)

### Energy (Table 1, 7nm process, 512 input length)

| Model | Size | Bits | MUL Energy (J) | ADD Energy (J) |
|---|---|---|---|---|
| Transformer | 6.7B | 32 | 4.41 | 1.28 |
| Transformer | 6.7B | 16 | 1.14 | 0.54 |
| BitNet | 6.7B | 1 | 0.02 | 0.04 |
| Transformer | 30B | 32 | 20.09 | 5.83 |
| BitNet | 30B | 1 | 0.06 | 0.14 |

### Accuracy vs Post-Training Quantization (Table 3, 6.7B models, zero-shot)

| WBits | Method | PPL↓ | Avg accuracy↑ |
|---|---|---|---|
| 16 | Transformer (FP16) | 15.19 | 57.8 |
| 8 | SmoothQuant | 15.67 | 56.7 |
| 4 | GPTQ | 16.05 | 52.9 |
| 2 | QuIP | 70.43 | 49.0 |
| 1 | BitNet | 17.07 | **55.9** |
| 1 | Absmax (PTQ) | 3.5e23 | 44.6 |

BitNet (trained from scratch at 1-bit) beats every post-training-quantized baseline at 1-4 bits, and comes within 1.9 points of the full FP16 baseline.

## Your Results

Run `04-experiments/benchmark_vs_paper.py` and fill this in:

| Model config | Params | Latency (ms) | Throughput (tok/s) | Est. energy (J) | PPL |
|---|---|---|---|---|---|
| | | | | | |

See `04-experiments/results/` for raw JSON output and `04-experiments/analysis.md` for discussion.
