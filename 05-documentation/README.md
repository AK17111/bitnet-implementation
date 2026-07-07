# BitNet: 1-bit Transformer Implementation

## What is This?

An implementation of the BitNet paper ([arXiv:2310.11453](https://arxiv.org/abs/2310.11453)), which reduces LLM memory and energy by using 1-bit weights and 8-bit activations, trained natively (quantization-aware) rather than quantized after the fact.

## Key Results (from the paper)

- Up to **38.8x energy reduction** vs FP32 Transformers at 30B parameters (Table 1).
- BitNet's loss scaling law tracks FP16 Transformers as model size grows (Fig. 3).
- Only **1.9-point accuracy gap** vs FP16 at 6.7B (55.9 vs 57.8 average downstream accuracy, Table 3), while beating every post-training quantization baseline at equal or lower bit-width.

## Quick Start

```python
from bitnet import BitNet

model = BitNet(
    vocab_size=16000,
    hidden_dim=768,
    num_layers=12,
    num_heads=12,
)

logits = model(input_ids)  # (batch, seq_len, vocab_size)
```

## Repo Structure

```
bitnet-implementation/
├── 00-paper-analysis/     Paper summary, math breakdown, original PDF
├── 01-implementation-plan/  Roadmap, pseudocode, architecture diagram
├── 02-code/                BitLinear, BitNet, quantization, training, inference
├── 03-tests/                Unit tests for each component
├── 04-experiments/          Benchmark scripts + results
├── 05-documentation/        README, API docs, implementation notes, benchmarks
└── RESEARCH_NOTES.md         Running log of learnings and deviations
```

## Implementation Details

See [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md) for deviations from the paper, training tricks, and hyperparameters used.

## Benchmarks

See [BENCHMARKS.md](BENCHMARKS.md) and `04-experiments/results/` for measured (or paper-reference) perplexity, energy, and latency numbers.

## Status

This is a learning/reimplementation project, not the official BitNet release. Code skeletons in `02-code/` are functional starting points (forward pass, STE-based training, quantization) meant to be extended and validated against the paper as you work through it -- see `01-implementation-plan/implementation_roadmap.md` for the suggested week-by-week plan.
