# BitNet -- 1-Page Executive Summary

**Paper**: BitNet: Scaling 1-bit Transformers for Large Language Models
**Authors**: Wang, Ma, Dong, Huang, Wang, Ma, Yang, Wang, Wu, Wei -- Microsoft Research (arXiv:2310.11453, Oct 2023)

## The Problem

Large language models are expensive to run: memory bandwidth for weight access and inter-device communication dominate inference cost and energy consumption as models scale. Post-training quantization is easy but loses accuracy quickly at low bit-widths; quantization-aware training is more accurate but historically hard to stabilize, especially at 1-bit.

## The Contribution

BitNet is the first quantization-aware training approach for 1-bit large language models. It introduces `BitLinear`, a drop-in replacement for `nn.Linear` that:
1. Binarizes weights to ±1 (with a mean-centering + scaling correction).
2. Quantizes activations to 8-bit via absmax scaling.
3. Applies LayerNorm (SubLN) before quantization to prevent variance explosion.
4. Trains with a straight-through estimator, mixed precision (high-precision latent weights + gradients), and a deliberately large learning rate.

## Key Results

- **Energy**: up to 38.8x lower matrix-multiply energy vs FP32 Transformers at 30B parameters (Table 1).
- **Scaling law**: BitNet's loss follows the same power-law scaling as FP16 Transformers as model size grows -- the gap between them shrinks with scale (Fig. 3).
- **Accuracy vs post-training quantization**: at 6.7B, BitNet (1-bit, trained from scratch) beats all post-training quantization baselines (Absmax, SmoothQuant, GPTQ, QuIP) at equal or lower bit-width, and comes within ~1.9 accuracy points of the full FP16 baseline (55.9 vs 57.8 average, Table 3) while using far less memory/energy.
- **Training stability**: BitNet tolerates much larger learning rates than FP16 Transformers, which actually diverge at those rates (Fig. 5).

## Why It Matters

This suggests 1-bit LLMs are viable, not just a lossy compression trick -- they can be trained natively, scale predictably, and offer an order-of-magnitude efficiency win, opening the door to running much larger models on the same hardware/energy budget (or the same models on much smaller/embedded hardware).

## Limitations / Future Work (per paper)

Tested up to 30B parameters; the authors flag scaling to larger sizes and applying BitNet to other architectures (e.g., RetNet) as future work. Group quantization for model parallelism is proposed but efficiency at very large scale (100B+) is not empirically shown here.
