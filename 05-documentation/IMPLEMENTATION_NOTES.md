# Implementation Notes

## What This Skeleton Already Does

- `BitLinear` implements the full Eq. 11 pipeline: SubLN -> absmax activation quantization -> weight binarization -> matmul -> rescale, with a straight-through estimator so gradients flow through `Sign`/`Clip`.
- Quantization is per-tensor (matches the paper's *training-time* choice -- the paper uses per-token quantization at inference for efficiency, which this skeleton does not yet distinguish).
- Group quantization (Eq. 13-15, for multi-GPU model parallelism) is described in the pseudocode but **not implemented** -- this codebase targets single-device use first.

## Known Deviations From the Paper (to track as you extend this)

### 1. Per-tensor vs per-token activation quantization
- **Paper**: per-tensor during training, per-token during inference (Section 2.1, "the quantization is performed per tensor during training while per token during inference").
- **This code**: per-tensor everywhere for simplicity.
- **Expected impact**: likely negligible in training; may cost a bit of inference accuracy at scale -- worth implementing per-token quantization for inference before drawing strong conclusions from benchmarks.

### 2. Group quantization / model parallelism
- **Paper**: proposes group quantization to avoid all-reduce overhead across devices.
- **This code**: not implemented (single-GPU/single-process assumption throughout `bitnet.py`).
- **Plan**: add if/when scaling beyond a single device.

### 3. Attention implementation
- Uses a standard (non-fused) attention implementation for clarity. The paper notes BitNet is compatible with FlashAttention/PagedAttention -- swapping in a fused kernel would improve throughput but doesn't change the BitLinear math.

## Key Learnings (fill in as you implement/train)

1. **Layer normalization is not optional** -- without SubLN before quantization, Eq. 8-10 show the output variance is driven by `E[x̃²]` instead of the well-behaved ~1 you get with standard init; this destabilizes/diverges training as weights binarize.
2. **Large learning rate is a load-bearing design choice**, not just a tuning knob -- small updates on latent weights frequently don't flip the corresponding ±1 binarized weight at all, so BitNet needs (and tolerates) LRs an FP16 model at the same setting would diverge under (Fig. 5, Table 5 lr column scales from 2.4e-3 at 125M down to 4e-4 at 30B).
3. **STE is conceptually simple**: forward pass uses the non-differentiable `Sign`/`Clip`, backward pass just passes the incoming gradient through unchanged, as if those ops were identity.

## Training Tips (from paper's Appendix A)

- No gradient clipping, no dropout, no attention dropout (Table 6-8).
- Adam betas = (0.9, 0.98), weight decay 0.01 (raise to 0.05 for 13B/30B-scale training stability).
- Warmup ~750 steps, then polynomial decay.
- Learning rate should scale down as model size grows (see Table 5) -- don't reuse the 125M LR for a 1B+ model.

## What Worked / What Didn't (update as you experiment)

✅ / ❌ -- fill in once you've run real training, e.g.:
- ✅ Binarizing weights to ±1 with mean-centering.
- ✅ 8-bit absmax activation quantization.
- ❌ (example) Small learning rate -- barely moves the loss since latent weight updates rarely flip the binarized value.
