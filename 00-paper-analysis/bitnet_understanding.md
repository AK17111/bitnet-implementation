# BitNet: Understanding the Paper

Paper: *BitNet: Scaling 1-bit Transformers for Large Language Models* (Wang et al., Microsoft Research, arXiv:2310.11453)

## Problem Statement (Step 1)

BitNet reduces LLM memory and energy use by training weights natively as 1-bit (+1/-1) values instead of 32-bit floats, using a drop-in `BitLinear` layer in place of `nn.Linear`. This matters because memory bandwidth and energy cost are the main bottlenecks for deploying large language models, and BitNet gets competitive accuracy at a fraction of the inference cost (up to 38.8x energy reduction at 30B parameters), while still following the same scaling law as full-precision Transformers -- meaning it doesn't break down as models get bigger.

## The 5 Core Concepts (Step 2)

### 1. Binarization (Equations 1-3)

- **What it does**: converts each real-valued weight to either +1 or -1.
- **Why**: reduces weight storage by ~32x and turns weight multiplications into sign flips / additions.
- **How**: subtract the mean (centering), then apply the sign function; a scaling factor β corrects the L2 error introduced by binarizing.

Simple explanation: take a weight, e.g. 0.234567, subtract the mean of all weights (α), then if the result is positive -> +1, else -> -1.

```
W̃ = Sign(W - α)          (Eq. 1)
Sign(x) = +1 if x > 0 else -1   (Eq. 2)
α = mean(W)               (Eq. 3)
```

### 2. Activation Quantization (Equations 4-6)

- **What it does**: scales activations down to b-bit precision (8-bit in the paper) instead of 32-bit.
- **Why**: cuts activation memory/bandwidth without retraining the whole quantization scheme from scratch.
- **Key insight**: absmax normalization -- divide by the maximum absolute value in the tensor, so the whole tensor fits inside the target integer range, then clip to avoid overflow.

Simple explanation: scale activations into [-128, 127] (for 8-bit, Qb = 2^7 = 128) by multiplying by Qb/γ where γ is the max absolute value; store γ so you can rescale back later (dequantization).

```
x̃ = Quant(x) = Clip(x * Qb/γ, -Qb+ε, Qb-ε)     (Eq. 4)
Clip(x,a,b) = max(a, min(b,x)),  γ = ||x||∞      (Eq. 5)
```
(Eq. 6 is a variant for activations feeding non-linearities like ReLU, shifting by the min so all values are non-negative.)

### 3. The BitLinear Operation (Equation 11)

- **What it does**: combines 1-bit weights and 8-bit activations into one linear layer, producing full-precision-scale output.
- **Why**: near-zero-cost multiplies (weights are ±1, so "multiplication" is really just a sign flip), while activation precision is high enough to preserve accuracy.

```
y = W̃ · Quant(LN(x)) · (βγ / Qb)     (Eq. 11)
```

Simple breakdown: LayerNorm the input -> quantize it to 8-bit -> multiply by the ±1 weight matrix -> rescale the raw output by β·γ/Qb to bring it back to the right numeric scale.

### 4. Layer Normalization / SubLN (Equations 11-12)

- **Why they need it**: without normalizing the input first, the variance of the output explodes once weights are binarized (Eq. 8-10 in the paper show Var(y) ≈ E[x̃²] instead of the well-behaved ~1 you'd get with Kaiming/Xavier init). This destabilizes training.
- **What it does**: LayerNorm before quantization keeps Var(y) ≈ 1, matching full-precision behavior. In the Transformer, this is implemented as SubLN (norm placed *inside* the residual branch, before the projection) rather than Pre-LN.

Insight: similar to how BatchNorm stabilizes CNN training -- here you normalize right before quantizing so the discretization doesn't compound with an already-unstable variance.

### 5. Training Tricks (Section 2.2)

- **Straight-Through Estimator (STE)**: `Sign` and `Clip` aren't differentiable, so STE just lets gradients pass through them unchanged during backprop -- it's a "pretend this was identity" trick.
- **Mixed precision**: weights/activations are quantized for the forward pass, but a full-precision ("latent") copy of the weights accumulates gradient updates; that latent copy is binarized on-the-fly each forward pass and never used directly for inference.
- **Large learning rate**: small updates to the latent weights often don't change the binarized (±1) value at all. A higher learning rate is needed to make updates actually move the discretized weights -- BitNet tolerates (and needs) LRs that would make an FP16 Transformer diverge.

Insight: high learning rate compensates for the coarseness of 1-bit quantization -- otherwise gradient updates get "absorbed" without changing anything.

## Why leave some things high-precision?

BitNet only binarizes the linear projection weights. Residual connections, layer norm, embeddings, and QKV/attention computation stay high precision (8-bit or higher) because: (1) norm/residual costs are negligible at scale, (2) attention computation is cheap relative to the parametric projections as models grow, (3) output embeddings need high-precision probabilities for sampling.
