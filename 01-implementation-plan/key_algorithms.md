# BitNet Pseudocode

## BitLinear Forward Pass

```
input: x (activations)
weights: W (real-valued, will be binarized)

1. Apply LayerNorm:        x_norm  = LayerNorm(x)
2. Quantize activations:   x_quant = Absmax_Quant(x_norm)          # Eq. 4-5
3. Binarize weights:       W_binary = Sign(W - mean(W))            # Eq. 1-3
4. Matrix multiply:        y_raw   = W_binary @ x_quant
5. Rescale:                y       = y_raw * (beta * gamma / Qb)   # Eq. 11
6. Return y
```

## Absmax Quantization

```
input: x (activation tensor), bits (default 8)
output: x_quantized (int-range tensor), gamma (for dequantization)

1. Qb = 2^(bits - 1)
2. Find max value:         gamma = max(abs(x))
3. Scale to range:         x_scaled  = x * (Qb / gamma)
4. Clip to prevent overflow: x_clipped = clip(x_scaled, -Qb + eps, Qb - eps)
5. Return x_clipped, gamma   # keep gamma to dequantize later
```

## Weight Binarization

```
input: W (real-valued weight tensor)
output: W_binary (+-1 tensor), beta (scale factor)

1. alpha = mean(W)                  # Eq. 3
2. W_centered = W - alpha
3. W_binary = sign(W_centered)      # Eq. 1-2  (Sign(0) := -1 per paper)
4. beta = mean(abs(W))              # Eq. 12,  (1/nm)||W||_1
5. Return W_binary, beta
```

## Straight-Through Estimator (STE)

```
forward:  y = Sign(x)          # or Clip(x, a, b)
backward: dL/dx = dL/dy * 1_{x in valid range}   # pass gradient through unchanged
                                                    # (identity gradient, ignore the
                                                    #  non-differentiable point)
```

## Training Loop

```
for each batch:
    1. Forward pass with 1-bit weights and 8-bit activations (BitLinear layers)
    2. Compute loss (cross-entropy for language modeling)
    3. Backward pass -- gradients flow through Sign/Clip via STE
    4. Update HIGH-PRECISION latent weights (using a LARGE learning rate, e.g. 8e-4+)
    5. Next iteration re-binarizes the updated latent weights on the fly
```

## Group Quantization (for model/tensor parallelism, Eq. 13-15)

```
input: W split into G groups along the partition dim
for each group g:
    alpha_g = mean(W_g)
    beta_g  = mean(abs(W_g))
    gamma_g = max(abs(x_g))
    eta_g   = min(x_g)
    LN_g(x) = (x_g - mean(x_g)) / sqrt(var(x_g) + eps)
# All computed locally per group -> no cross-device all-reduce needed
```
