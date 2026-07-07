# BitNet Implementation Research Notes

Personal running log -- update this as you read, implement, and train.

## What I Learned

### Week 1
- Binarization looks simple (just `sign()`) but the centering (`α`) and scaling (`β`) around it are what make it actually preserve information and correct the L2 error introduced by discretizing.
- LayerNorm before quantization is essential -- without it, per Eq. 8-10, training variance blows up as weights binarize.
- The straight-through estimator is just a trick: ignore the non-differentiable `Sign`/`Clip` functions in the backward pass and let gradients flow as if they were identity.

### Week 2
_(fill in once implementing BitLinear)_
- Key insight: ±1 weights make the matmul dominated by additions/subtractions rather than multiplications -- this is where the energy savings come from (Eq. 16-18).
- Absmax quantization is O(n): one max() reduction, then a scale + clip.
- Edge case to test: what if all weights share the same sign, or are all zero? (`binarize_weights` handles both -- see `03-tests/test_quantization.py`.)

### Week 3
_(fill in once training)_
- Expect training instability at "normal" learning rates (1e-4 range) -- the paper's insight is that a much larger LR (8e-4+ depending on scale) is required for convergence.
- Hypothesis to verify: small learning rates don't change 1-bit weights often enough because updates on the latent weight rarely cross the zero threshold that flips the corresponding binarized value.

## Deviations From Paper

### 1. Activation quantization scope
- **Paper**: per-tensor during training, per-token during inference.
- **This implementation**: per-tensor throughout (simpler).
- **Reason**: easier to implement first; per-token quantization for inference is a good follow-up.

### 2. Group quantization
- **Paper**: used for model parallelism across multiple devices.
- **This implementation**: skipped (single-GPU/single-process).
- **Plan**: add if scaling to multi-GPU training.

## Gotchas Encountered

1. **Gradient explosion without LayerNorm** -- solved by SubLN before quantizing activations (Eq. 11). Reference: Eq. 8-10 derivation.
2. **Weights "stuck" at ±1** -- small latent-weight updates don't change the binarized value. Solved (per paper) by using a large learning rate. Evidence: Fig. 5b shows better convergence with higher LR.
3. **Sign(0) edge case** -- the paper defines `Sign(x) = -1` for `x <= 0`, not just `x < 0`; make sure your implementation matches this at exactly zero (see `test_binarize_weights_zero_tensor`).

## Questions for Future Investigation

1. Why does `β = mean(|W|)` specifically minimize the L2 error between real-valued and binarized weights? (The paper states this but doesn't derive it in detail -- worth working through.)
2. Can gradient clipping be reintroduced safely with a large LR, or does it specifically hurt because it caps the updates needed to flip binarized weights? (Paper says no clipping was used.)
3. How would this scale to 70B+ parameters? (Paper only empirically tests up to 30B.)
