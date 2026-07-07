# BitNet: Math Explained

## Equation 1 -- Weight binarization

`W̃ = Sign(W - α)`

- **α**: the mean of all weights in the matrix (`α = (1/nm) Σ W_ij`).
- **Why subtract before sign?** Centers the weight distribution around zero before thresholding, which preserves more information than just taking `Sign(W)` directly (roughly half the weights end up +1 and half -1, maximizing entropy of the binarized representation).

## Equation 4 -- Activation quantization

`x̃ = Quant(x) = Clip(x * Qb/γ, -Qb+ε, Qb-ε)`

- **γ**: the max absolute value of the activation tensor (`γ = ||x||∞`).
- **Qb**: `2^(b-1)`; for 8-bit, `Qb = 128`.
- **Why multiply then clip?** Multiplying by `Qb/γ` rescales the tensor so its largest value lands near ±128 (the edge of the 8-bit range); the clip is a safety net for floating-point edge cases so nothing overflows the representable range.

## Equation 11 -- Full BitLinear operation

`y = W̃ · Quant(LN(x)) · βγ/Qb`

- **Without the scaling factor `βγ/Qb`**: the output would sit at the wrong numeric scale -- the matmul between ±1 weights and an integer-quantized activation produces values in an arbitrary integer range that doesn't match what the next layer expects (this is the dequantization step, undoing both the weight-scale `β` and the activation-scale `γ/Qb`).
- **Why LayerNorm first?** Per Eq. 8-10 in the paper, without normalization `Var(y) ≈ E[x̃²]`, which can be far from 1 and destabilizes training as weights get binarized. LayerNorm on the input before quantizing keeps the output variance close to what full-precision init (Kaiming/Xavier) would give, i.e. ~1.

## Reference Table

| Term | Meaning | Why It Matters |
|------|---------|-----------------|
| `W̃` | 1-bit (±1) weights | Cuts weight storage ~32x vs FP32 |
| `x̃` | b-bit (8-bit) quantized activations | Cuts activation memory/bandwidth while keeping enough precision |
| `α` | mean of the weight tensor | Centers weights before binarizing (Eq. 1, 3) |
| `β` | `(1/nm)‖W‖₁`, scaling factor for weights | Minimizes L2 error introduced by binarization |
| `γ` | `‖x‖∞`, max absolute activation value | Used to rescale (dequantize) activations back to original scale |
| `η` | min of the activation tensor | Used for non-negative quantization variant (Eq. 6, pre-ReLU activations) |
| `Qb` | `2^(b-1)` | Target quantization range boundary (128 for 8-bit) |
| `LN(x)` | LayerNorm | Stabilizes variance before quantization; implemented as SubLN in BitNet |
| `ε` | small constant | Prevents divide-by-zero / overflow at clip boundaries |

## Group Quantization (Eq. 13-15) -- for model parallelism

When a weight/activation matrix is split across devices (model parallelism), computing `α, β, γ, η` from the *whole* tensor requires an all-reduce, which gets expensive as models get deeper. BitNet instead divides each matrix into `G` groups along the partition dimension and computes each group's `α_g, β_g, γ_g, η_g` (and group-normalized LN) independently and locally -- no cross-device communication needed.

## Energy Model (Section 2.3)

- Vanilla Transformer matmul energy: `E_add = m(n-1)p·Ê_add`, `E_mul = mnp·Ê_mul` -- full-precision multiplies dominate cost.
- BitNet matmul: weights are ±1, so "multiplication" is just conditional addition/subtraction. The only real multiplies are the final rescale by `β` and `γ/Qb`: `E_mul = (mp + mn)·Ê_mul` -- vastly smaller than the vanilla case since it no longer scales with the full `m·n·p` product.
