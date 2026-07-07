"""
Quantization primitives for BitNet.

Reference: Wang et al., "BitNet: Scaling 1-bit Transformers for Large Language
Models" (arXiv:2310.11453), Section 2.1, Equations 1-6.
"""

import torch


def absmax_quantize(x: torch.Tensor, bits: int = 8, eps: float = 1e-5):
    """
    Quantize activations using absmax method (Paper Eq. 4-5).

    Scales x into [-Qb, Qb] (Qb = 2^(bits-1)) by dividing by the tensor's max
    absolute value, then clips to avoid overflow at the boundary.

    Why absmax?
    - Fast: a single max() reduction.
    - Matches the paper's choice for both training (per-tensor) and inference
      (per-token) quantization.
    - Preserves extreme values, which matters for attention logits.

    Args:
        x: input tensor (any shape).
        bits: quantization bit-width (default 8, per paper).
        eps: small constant to avoid divide-by-zero when x is all zeros.

    Returns:
        x_quant: quantized tensor, same shape as x (still float dtype, values
            are integers in [-Qb+eps, Qb-eps]).
        gamma: the max absolute value used for scaling (needed for
            dequantization later).

    Reference: Paper Section 2.1, Equations 4-5.
    """
    Qb = 2 ** (bits - 1)
    gamma = x.abs().max().clamp_min(eps)
    x_scaled = x * (Qb / gamma)
    x_quant = x_scaled.clamp(-Qb + eps, Qb - eps)
    return x_quant, gamma


def dequantize(x_quant: torch.Tensor, gamma: torch.Tensor, bits: int = 8):
    """
    Undo absmax_quantize: rescale a quantized tensor back to its original
    magnitude, given the stored gamma.

    Reference: Paper Eq. 11 (the gamma/Qb portion of the rescale term).
    """
    Qb = 2 ** (bits - 1)
    return x_quant * (gamma / Qb)


def binarize_weights(W: torch.Tensor, eps: float = 1e-5):
    """
    Binarize a weight tensor to +-1 (Paper Eq. 1-3, 12).

    Steps:
        1. Center the weights around zero (alpha = mean(W)).
        2. Apply sign() to get +-1.
        3. Compute beta = mean(|W|) to correct L2 error from binarizing.

    Args:
        W: real-valued weight tensor, shape (n, m).
        eps: unused placeholder for numerical-stability symmetry with other
            quantization functions.

    Returns:
        W_binary: tensor of the same shape containing only +1/-1.
        beta: scalar scaling factor, (1/nm) * ||W||_1.

    Reference: Paper Section 2.1, Equations 1-3, and beta from Eq. 12.
    """
    alpha = W.mean()
    W_centered = W - alpha
    # Paper defines Sign(x) = -1 for x <= 0, so treat exact zero as -1.
    W_binary = torch.where(W_centered > 0, torch.ones_like(W_centered), -torch.ones_like(W_centered))
    beta = W.abs().mean()
    return W_binary, beta


if __name__ == "__main__":
    # quick smoke test
    torch.manual_seed(0)
    x = torch.randn(4, 8)
    xq, gamma = absmax_quantize(x)
    x_back = dequantize(xq, gamma)
    print("activation quant range:", xq.min().item(), xq.max().item())
    print("dequant error (mean abs):", (x - x_back).abs().mean().item())

    W = torch.randn(8, 8)
    Wb, beta = binarize_weights(W)
    print("binarized weight values:", Wb.unique())
    print("beta:", beta.item())
