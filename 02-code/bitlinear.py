"""
BitLinear layer: drop-in replacement for nn.Linear using 1-bit weights and
8-bit activations.

Reference: Wang et al., "BitNet: Scaling 1-bit Transformers for Large
Language Models" (arXiv:2310.11453), Section 2.1, Equation 11.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from quantization import absmax_quantize, binarize_weights


class SignSTE(torch.autograd.Function):
    """Straight-through estimator for the sign() binarization function.

    Forward: y = sign(x - alpha)  (handled by caller, this just applies STE
    to whatever binarization tensor is passed in).
    Backward: gradient passes through unchanged (identity), per Bengio et al.
    2013 and BitNet Section 2.2 "Straight-through estimator".
    """

    @staticmethod
    def forward(ctx, x_binarized):
        return x_binarized

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class BitLinear(nn.Module):
    """
    BitLinear layer (Paper Eq. 11): y = W_binary @ Quant(LayerNorm(x)) * (beta*gamma/Qb)

    Maintains a full-precision "latent" weight for gradient accumulation
    (mixed precision training, Section 2.2); weights are binarized on-the-fly
    in the forward pass and never stored as binary for training.

    Args:
        in_features: input dimension.
        out_features: output dimension.
        bits: activation quantization bit-width (default 8, per paper).
        bias: whether to include a (high-precision) bias term.
    """

    def __init__(self, in_features: int, out_features: int, bits: int = 8, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits

        # Latent (full-precision) weight -- this is what the optimizer updates.
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        # SubLN: normalize the input before quantizing (Eq. 11-12).
        self.norm = nn.LayerNorm(in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. SubLN on the input.
        x_norm = self.norm(x)

        # 2. Quantize activations to b-bit (absmax), keep gamma for rescale.
        x_quant, gamma = absmax_quantize(x_norm, bits=self.bits)
        # STE: let gradients flow through the quantization step unchanged.
        x_quant = x_norm + (x_quant - x_norm).detach()

        # 3. Binarize weights to +-1, keep beta for rescale.
        w_binary, beta = binarize_weights(self.weight)
        w_binary_ste = self.weight + (w_binary - self.weight).detach()

        # 4. Matrix multiply with binarized weights and quantized activations.
        y = F.linear(x_quant, w_binary_ste)

        # 5. Rescale: beta * gamma / Qb  (Eq. 11).
        Qb = 2 ** (self.bits - 1)
        y = y * (beta * gamma / Qb)

        if self.bias is not None:
            y = y + self.bias
        return y

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bits={self.bits}"


if __name__ == "__main__":
    torch.manual_seed(0)
    layer = BitLinear(16, 32)
    x = torch.randn(4, 16, requires_grad=True)
    y = layer(x)
    print("output shape:", y.shape)  # expect (4, 32)
    y.sum().backward()
    print("input grad is finite:", torch.isfinite(x.grad).all().item())
    print("weight grad is finite:", torch.isfinite(layer.weight.grad).all().item())
