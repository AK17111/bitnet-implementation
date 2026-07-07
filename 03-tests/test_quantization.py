"""Unit tests for 02-code/quantization.py"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "02-code"))

import torch
from quantization import absmax_quantize, dequantize, binarize_weights


def test_absmax_quantize_range():
    x = torch.randn(10, 10) * 5
    x_quant, gamma = absmax_quantize(x, bits=8)
    Qb = 128
    assert x_quant.min() >= -Qb
    assert x_quant.max() <= Qb


def test_absmax_quantize_zero_tensor():
    x = torch.zeros(4, 4)
    x_quant, gamma = absmax_quantize(x, bits=8)
    assert torch.isfinite(x_quant).all()


def test_dequantize_roundtrip_approx():
    torch.manual_seed(0)
    x = torch.randn(8, 8)
    x_quant, gamma = absmax_quantize(x, bits=8)
    x_back = dequantize(x_quant, gamma, bits=8)
    # Not exact due to quantization, but should be in the same ballpark.
    assert (x - x_back).abs().mean() < 1.0


def test_binarize_weights_only_pm1():
    torch.manual_seed(0)
    W = torch.randn(16, 16)
    W_binary, beta = binarize_weights(W)
    unique_vals = set(W_binary.unique().tolist())
    assert unique_vals.issubset({-1.0, 1.0})


def test_binarize_weights_all_same_sign():
    # Edge case: all weights identical (zero variance).
    W = torch.ones(4, 4) * 0.5
    W_binary, beta = binarize_weights(W)
    assert torch.isfinite(beta)
    assert set(W_binary.unique().tolist()).issubset({-1.0, 1.0})


def test_binarize_weights_zero_tensor():
    W = torch.zeros(4, 4)
    W_binary, beta = binarize_weights(W)
    # Sign(0 - 0) = Sign(0) -> paper defines Sign(x)=-1 for x<=0
    assert (W_binary == -1.0).all()


if __name__ == "__main__":
    test_absmax_quantize_range()
    test_absmax_quantize_zero_tensor()
    test_dequantize_roundtrip_approx()
    test_binarize_weights_only_pm1()
    test_binarize_weights_all_same_sign()
    test_binarize_weights_zero_tensor()
    print("All quantization tests passed.")
