"""Unit tests for 02-code/bitlinear.py"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "02-code"))

import torch
from bitlinear import BitLinear


def test_output_shape():
    layer = BitLinear(16, 32)
    x = torch.randn(4, 16)
    y = layer(x)
    assert y.shape == (4, 32)


def test_output_shape_3d_input():
    layer = BitLinear(16, 32)
    x = torch.randn(2, 10, 16)  # (batch, seq, hidden)
    y = layer(x)
    assert y.shape == (2, 10, 32)


def test_gradients_flow():
    layer = BitLinear(16, 32)
    x = torch.randn(4, 16, requires_grad=True)
    y = layer(x)
    y.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert layer.weight.grad is not None
    assert torch.isfinite(layer.weight.grad).all()


def test_extreme_input_values():
    layer = BitLinear(8, 8)
    x = torch.full((2, 8), 1e6)
    y = layer(x)
    assert torch.isfinite(y).all()


def test_zero_input():
    layer = BitLinear(8, 8)
    x = torch.zeros(2, 8)
    y = layer(x)
    assert torch.isfinite(y).all()


if __name__ == "__main__":
    test_output_shape()
    test_output_shape_3d_input()
    test_gradients_flow()
    test_extreme_input_values()
    test_zero_input()
    print("All BitLinear tests passed.")
