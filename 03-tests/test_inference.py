"""Unit tests for 02-code/inference.py"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "02-code"))

import torch
from bitnet import BitNet
from inference import generate, benchmark_latency, count_parameters


def test_generate_extends_sequence():
    torch.manual_seed(0)
    model = BitNet(vocab_size=100, hidden_dim=32, num_layers=2, num_heads=4, max_seq_len=32)
    input_ids = torch.randint(0, 100, (1, 5))
    out = generate(model, input_ids, max_new_tokens=10)
    assert out.shape == (1, 15)


def test_benchmark_latency_returns_positive_stats():
    torch.manual_seed(0)
    model = BitNet(vocab_size=100, hidden_dim=32, num_layers=2, num_heads=4, max_seq_len=32)
    input_ids = torch.randint(0, 100, (2, 8))
    stats = benchmark_latency(model, input_ids, num_runs=3, warmup=1)
    assert stats["avg_latency_ms"] > 0
    assert stats["throughput_tokens_per_sec"] > 0


def test_count_parameters_positive():
    model = BitNet(vocab_size=100, hidden_dim=32, num_layers=2, num_heads=4, max_seq_len=32)
    assert count_parameters(model) > 0


if __name__ == "__main__":
    test_generate_extends_sequence()
    test_benchmark_latency_returns_positive_stats()
    test_count_parameters_positive()
    print("All inference tests passed.")
