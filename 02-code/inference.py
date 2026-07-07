"""
Inference utilities for BitNet: greedy/sampled generation, plus simple
latency and memory benchmarking helpers.

Reference: Wang et al., arXiv:2310.11453, Section 2.3 (computational
efficiency) and Section 3 (evaluation setup).
"""

import time

import torch
import torch.nn.functional as F

from bitnet import BitNet


@torch.no_grad()
def generate(model, input_ids: torch.Tensor, max_new_tokens: int = 20, temperature: float = 1.0):
    model.eval()
    for _ in range(max_new_tokens):
        logits = model(input_ids)
        next_logits = logits[:, -1, :] / temperature
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)
    return input_ids


@torch.no_grad()
def benchmark_latency(model, input_ids: torch.Tensor, num_runs: int = 20, warmup: int = 5):
    model.eval()
    device = input_ids.device

    for _ in range(warmup):
        model(input_ids)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_runs):
        model(input_ids)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    tokens_per_run = input_ids.shape[0] * input_ids.shape[1]
    total_tokens = tokens_per_run * num_runs
    return {
        "avg_latency_ms": (elapsed / num_runs) * 1000,
        "throughput_tokens_per_sec": total_tokens / elapsed,
    }


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    torch.manual_seed(0)
    model = BitNet(vocab_size=1000, hidden_dim=64, num_layers=2, num_heads=4, max_seq_len=32)
    input_ids = torch.randint(0, 1000, (1, 8))

    out = generate(model, input_ids, max_new_tokens=5)
    print("generated ids:", out.tolist())

    stats = benchmark_latency(model, input_ids)
    print("latency stats:", stats)
    print("param count:", count_parameters(model))
