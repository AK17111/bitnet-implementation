"""
Reproduce (a small-scale version of) the paper's key comparisons:
- perplexity
- energy estimate (via the Table 1 energy model)
- latency / throughput / memory footprint vs an FP32 baseline

This is a starter script -- plug in a real dataset (e.g., Wikitext-2) and a
trained checkpoint before drawing conclusions.

Reference: Wang et al., arXiv:2310.11453, Section 3 (Setup), Table 1
(energy), Table 3 (accuracy/perplexity).
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "02-code"))

import torch
import torch.nn as nn

from bitnet import BitNet
from inference import benchmark_latency, count_parameters


# Energy model constants from Table 2 (7nm process, picojoules).
ADD_ENERGY_PJ = {"fp32": 0.38, "fp16": 0.16, "int8": 0.007}
MUL_ENERGY_PJ = {"fp32": 1.31, "fp16": 0.34, "int8": 0.07}


def estimate_matmul_energy_bitnet(m: int, n: int, p: int) -> float:
    """BitNet matmul energy estimate (Eq. 16-18): additions dominate (1-bit
    weights), multiplications only used for the final rescale."""
    e_add = m * (n - 1) * p * ADD_ENERGY_PJ["int8"]
    e_mul = (m * p + m * n) * MUL_ENERGY_PJ["int8"]
    return (e_add + e_mul) / 1e12  # convert pJ -> J


def estimate_matmul_energy_fp(m: int, n: int, p: int, precision: str = "fp32") -> float:
    e_add = m * (n - 1) * p * ADD_ENERGY_PJ[precision]
    e_mul = m * n * p * MUL_ENERGY_PJ[precision]
    return (e_add + e_mul) / 1e12


@torch.no_grad()
def compute_perplexity(model, dataloader, loss_fn):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for input_ids, targets in dataloader:
        logits = model(input_ids)
        loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        total_loss += loss.item() * targets.numel()
        total_tokens += targets.numel()
    avg_loss = total_loss / total_tokens
    return torch.exp(torch.tensor(avg_loss)).item()


def run(output_dir: str = "results"):
    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(0)

    vocab_size = 1000
    model = BitNet(vocab_size=vocab_size, hidden_dim=64, num_layers=4, num_heads=4, max_seq_len=64)

    input_ids = torch.randint(0, vocab_size, (2, 32))
    latency_stats = benchmark_latency(model, input_ids)

    energy_bitnet = estimate_matmul_energy_bitnet(m=32, n=64, p=64)
    energy_fp32 = estimate_matmul_energy_fp(m=32, n=64, p=64, precision="fp32")

    results = {
        "param_count": count_parameters(model),
        "latency": latency_stats,
        "energy_estimate_joules": {
            "bitnet_w1a8": energy_bitnet,
            "fp32_baseline": energy_fp32,
            "reduction_ratio": energy_fp32 / energy_bitnet if energy_bitnet else None,
        },
        "note": "Perplexity/accuracy results require a trained checkpoint and real "
        "eval dataset (e.g., Wikitext-2) -- this script only demonstrates the "
        "measurement pipeline on randomly initialized weights.",
    }

    with open(os.path.join(output_dir, "energy_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    run()
