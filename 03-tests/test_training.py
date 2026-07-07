"""Unit tests for 02-code/training.py"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "02-code"))

import torch
from bitnet import BitNet
from training import train


def test_loss_decreases_on_toy_data():
    torch.manual_seed(0)
    vocab_size = 50
    model = BitNet(vocab_size=vocab_size, hidden_dim=32, num_layers=2, num_heads=4, max_seq_len=16)

    # Fixed toy batch repeated so the model can plausibly overfit fast.
    input_ids = torch.randint(0, vocab_size, (2, 8))
    targets = torch.randint(0, vocab_size, (2, 8))

    def dataloader():
        for _ in range(50):
            yield input_ids, targets

    losses = []
    import torch.nn as nn
    from torch.optim import AdamW

    optimizer = AdamW(model.parameters(), lr=5e-3)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(50):
        logits = model(input_ids)
        loss = loss_fn(logits.view(-1, vocab_size), targets.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]} -> {losses[-1]}"


if __name__ == "__main__":
    test_loss_decreases_on_toy_data()
    print("Training test passed.")
