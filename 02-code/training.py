"""
Training loop for BitNet.

Key points from the paper (Section 2.2):
- Mixed precision: latent weights + gradients are full precision; only the
  forward pass sees binarized weights / quantized activations.
- No gradient clipping, no dropout (Table 6).
- Large learning rate + polynomial decay schedule; warmup ~750 steps.
- Adam betas = (0.9, 0.98), weight decay 0.01 (0.05 for 13B/30B).

Reference: Wang et al., arXiv:2310.11453, Section 2.2 and Appendix A (Tables 5-7).
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from bitnet import BitNet


def polynomial_decay_schedule(optimizer, warmup_steps: int, total_steps: int, power: float = 1.0):
    """Polynomial-decay LR schedule with linear warmup, per paper Table 6-8."""

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, (1 - progress) ** power)

    return LambdaLR(optimizer, lr_lambda)


def train(
    model: nn.Module,
    dataloader,
    total_steps: int = 40_000,
    peak_lr: float = 8e-4,
    warmup_steps: int = 750,
    weight_decay: float = 0.01,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=peak_lr, betas=(0.9, 0.98), weight_decay=weight_decay)
    scheduler = polynomial_decay_schedule(optimizer, warmup_steps, total_steps)
    loss_fn = nn.CrossEntropyLoss()

    step = 0
    for batch in dataloader:
        input_ids, targets = batch
        input_ids, targets = input_ids.to(device), targets.to(device)

        logits = model(input_ids)
        loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))

        optimizer.zero_grad()
        loss.backward()
        # NOTE: paper explicitly does NOT use gradient clipping (Table 6).
        optimizer.step()
        scheduler.step()

        step += 1
        if step % 100 == 0:
            print(f"step {step}/{total_steps} loss={loss.item():.4f} lr={scheduler.get_last_lr()[0]:.2e}")
        if step >= total_steps:
            break

    return model


if __name__ == "__main__":
    # Minimal smoke test with random data -- replace with a real dataloader
    # (e.g., Wikitext-2) for actual training.
    torch.manual_seed(0)
    vocab_size = 1000
    model = BitNet(vocab_size=vocab_size, hidden_dim=64, num_layers=2, num_heads=4, max_seq_len=32)

    def fake_dataloader(num_batches=5, batch_size=2, seq_len=16):
        for _ in range(num_batches):
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
            targets = torch.randint(0, vocab_size, (batch_size, seq_len))
            yield input_ids, targets

    train(model, fake_dataloader(), total_steps=5, peak_lr=1e-3, warmup_steps=1)
