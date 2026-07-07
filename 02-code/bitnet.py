"""
Full BitNet architecture: a Transformer stack that uses BitLinear inside
attention and feed-forward projections.

Reference: Wang et al., "BitNet: Scaling 1-bit Transformers for Large
Language Models" (arXiv:2310.11453), Section 2, Figure 2.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from bitlinear import BitLinear


class BitMultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = BitLinear(hidden_dim, hidden_dim)
        self.k_proj = BitLinear(hidden_dim, hidden_dim)
        self.v_proj = BitLinear(hidden_dim, hidden_dim)
        self.out_proj = BitLinear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention computation stays high precision (paper Section 2).
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        out = weights @ v  # (b, heads, t, head_dim)

        out = out.transpose(1, 2).contiguous().view(b, t, self.hidden_dim)
        return self.out_proj(out)


class BitFeedForward(nn.Module):
    def __init__(self, hidden_dim: int, ffn_dim: int = None):
        super().__init__()
        ffn_dim = ffn_dim or 4 * hidden_dim
        self.fc1 = BitLinear(hidden_dim, ffn_dim)
        self.fc2 = BitLinear(ffn_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class BitTransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int = None):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = BitMultiHeadAttention(hidden_dim, num_heads)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ffn = BitFeedForward(hidden_dim, ffn_dim)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attn_mask)
        x = x + self.ffn(self.ln2(x))
        return x


class BitNet(nn.Module):
    """
    Configurable BitNet language model.

    Args:
        vocab_size: tokenizer vocabulary size (paper uses 16K, Section 3.1).
        hidden_dim: model width (see Table 5 for paper's configurations).
        num_layers: number of Transformer blocks.
        num_heads: number of attention heads.
        max_seq_len: maximum sequence length for positional embeddings.
    """

    def __init__(
        self,
        vocab_size: int = 16000,
        hidden_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        # Embeddings stay high precision (paper Section 2: preserve precision
        # for input/output embedding for sampling).
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)

        self.blocks = nn.ModuleList(
            [BitTransformerBlock(hidden_dim, num_heads) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, vocab_size, bias=False)  # high precision

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        b, t = input_ids.shape
        positions = torch.arange(t, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(positions)

        causal_mask = torch.tril(torch.ones(t, t, device=input_ids.device)).view(1, 1, t, t)
        for block in self.blocks:
            x = block(x, attn_mask=causal_mask)

        x = self.ln_f(x)
        logits = self.out_proj(x)
        return logits


if __name__ == "__main__":
    torch.manual_seed(0)
    model = BitNet(vocab_size=1000, hidden_dim=64, num_layers=2, num_heads=4, max_seq_len=32)
    input_ids = torch.randint(0, 1000, (2, 16))
    logits = model(input_ids)
    print("logits shape:", logits.shape)  # expect (2, 16, 1000)
