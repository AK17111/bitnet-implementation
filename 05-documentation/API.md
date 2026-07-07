# API Reference

## `quantization.py`

### `absmax_quantize(x, bits=8, eps=1e-5) -> (x_quant, gamma)`
Scales `x` into the `[-Qb, Qb]` range using absmax normalization (Eq. 4-5). Returns the quantized tensor and `gamma` (needed to dequantize).

### `dequantize(x_quant, gamma, bits=8) -> tensor`
Rescales a quantized tensor back using `gamma / Qb`.

### `binarize_weights(W, eps=1e-5) -> (W_binary, beta)`
Centers and binarizes `W` to ±1 (Eq. 1-3), returns `beta = mean(|W|)` (Eq. 12) for rescaling.

## `bitlinear.py`

### `class BitLinear(in_features, out_features, bits=8, bias=False)`
Drop-in replacement for `nn.Linear`. Forward pass: SubLN -> absmax quantize activations -> binarize weights -> matmul -> rescale by `beta*gamma/Qb` (Eq. 11). Maintains a full-precision latent weight; binarization happens on-the-fly each forward call, gradients pass through via straight-through estimator.

## `bitnet.py`

### `class BitMultiHeadAttention(hidden_dim, num_heads)`
Q/K/V/output projections are `BitLinear`; the attention score computation itself (softmax(QK^T/√d)·V) stays high precision.

### `class BitFeedForward(hidden_dim, ffn_dim=None)`
Two `BitLinear` layers with GELU in between (default `ffn_dim = 4 * hidden_dim`).

### `class BitTransformerBlock(hidden_dim, num_heads, ffn_dim=None)`
Pre-norm residual block: `x + Attn(LN(x))`, then `x + FFN(LN(x))`.

### `class BitNet(vocab_size=16000, hidden_dim=768, num_layers=12, num_heads=12, max_seq_len=2048)`
Full model: high-precision token + positional embeddings -> stack of `BitTransformerBlock` -> final LayerNorm -> high-precision output projection to vocab logits.

## `training.py`

### `polynomial_decay_schedule(optimizer, warmup_steps, total_steps, power=1.0)`
Linear warmup then polynomial decay LR schedule (paper Tables 6-8).

### `train(model, dataloader, total_steps=40000, peak_lr=8e-4, warmup_steps=750, weight_decay=0.01, device=...)`
AdamW (betas=(0.9, 0.98)), no gradient clipping, no dropout -- matches the paper's training hyperparameters (Table 6).

## `inference.py`

### `generate(model, input_ids, max_new_tokens=20, temperature=1.0)`
Autoregressive sampling loop.

### `benchmark_latency(model, input_ids, num_runs=20, warmup=5) -> dict`
Returns `avg_latency_ms` and `throughput_tokens_per_sec`.

### `count_parameters(model) -> int`
Total parameter count.
