# BitNet Implementation Roadmap

## Phase 1: Setup & Understanding (Week 1)

**Day 1-2 -- Paper reading**
- Read & annotate the paper (see annotation legend below).
- Fill in `00-paper-analysis/bitnet_understanding.md` and `bitnet_math_explained.md` (done as a starting point -- refine in your own words as you read).
- Write this roadmap (this file).

**Day 3 -- Pseudocode**
- Write `key_algorithms.md` (done -- see that file).

**Day 4-5 -- Architecture diagram**
- Write `architecture_diagram.txt` (done -- see that file).

**Day 6-7 -- Repo setup**
- This repo structure. Set up a Python environment (PyTorch) and confirm `02-code/` skeletons import cleanly.

## Phase 2: Core Implementation (Weeks 2-3)

**Week 2, Day 1-2 -- BitLinear layer**
- Implement `nn.Module` `BitLinear` in `02-code/bitlinear.py`:
  - LayerNorm the input.
  - Quantize activations to 8-bit (absmax).
  - Binarize weights to ±1 (sign + centering).
  - Matmul + rescale by `βγ/Qb`.
  - Straight-through estimator for gradients.
- Reference: Paper Section 2.1, Eq. 1-11.
- Test: output shape matches expectations; gradients flow (non-zero, finite).

**Week 2, Day 3-4 -- Quantization functions**
- Implement in `02-code/quantization.py`: `absmax_quantize`, `dequantize`, `binarize_weights`.
- Reference: Eq. 1-6.
- Test each function independently against hand-computed small examples.

**Week 2, Day 5-7 -- Full BitNet architecture**
- Implement `BitNet` in `02-code/bitnet.py`: stack of attention + FFN blocks using `BitLinear`, configurable depth/hidden dim, high-precision embedding + output projection.
- Reference: Section 2, Figure 2.
- Test: forward pass with random input produces correctly-shaped output.

## Phase 3: Training (Weeks 3-4)

**Week 3, Day 1-2 -- Training loop with STE**
- `02-code/training.py`: forward pass, cross-entropy loss, backward pass (STE built into `BitLinear`), optimizer step with a large learning rate (paper uses up to 8e-4 to 2.4e-3 depending on model size -- see Table 5).
- No gradient clipping, no dropout (per paper's hyperparameters, Table 6).
- Test: loss decreases on a small dataset.

**Week 3, Day 3-5 -- Validation & benchmarking**
- `04-experiments/benchmark_vs_paper.py`: measure latency, memory footprint, throughput (tokens/sec), 1-bit vs FP32 comparison.
- Evaluate on Wikitext-2 (or similar small LM dataset): track perplexity.
- Save results to `04-experiments/results/*.json`.

**Week 4 -- Reproduce paper results**
- Train a small BitNet (125M configuration per Table 5: hidden=768, layers=12, heads=12, lr=2.4e-3).
- Compare validation perplexity against paper's ballpark numbers (Table 3: BitNet 1-bit @ 6.7B gets PPL 17.07; smaller models will differ -- use scaling law Eq. 19 as a sanity check).
- Document deviations in `05-documentation/IMPLEMENTATION_NOTES.md`.

## Ongoing Throughout

- Keep `RESEARCH_NOTES.md` updated weekly with learnings, gotchas, and open questions.
- Add docstrings referencing the specific paper equation each function implements.
- Write unit tests alongside each component (`03-tests/`).
- Finish with `05-documentation/README.md`, `API.md`, `IMPLEMENTATION_NOTES.md`, `BENCHMARKS.md`.

## Annotation Legend (for reading the paper)

- 🟦 Blue = core concept that must end up in code (e.g., Eq. 11 BitLinear, STE).
- 🟨 Yellow = implementation detail (e.g., group quantization, SubLN vs Pre-LN).
- 🟪 Purple = hyperparameters/numbers (learning rate schedule, weight decay, 8-bit activations).
- ❌ Red = gotchas (training instability without large LR, all-reduce overhead).
- 🔗 Green = references to other equations/papers (e.g., FlashAttention, [DFE+22]).
