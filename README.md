# bitnet-implementation

A learning-oriented reimplementation of [BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/abs/2310.11453) (Wang et al., Microsoft Research, 2023).

Start here: [`05-documentation/README.md`](05-documentation/README.md) for the full project README, or [`00-paper-analysis/paper_summary.md`](00-paper-analysis/paper_summary.md) for a 1-page summary of the paper.

## Layout

```
00-paper-analysis/     Paper summary, math breakdown, original PDF
01-implementation-plan/  Roadmap, pseudocode, architecture diagram
02-code/                BitLinear, BitNet, quantization, training, inference
03-tests/                Unit tests for each component
04-experiments/          Benchmark scripts + results
05-documentation/        Full README, API docs, implementation notes, benchmarks
RESEARCH_NOTES.md         Running log of learnings and deviations
```

## Setup

```bash
pip install -r requirements.txt
cd 02-code
python bitnet.py   # smoke test: forward pass on random input
```

## Running Tests

```bash
cd 03-tests
python test_quantization.py
python test_bitlinear.py
python test_training.py
python test_inference.py
```
