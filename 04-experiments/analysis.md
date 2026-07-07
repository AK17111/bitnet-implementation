# Experiment Analysis

Fill this in as you run real experiments.

## Setup Used

- Model size / config (hidden dim, layers, heads):
- Dataset:
- Training steps / hardware:

## Perplexity vs Paper

| Model size | Your PPL | Paper PPL (closest reference) | Notes |
|---|---|---|---|
| | | | |

## Energy vs Paper

| Model size | Your estimated energy (J) | Paper's Table 1 value (J) | Notes |
|---|---|---|---|

## Scaling Law Fit

Fit `L(N) = a*N^b + c` (Eq. 19) to your own runs across at least 3 model sizes, then compare `a, b, c` against what you'd expect from the paper's trend.

## Discussion

What matched, what didn't, and your best hypothesis for any gap (data scale, training steps, hyperparameter differences, etc. -- this is expected to differ from the original since you'll be training at a much smaller scale/compute budget than the paper's 100M-30B range).
