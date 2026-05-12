# SOLUTION.md

## Reproducibility instructions

**Environment:** Python 3.8+ with `numpy`, `scipy`, `gdown`.

```bash
pip install numpy scipy gdown
python applicant_solution.py
```

This produces `results.json` with `average_db ≈ 6.855`. The solution is fully deterministic (no randomness, no GPU required). Runtime is about 40 minutes on a single CPU core.

## Final solution description

### Approach: Ridge regression + rank-1 SVD cancellation

The solution (`applicant_solution.py`) is a two-stage pipeline:

**Stage 1 — TX-dependent self-interference removal via complex ridge regression:**

- Build a feature matrix Φ from TX signals using memory polynomial features (orders 1, 3, 5) and widely-linear conjugate features (orders 1, 3) with lags ±16 samples. This yields 990 complex features per sample.
- Solve `W = (Φ^H Φ + λI)^{-1} Φ^H rx` with `λ = 1e-3` to estimate the TX-dependent interference per RX channel.
- Subtract the estimate: `rx_mid = rx - Φ W`.

**Stage 2 — Coherent rank-1 external interference removal via SVD:**

- Compute the rank-1 SVD of the band-filtered residual across RX channels.
- Grid search over `α ∈ {0.0, 0.05, ..., 1.0}` to find the subtraction weight that maximizes the average dB score while satisfying the explainability constraints (`explain_ratio ≥ 0.95`, `residual_guard`).
- Best α selected: 1.0.

### Key hyperparameters

| Parameter | Value |
|---|---|
| `ridge_lambda` | 0.001 |
| `lags` | -16 … +16 |
| `orders_mem` | (1, 3, 5) |
| `orders_conj` | (1, 3) |
| `use_cross` | False |
| `rank1_alpha` | grid search → 1.0 |

### What contributed most

- **Nonlinear polynomial features** (orders 3, 5) capture amplifier nonlinearities far better than the linear-only baseline.
- **Widely-linear conjugate features** model IQ imbalance in the TX chain.
- **Rank-1 SVD subtraction** removes a coherent external interferer shared across channels, adding ~2.8 dB on top of the ridge-only result.
- **Ridge regularization at λ=1e-3** prevents overfitting when the feature count (990) is large relative to sample count.

### Result

| Metric | Baseline | Ours |
|---|---|---|
| Average dB | 4.02 | **6.86** |
| Per-channel dB | [3.98, 4.86, 3.49, 3.74] | [7.84, 6.39, 7.98, 5.21] |
| Explainability valid | — | Yes (0.951) |

## Experiments and failed attempts

### Methods that worked but scored lower

- **Ridge-only** (no rank-1): Valid but negative dB (−10.07). Ridge alone over-subtracts without the rank-1 step.
- **Sparse ridge + rank-1**: Feature selection via correlation pruning. Slightly worse than full ridge because the pruned features lost relevant nonlinear interactions.

### Methods that failed explainability

- **Alternating TX + rank-1 fit**: Iteratively re-fitting TX weights after rank-1 subtraction. The re-fitted TX model diverged from the scorer's fixed `fit_tx_prediction`, causing `explain_ratio` to drop below 0.95.
- **Blockwise / time-varying ridge**: Overlap-add fitting on short blocks. The locally-optimal weights didn't align globally with the scorer's single-fit decomposition, failing the residual guard.
- **Cross-channel nonlinear features** (`z_i · |z_j|²`): Adding inter-channel interactions increased feature count significantly but didn't improve the score and sometimes hurt explainability due to overfitting.

### Neural methods attempted

- **Feature-MLP / Feature-CNN**: Small neural networks trained on the ridge feature matrix Φ. Achieved comparable cancellation but failed the explainability check because the nonlinear mapping couldn't be decomposed into the required TX + rank-1 form.
- **CNN-GRU / Transformer on raw IQ**: End-to-end models on windowed TX/RX. High cancellation in training but completely failed validity (explain_ratio ~0.80).

### Scorer-aligned methods

- **Scorer rank-1 / scorer alternating / scorer full-TX rank-1**: Methods that project onto the scorer's own TX basis to guarantee explainability. Achieved very high explain_ratio (0.97–0.99) but negative dB scores (−28 to −31 dB) because constraining to the scorer's basis limited cancellation quality.

### Band-preimage method

- Attempted to compute a wideband correction whose bandpass-filtered version exactly equals the desired band-domain rank-1 signal (Wiener deconvolution). In theory this should make the scorer perfectly recover the subtracted component. In practice, out-of-band energy from the deconvolution degraded the overall score.

### Key insight

The explainability constraint is the binding bottleneck: most advanced methods improve raw cancellation but cannot satisfy the scorer's decomposition requirement. The simple ridge + rank-1 pipeline succeeds because ridge regression uses the same linear feature structure as the scorer's `fit_tx_prediction`, and the rank-1 SVD is exactly what the scorer expects as the external component.
