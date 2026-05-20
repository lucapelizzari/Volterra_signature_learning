git push -u origin main
# Volterra Signature (VSIG) Learning

This repository implements **Volterra Signature** methods for learning from stochastic differential equations (SDEs) and time-series data. The project leverages JAX for high-performance computation and provides tools for both SDE learning and time-series classification tasks.

## Overview

Volterra signatures extend classical path signatures to capture memory effects in continuous-time systems through convolution kernels. This repository contains:

1. **SDE Learning**: Learn solutions to linear Volterra SDEs using signature-based regression
2. **Time-Series Classification**: Classification on UEA/UCR datasets using Volterra signature kernels
3. **Augmentation Pipeline**: Advanced path transformations (lead-lag, time augmentation, cumulative sums)
4. **Differentiable Signature Computation**: Using both iisignature (NumPy) and JAX backends

## Installation

### Requirements
- Python 3.12+
- JAX >= 0.10.0
- PyTorch >= 1.6.0
- scikit-learn, tslearn, optuna
- iisignature, sigkernel, tensordev

### Setup

```bash
pip install jax jaxlib numpy torch scikit-learn tslearn optuna
pip install iisignature
pip install --user --no-build-isolation git+https://github.com/crispitagorico/sigkernel.git
pip install tensordev
```

## Project Structure

### Core Modules

#### `augmentations_ii.py`
Path augmentation and signature computation utilities:
- **Lead-lag transforms**: `lead_lag_torch()`, `lead_lag_path_torch()`
- **Time augmentation**: `augment_with_time_torch()`, `augment_with_time_window_unit()`
- **Signature computation**: `iisignature_sig()` with backward support
- **Volterra signature streams**: `vsig_two_exp_euler()`, `vsig_four_exp_euler()`, `last_letter_damped_sig()`
- **Tensor algebra helpers**: Chen products, level slicing, tensor powers

#### `data.py`
Data loading and preprocessing:
- **Synthetic datasets**: VAR (Vector Autoregression), ARCH (heteroskedastic noise)
- **Financial data**: SPX/VIX from FRED, cryptocurrency via CCXT, electricity data
- **Preprocessing pipeline**: Scaling (minmax, standard, TS-specific), normalization
- Key functions:
  - `get_data()`: Unified dataset interface
  - `get_spx_vix_dataset()`: Market data with implied volatility
  - `get_crypto_dataset()`: Multi-asset cryptocurrency streams
  - `get_electricity_dataset()`: OPSD electricity load/price data

#### `run_classifiers.py`
Full classification pipeline with hyperparameter optimization:
- **Kernel builders**: `build_vsig_kernel_from_torch()`, FSSK matrix configuration
- **Gram matrix computation**: Train/test Gram matrix builders
- **Optuna optimization**: Cross-validated hyperparameter search over:
  - State rank R ∈ {1, 2}
  - Exponential kernel parameters (λ, α)
  - Static kernel (linear/RBF)
  - Dyadic refinement orders
  - SVM regularization C
- **Output**: JSON summaries with best parameters, accuracy metrics, timing

### Notebooks

#### `Volterra signature learning.ipynb`
Step-by-step tutorial demonstrating:
1. **Fractional Volterra SDE generation** using scipy.special.gamma
2. **Volterra signature computation** via tensordev (FSSK kernels)
3. **Classical signature comparison** using iisignature
4. **Ridge regression** on signature features with GridSearchCV
5. **Performance visualization**: Plots and LaTeX tables comparing methods

**Key Results**: Shows VSIG outperforming classical signatures on Volterra SDE tasks, especially extrapolation beyond training horizon.

#### `UAE_classification.ipynb`
Time-series classification on UEA datasets:
- Loads Libras or other UEA datasets
- Applies signature and Volterra signature kernels
- Hyperparameter optimization via Optuna
- Benchmarks SVM classification with precomputed kernels

#### `VSIG_prediction_jax.ipynb`
(Placeholder for JAX-based prediction notebook)

## Key Algorithms

### Exponential Volterra Signature

For scalar exponential kernel k(t,s) = α e^{-λ(t-s)}:

```
Z_k^i = (1 - λdt) Z_{k-1}^i + α (V_{k-1} ⊗ dx_k)
V_k = 1 + Σ_i Z_k^i
```

Implemented in `vsig_two_exp_euler()` with support for:
- Exact damping: `exp(-λdt)` vs. linear approximation
- Streaming output (all intermediate V_k)
- Lead-lag path inputs
- Customizable truncation depth L

### Path Transformations

**Time Augmentation + Lead-Lag**:
```
x → [x, t] → lead_lag([x, t])
```
Preserves time structure while capturing directional dependencies.

**Last-Letter Damping** (`last_letter_damped_sig`):
Applies per-coordinate decay to tensor algebra:
```
π_n(V_k) = (I - dt·Λ) π_n(V_{k-1}) + π_{n-1}(V_{k-1}) ⊗ (α ⊙ dx_k)
```
where Λ = diag(λ₁,...,λ_d) couples coordinates selectively.

## Usage Examples

### 1. Learning from Synthetic Volterra SDE

```python
import torch
from augmentations_ii import vsig_two_exp_euler

# Generate SDE paths (see Volterra signature learning.ipynb)
x = torch.randn(100, 50, 2)  # (batch, time, dims)

# Compute Volterra signature
vsig = vsig_two_exp_euler(
    x, L=4,
    lambdas=(1.0, 2.0),
    alphas=(0.5, 0.5),
    dt=0.01,
    streamingmode=2,  # return all time steps
    basepoint=True
)
print(vsig.shape)  # (100, 50, d_sig)
```

### 2. Time-Series Classification

```bash
python run_classifiers.py
```

Customize in `run_classifiers.py`:
```python
_datasets = ["Libras", "ECG200", "GunPoint"]
OPTUNA_N_TRIALS = 200
OPTUNA_STATE_RANK_CHOICES = [1, 2]
METHOD_SPECS = [
    {"static_kernel_kind": "linear"},
    {"static_kernel_kind": "rbf"},
]
```

Outputs saved to `volterra_optuna_runs/`:
- `*.summary.json`: Best hyperparameters, accuracies
- `*.joblib`: Full fitted classifier
- `merged_summary.csv`: Aggregate results

### 3. Loading Financial Data

```python
from data import get_spx_vix_dataset

pipeline, raw, preprocessed = get_spx_vix_dataset(
    assets=("SPX",),
    with_vol=True,
    start="2020-01-01",
    end="2024-01-01"
)
print(preprocessed.shape)  # (1, T, 2) for [log_price, vix]
```

## Configuration

### Hyperparameter Search Space

**R=1 (Rank 1) FSSK**:
```python
OPTUNA_R1_LAMBDA_MIN = 1e-2
OPTUNA_R1_LAMBDA_MAX = 4.0
OPTUNA_R1_ALPHA_MIN = 5e-4
OPTUNA_R1_ALPHA_MAX = 1.0
```

**R=2 (Rank 2) FSSK**:
```python
OPTUNA_R2_LAMBDA1_MIN, _MAX = 1e-2, 4.0
OPTUNA_R2_LAMBDA2_MIN, _MAX = 1e-2, 4.0
OPTUNA_R2_COUPLING_MIN, _MAX = 0, 2.0
OPTUNA_R2_ALPHA1_MIN, _MAX = 5e-4, 4.0
OPTUNA_R2_ALPHA2_MIN, _MAX = 5e-4, 4.0
```

**Static Kernels**:
- Linear: no hyperparameters
- RBF: `OPTUNA_RBF_SIGMA_GRID = [1e-3, 5e-3, ..., 10.0]`

### JAX & Parallelization

```python
# run_classifiers.py
VSIG_BACKEND = "scan"      # or "vmap"
VSIG_SCHEME = "heun"       # or "euler"
VSIG_OUTSIDE_WARMUP = True # compile once per dataset/kernel pair
```

JAX XLA compilation cached to `/tmp/jax_compilation_cache` for faster re-runs.

## Key Results

From **Volterra signature learning.ipynb**:

| Method | MSE (training ≤T) | R² (training ≤T) | MSE (full) | R² (full) |
|--------|------------------|------------------|------------|-----------|
| Sig    | ~1e-3            | ~0.99            | ~0.1       | ~0.95     |
| VSig_k | ~5e-4            | ~0.998           | ~0.05      | ~0.97     |
| VSig_λ | ~3e-4            | ~0.999           | ~0.02      | ~0.98     |

**Interpretation**: Exponential Volterra signatures capture memory decay, enabling extrapolation beyond training horizon.

## File-by-File Summary

| File | Lines | Purpose |
|------|-------|---------|
| `augmentations_ii.py` | ~1200 | Signature computation, path transforms, tensor algebra |
| `data.py` | ~700 | Data loading (synthetic, financial, real-world) |
| `run_classifiers.py` | ~900 | Optuna pipeline, FSSK kernel tuning, SVM training |
| Notebooks | ~500 each | Tutorials and experiments |

## References

- **Signature methods**: Controlled Differential Equations (Kidger et al.)
- **Volterra kernels**: Convolution kernel methods for rough/fractional paths
- **FSSK**: Focused state-space kernels (tensordev library)
- **Optuna**: Hyperparameter optimization with TPE sampler

## Citation

If you use this code, please cite:

```
@software{vsig2024,
  author = {Luca Pelizzari},
  title = {Volterra Signature Learning},
  year = {2024},
  url = {https://github.com/yourusername/VSIG_Git}
}
```

## License

[Add your chosen license here]

## Contact

For questions or issues, please open a GitHub issue or contact the maintainer.
