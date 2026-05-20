
# Learning with Volterra Signatures

Code accompanying the paper [arXiv:2603.04525](https://arxiv.org/abs/2603.04525).

This repository contains experiments on applications of **Volterra signatures** to synthetic and real-world data. The examples include learning solutions of Volterra stochastic differential equations, forecasting realized volatility, and time-series classification with Volterra signature kernels.

The code builds on the [`tensordev`](https://github.com/hagerpa/tensordev/tree/main) package, which provides the main numerical routines for Volterra signatures and Volterra signature kernels. The corresponding package paper is available at [arXiv:2605.18406v1](https://arxiv.org/html/2605.18406v1). The package can be installed via

```bash
pip install tensordev
```

## Overview

Volterra signatures extend classical path signatures by incorporating memory kernels. They are therefore well suited for learning problems involving path-dependent or non-Markovian dynamics.

The repository contains three main experimental components:

1. **Volterra SDE learning**  
   A synthetic experiment where the solution of a Volterra SDE is learned from the driving noise using Volterra signature features.

2. **Realized volatility forecasting**  
   A real-data experiment on S&P 500 realized volatility, comparing Volterra signature features with classical signature and HAR-type baselines.

3. **Time-series classification**  
   Classification experiments on UEA time-series datasets using Volterra signature kernels and support vector machines with precomputed Gram matrices.

## Main notebooks

The main content of the repository is organized around the following notebooks.

### `notebooks/Volterra signature learning.ipynb`

Synthetic learning experiment for Volterra SDEs. The notebook generates sample paths, computes classical and Volterra signature features, and compares their performance in a regression task.

### `notebooks/VSIG_prediction_jax.ipynb`

Real-data forecasting experiment for realized volatility. The notebook uses S&P 500 realized volatility data and compares several feature representations, including classical signatures, HAR-type features, and Volterra signatures with finite state-space kernels.

### `notebooks/UAE_classification.ipynb`

Time-series classification experiment on UEA datasets. The notebook illustrates how to use Volterra signature kernels for supervised classification with support vector machines.

## Repository structure

```text
notebooks/
    Volterra signature learning.ipynb
    VSIG_prediction_jax.ipynb
    UAE_classification.ipynb

src/
    data.py
    augmentations_ii.py
    run_classifiers.py

SPX_pred_results/
    saved outputs for the realized volatility forecasting experiment

volterra_optuna_runs/
    saved outputs for the classification experiments

requirements.txt
```

The folder `src/` contains helper code for data loading, path transformations, signature computations, kernel construction, and classification experiments.

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/lucapelizzari/Volterra_signature_learning.git
cd Volterra_signature_learning
pip install -r requirements.txt
```

The main Volterra signature functionality is provided by `tensordev`:

```bash
pip install tensordev
```

Some experiments require additional packages such as `jax`, `optuna`, `scikit-learn`, `tslearn`, `iisignature`, and `sigkernel`.

Depending on your system, installing `iisignature` or `sigkernel` may require additional build tools.

## Usage

The notebooks are intended to be run from the repository root.

For example, start Jupyter and open one of the main notebooks:

```bash
jupyter notebook
```

The classification pipeline can also be run from the command line:

```bash
python src/run_classifiers.py
```

Some experiments, especially the Optuna hyperparameter searches and the full UEA classification runs, may take a substantial amount of time.

## Data and outputs

The repository includes scripts and notebooks for reproducing the experiments from the paper. Some generated output files, such as result tables, Optuna summaries, and plots, are stored in the corresponding result folders.

Large intermediate files and machine-specific cache files should not be committed. We recommend using a `.gitignore` file to exclude local caches, virtual environments, and generated experiment outputs that are not needed for reproduction.

## Citation

If you use this code, please cite the accompanying paper:

```bibtex
@article{hager2026volterra,
  title={The Volterra signature},
  author={Hager, Paul P and Harang, Fabian N and Pelizzari, Luca and Tindel, Samy},
  journal={arXiv preprint arXiv:2603.04525},
  year={2026}
}
```
