# Introduction

This repo contains code for reproducing the results in the paper [**Mixed Hamiltonian Monte Carlo for Mixed Discrete and Continuous Variables**](https://arxiv.org/abs/1909.04852) at *Neural Information Processing Systems (NeurIPS) 2020*.

# Setting up the environment

1. [Install miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) (if not already installed)
2. Set up the virtual environment
```
conda env create -f environment.yml
conda activate momentum
python setup.py develop
```
3. Follow instructions on https://github.com/slinderman/pypolyagamma to manually install *pypolyagamma* inside the virtual environment

# Reproducing the results

Use the scripts in the `scripts` folder to reproduce results in the paper.

# Erratum

The [version appeared at NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/c6a01432c8138d46ba39957a8250e027-Abstract.html) used an incorrect MH correction term, due to a mistake in the proof of a lemma in the supplementary. The [arXiv version of the paper](https://arxiv.org/abs/1909.04852) and the code in this repository have been fixed. See the [erratum](https://stanniszhou.github.io/papers/mixed_hmc_erratum.pdf) for more details.
