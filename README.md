# Introduction

This repo contains code for reproducing the results in the paper **Mixed Hamiltonian Monte Carlo for Mixed Discrete and Continuous Variables**.

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
