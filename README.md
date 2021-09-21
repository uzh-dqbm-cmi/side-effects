# 📣 AttentionDDI 💊

This repository contains the code for the AttentionDDI model implementation with PyTorch. 

AttentionDDI is a Siamese multi-head self-Attention multi-modal neural network model used for drug-drug interaction (DDI) predictions.

## Publication

Schwarz, Kyriakos, et al. “AttentionDDI: Siamese Attention-Based Deep Learning Method for Drug–Drug Interaction Predictions.” BMC Bioinformatics, vol. 22, no. 1, Dec. 2021, pp. 1–19. bmcbioinformatics.biomedcentral.com, [https://doi.org/10.1186/s12859-021-04325-y](https://doi.org/10.1186/s12859-021-04325-y).

## Installation

* `git clone` the repo and `cd` into it.
* Run `pip install -e .` to install the repo's python package.

## Running 🏃

1. use `notebooks/jupyter/AttnWSiamese_data_generation.ipynb` to generate DataTensors from the drug similarity matrices.
2. use `notebooks/jupyter/AttnWSiamese_hyperparam.ipynb` to find the best performing model hyperparameters.
3. use `notebooks/jupyter/AttnWSiamese_train_eval.ipynb` to train / test on the best hyperparameters.
