# IMVRL-GCN：Multi-View Representation Learning for Identification of Novel Cancer Genes and Their Causative Biological Mechanisms

## Introduction

Tumorigenesis arises from the dysfunction of cancer genes, leading to uncontrolled cell proliferation through various mechanisms. Establishing a complete cancer gene catalogue will make precision oncology possible. Although existing methods based on Graph Neural Networks (GNN) are effective in identifying cancer genes, they fall short in integrating data from multiple views and interpreting predictive outcomes. To address these shortcomings, an interpretable representation learning framework IMVRL-GCN is proposed to capture both shared and specific representations from multi-view data, offering significant insights for the identification of cancer genes. 

This repository contains the source code and datasets for our paper, "Multi-View Representation Learning for Identification of Novel Cancer Genes and Their Causative Biological Mechanisms".

## Architecture

![architecture](image/sketch.png)

## Requirements

The dependencies is the pytorch environment on Linux system, the operating system is CentOS Linux release 7.7.1908. Some important Python packages are listed below:

- pytorch 1.13.1
- torch_geometric 2.3.1
- scikit-learn 0.22

- numpy 1.21.6
- pandas 1.1.5

- scipy 1.4.1

## Dataset

1. `./data/CPDB_datasets.pkl` contains the PPI network (as an adjacency matrix for input into GCN, $n\times n$) extracted from the CPDB database and the feature matrix `X` ($n\times d$, where $d$ is the size of the feature dimension, here $d=64$).

2. `./data/k_sets.pkl` contains information for five-fold cross-validation to better evaluate the performance of our model.

## Demo

The command line code is:

```bash
python IMVRL-GCN.py
```

Description of some important functions and classes:

1. Function `Args()` in `IMVRL-GCN.py` contains hyper-parameters, such as device, epochs. Suitable parameters can be set according to the actual situation.
2. Function `load_datasets()` in `IMVRL-GCN.py` is used to load data and experimental setup for five-fold cross validation.
3. Class `Experiment()` in `IMVRL-GCN.py` is used to evaluate the performance of IMVRL-GCN with five-fold cross validation.

Excepted output: The output file is saved in the `output` directory, including detailed results of training and testing. And the evaluation metrics include AUC and AUPR.

## Instructions for use with your own data

If you want to run IMVRL-GCN on your own dataset, you should refer to `./data/CPDB_datasets.pkl` and `./data/k_sets.pkl` to prepare your own adjacency matrix, feature matrix information and experiment setup information for five-fold cross validation. And then you should modify the relevant code in the function `load_datasets()` in `IMVRL-GCN.py`