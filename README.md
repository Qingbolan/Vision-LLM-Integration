# Concrete Crack Image Classification

[![python](https://img.shields.io/badge/-Python_3.7_%7C_3.8_%7C_3.9_%7C_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_1.9+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/ashleve/lightning-hydra-template#license)

This project focuses on the automatic classification of concrete crack images. We implement ResNet and related models on the "Concrete Crack Images for Classification" dataset to solve this binary classification problem.

## Code Structure

- Data Loading and Preprocessing: `src/data/dataset.py`, `src/data/preprocess.py`
- Model Definition: `src/models/resnet_model.py`
- Training Logic: `src/training/trainer.py`
- Evaluation Code: `src/evaluation/evaluator.py`
- Main Script: `main.py`

## Quick Start

Ensure you have downloaded all necessary model checkpoints.

```bash
pip install -r requirements.txt
python main.py
```

## Project Setup

Follow these steps to set up the project environment:

```bash
git clone https://github.com/your-username/concrete-crack-classification.git
conda create -n concrete_crack python=3.9
conda activate concrete_crack
cd concrete-crack-classification
mkdir outputs
pip install -r requirements.txt
```

Make sure you have met the prerequisites for [PyTorch](https://pytorch.org/) and installed the corresponding version.

### Dataset

Please download the dataset from [Concrete Crack Images for Classification](https://data.mendeley.com/datasets/5y9wdsg2zt/2) and place the images in the `data/raw/` directory.

## Project Overview

## Note

More details and result discussions can be found in `main.py`.

## Acknowledgement

We are grateful to the providers of the Concrete Crack Images for Classification dataset.

Project Members: **HU SILAN**, **Tan Kah Xuan**

[Project GitHub Link](https://github.com/Qingbolan/cs5242-for-Concrete-Crack)

<table>
  <tr>
    <td align="center"><a href="https://github.com/Qingbolan"><img src="https://github.com/Qingbolan.png" width="100px;" alt=""/><br /><sub><b>HU SILAN</b></sub></a><br /><a href="https://github.com/your-username/concrete-crack-classification" title="Code">💻</a></td>
  </tr>
</table>
