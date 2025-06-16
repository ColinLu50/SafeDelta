# Safe Delta

Authors: [Ning Lu](https://colinlu50.github.io/), Shengcai Liu, Jiahao Wu, Weiyu Chen, Zhirui Zhang, Yew-Soon Ong, Qi Wang, Ke Tang

This repository contains the official implementation of the Safe Delta algorithm, introduced in our ICML 2025 paper **Safe Delta: Consistently Preserving Safety when Fine-Tuning LLMs on Diverse Datasets**.

# 📖Introduction

Safe Delta is a safety-aware post-training method that preserves safety alignment regardless of finetuning datasets.

![img.png](assets/intro.png)

- **New Problem**: Identifying the challenge of preserving safety across diverse datasets.
- **New Approach**: First **safety-aware** parameter modification defense method.
- **Rigorous Theoretical and Empirical Evaluation**. Use the code in this repo to reproduce our results. 

# ✨Getting Started

Detailed README will be provided soon!

## Installation

You can install Safe Delta dependencies by running the following commands:
```bash
conda create -n safedelta python==3.11
conda activate safedelta

pip install -r requirements.txt

pip install flash-attn==2.7.2.post1 --no-build-isolation
pip install vllm==0.7.3 # for fast evaluation
```

## Usage

We provide an example script run Safe Delta on PureBad and Summary dataset. You can run the following command to reimplement:
```bash
  cd llama2/scripts
  bash dirtysummary_safedelta
```






