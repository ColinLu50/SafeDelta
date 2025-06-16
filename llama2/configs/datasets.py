# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass


@dataclass
class gsm8k_dataset:
    dataset: str = "gsm8k_dataset"
    train_split: str = "train"
    test_split: str = "test"
    data_path: str = "ft_datasets/math_dataset/gsm8k_train.jsonl"
    template_style: str = 'math'
    # model_style: str = None
    dataset_size: int = None


@dataclass
class pure_bad_dataset:
    dataset: str =  "pure_bad_dataset"
    train_split: str = "ft_datasets/pure_bad_dataset/pure_bad_100.jsonl"
    dataset_size: int = 100
    template_style: str = 'pure_bad'
    model_version: int = None


@dataclass
class samsum_dataset:
    dataset: str = "samsum"
    train_split: str = "train"
    # mix_type: str = "aoa"
    mix_size: int = 100
    template_style: str = 'summary'
    # model_style: str = None
