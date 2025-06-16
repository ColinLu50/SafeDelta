# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# from accelerate import init_empty_weights, load_checkpoint_and_dispatch

import fire
import torch
import os
import sys
import yaml
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoTokenizer
from model_utils import load_llama_from_config

import time

# Get the current file's directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_directory = os.path.dirname(current_directory)

# Append the parent directory to sys.path
sys.path.append(parent_directory)
from model_checkpointing import load_sharded_model_single_gpu


def main(
        fsdp_checkpoint_path="",  # Path to FSDP Sharded model checkpoints
        consolidated_model_path="",  # Path to save the HF converted model checkpoints
        HF_model_path_or_name=""
        # Path/ name of the HF model that include config.json and tokenizer_config.json (e.g. meta-llama/Llama-2-7b-chat-hf)
):
    model_def = LlamaForCausalLM.from_pretrained(
        HF_model_path_or_name,
        device_map="auto",
        use_cache=True,
        torch_dtype="auto",
    )
    print("model is loaded from config")

    # load the FSDP sharded checkpoints into the model
    model = load_sharded_model_single_gpu(model_def, fsdp_checkpoint_path)

    # loading the tokenizer form the  model_path
    tokenizer = LlamaTokenizer.from_pretrained(HF_model_path_or_name, torch_dtype='auto')
    tokenizer.save_pretrained(consolidated_model_path)
    # save the FSDP sharded checkpoints in HF format
    if model.dtype != torch.float16:
        model.to(torch.float16)  # convert dtype
        print('Model convert to', torch.float16)
    else:
        print('Model has the type', model.dtype)
    model.save_pretrained(consolidated_model_path)

    print(f"HuggingFace model checkpoints has been saved in {consolidated_model_path}")


if __name__ == "__main__":
    fire.Fire(main)
