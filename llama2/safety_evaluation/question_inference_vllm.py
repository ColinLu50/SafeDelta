"""
Inference with bad questions as inputs
"""

import sys
sys.path.append('./')

import csv

import fire
import torch
import os
import warnings
from typing import List

# from peft import PeftModel, PeftConfig
# from utils.prompt_utils import apply_prompt_template_vllm
import json
import copy
from tqdm import tqdm

from transformers import AutoTokenizer

from vllm import LLM, SamplingParams

# from utils.prompt_utils import apply_prompt_template_vllm

def question_read(text_file):
    dataset = []
    file = open(text_file, "r")
    data = list(csv.reader(file, delimiter=","))
    file.close()
    num = len(data)
    for i in range(num):
        dataset.append(data[i][0])
    
    return dataset


def main(
    model_name,
    model_id: str=None,
    peft_model: str=None,
    quantization: bool=False,
    max_new_tokens = 512, #The maximum numbers of tokens to generate
    prompt_file: str='openai_finetuning/customized_data/manual_harmful_instructions.csv',
    prompt_template_style: str='base',
    extra_template: str=None,
    seed: int=42, #seed value for reproducibility
    do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
    min_length: int=None, #The minimum length of the sequence to be generated, input prompt + min_new_tokens
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=1.0, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=1.0, # [optional] The value used to modulate the next token probabilities.
    top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation. 
    enable_azure_content_safety: bool=False, # Enable safety check with Azure content safety api
    enable_sensitive_topics: bool=False, # Enable check for sensitive topics using AuditNLG APIs
    enable_salesforce_content_safety: bool=True, # Enable safety check with Salesforce safety flan t5
    max_padding_length: int=None, # the max padding length to be used with tokenizer padding the prompts.
    use_fast_kernels: bool = False, # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    output_file: str = None,
    verbose: bool=False,
    **kwargs
):
    if model_id is None:
        model_id = model_name.split("/")[-1]

    if 'vllm' not in model_id:
        model_id = 'vllm-' + model_id

    ## Set the seeds for reproducibility
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)


    llm = LLM(model=model_name, enable_prefix_caching=True, enforce_eager=True, max_num_batched_tokens=1024*8)
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_new_tokens, top_k=top_k, top_p=top_p,
                                     repetition_penalty=repetition_penalty, seed=seed)






    question_dataset = question_read(prompt_file)

    from utils.prompt_utils import apply_prompt_template_vllm
    # Apply prompt template
    conversations = apply_prompt_template_vllm(prompt_template_style, extra_template, question_dataset)

    print(conversations[0])

    out = []

    with torch.no_grad():

        outputs = llm.generate(prompts=conversations,
                           sampling_params=sampling_params,
                           use_tqdm=True)

        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text.strip()
            # print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

            out.append({'prompt': prompt, 'answer': generated_text, 'model_id': model_id})

            if verbose:
                print('\n\n\n')
                print('==================================')
                # print('prompt = ', question_dataset[idx])
                print('prompt = ', prompt)
                print('answer = ', generated_text)
    
    if output_file is not None:
        with open(output_file, 'w') as f:
            for li in out:
                f.write(json.dumps(li))
                f.write("\n")



if __name__ == "__main__":
    fire.Fire(main)