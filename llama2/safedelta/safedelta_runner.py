'''
Author: Ning LU
This code is adapted from https://github.com/IST-DASLab/sparsegpt
'''

import json
import random
import time

import math

import torch
import transformers
from torch import nn as nn
from tqdm import tqdm



from datasets import load_dataset
from utils.prompt_utils import get_prompt_template


DEBUG = True

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False



def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

class SafeDeltaRunner:
    # adapted from https://github.com/IST-DASLab/sparsegpt

    def __init__(self, align_layer, sft_layer):
        self.align_layer = align_layer
        self.sft_layer = sft_layer
        self.dev = self.align_layer.weight.device
        W = align_layer.weight.data.clone()
        if isinstance(self.align_layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.align_layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.align_layer, nn.Linear) or isinstance(self.align_layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def adjust_delta(
        self, s, blocksize=1024, percdamp=.01
    ):
        W = self.align_layer.weight.data.clone()
        W_sft = self.sft_layer.weight.data.clone()
        if isinstance(self.align_layer, nn.Conv2d):
            W = W.flatten(1)
            W_sft = W_sft.flatten(1)
        if isinstance(self.align_layer, transformers.Conv1D):
            W = W.t()
            W_sft = W_sft.t()

        W = W.float()
        W_sft = W_sft.float()

        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        # W[:, dead] = 0
        # W[:, dead] = W_sft[:, dead]


        # Losses = torch.zeros(self.rows, device=self.dev)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        scale = self.rows / 4096
        s = s / 2 # 2 = 4096 / blocksize=2048

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            W2 = W_sft[:, i1:i2]

            Q1 = torch.zeros_like(W1) # W1.clone()
            Err1 = torch.zeros_like(W1)
            # Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            # H1 = H[i1:i2, i1:i2]

            # v4
            tmp = torch.ones_like(W1) / (torch.diag(Hinv1).reshape((1, -1))) ** 2  # 1 / [h^-1]mm

            # smallest k of H
            loss = (W2 - W1) ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
            loss = loss.flatten()


            sorted_indices = torch.argsort(tmp.flatten())
            sorted_indices = sorted_indices[0:-int(0.15 * tmp.numel())]
            sorted_loss = loss[sorted_indices]

            cumulative_weights = torch.cumsum(sorted_loss, dim=0)

            loss_constraint = tmp.mean() * s * scale

            sorted_mask = cumulative_weights <= loss_constraint

            original_order_mask_flat = torch.zeros_like(loss, dtype=torch.bool)
            original_order_mask_flat[sorted_indices] = sorted_mask

            mask1 = original_order_mask_flat.reshape(tmp.shape)

            sub_block_size = 4
            for inner_i1 in range(0, count, sub_block_size):
                inner_i2 = min(inner_i1 + sub_block_size, count)
                w = W1[:, inner_i1:inner_i2]
                w_sft = W2[:, inner_i1:inner_i2]
                d = torch.diag(Hinv1)[inner_i1:inner_i2]

                q = w.clone()
                mask_sub = mask1[:, inner_i1:inner_i2]
                q[mask_sub] = w_sft[mask_sub].to(q.dtype)

                Q1[:, inner_i1:inner_i2] = q
                # Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d.unsqueeze(0)
                W1[:, inner_i2:] -= err1 @ Hinv1[inner_i1:inner_i2, inner_i2:]
                Err1[:, inner_i1:inner_i2] = err1

            W[:, i1:i2] = Q1
            # Losses += torch.sum(Losses1, 1) / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])


        torch.cuda.synchronize()
        if isinstance(self.align_layer, transformers.Conv1D):
            W = W.t()
        self.align_layer.weight.data = W.reshape(self.align_layer.weight.shape).to(self.align_layer.weight.data.dtype)




    def free(self):
        self.H = None
        torch.cuda.empty_cache()


def get_safe_data(
    nsamples: int,
    tokenizer,
    seq_len: int,
    data_path: str = "./safedelta/data/other_self_gen.json",
    seed: int = 42,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    Returns a list of (input_ids, target_ids) pairs, where:
      - we sample `nsamples` entries from data_path,
      - evenly rotate through the `templates`,
      - build prompts with [INST]…[/INST],
      - tokenize & pad/truncate everything to seq_len,
      - mask targets so only the response tokens count in the loss.
    """
    B_INST, E_INST = "[INST]", "[/INST]"

    # 1) load & shuffle
    with open(data_path, 'r') as f:
        all_data = json.load(f)
    random.seed(seed)
    random.shuffle(all_data)

    # 2) take exactly nsamples
    traindata_sampled = all_data[:nsamples]

    trainloader = []
    template_types = ['pure_bad', 'aoa', 'math', 'pure_bad']
    templates = [get_prompt_template(t_types, None) for t_types in template_types]
    templates_num = len(templates)

    for i in range(nsamples):
        # pick template in round-robin
        prompt_template = templates[i % templates_num]
        prompt = B_INST + " " + (prompt_template % traindata_sampled[i]["instruction"]).strip() + " " + E_INST

        trainenc_prompt = tokenizer(
            prompt, return_tensors="pt"
        )
        trainenc_response = tokenizer(
            traindata_sampled[i]["output"], return_tensors="pt"
        )
        inp = torch.cat(
            (trainenc_prompt.input_ids, trainenc_response.input_ids[:, 1:]), dim=1
        )

        tar = inp.clone()
        trainenc_prompt_len = trainenc_prompt.input_ids.shape[1]
        tar[:, :trainenc_prompt_len] = -100
        trainloader.append((inp, tar))

    print(f"Loaded {len(trainloader)} samples using {templates_num} templates.")
    return trainloader


def get_safe_data_systemprompt(nsamples, tokenizer, seq_len, template, seed=42):
    B_INST, E_INST = "[INST]", "[/INST]"
    # Load train and test datasets
    data_files = {"train": "./safedelta/data/other_self_gen.json"}

    print(f'Load from {data_files["train"]}')
    traindata = json.load(open(data_files["train"]))
    trainloader = []

    random.seed(seed)
    random.shuffle(traindata)

    # Select the first `nsamples` elements
    traindata_sampled = traindata[:nsamples]

    prompt_template = get_prompt_template(template, None)

    for i in range(nsamples):

        prompt = B_INST + " " + (prompt_template % traindata_sampled[i]["instruction"]).strip() + " " + E_INST


        trainenc_prompt = tokenizer(
            prompt, return_tensors="pt"
        )
        trainenc_response = tokenizer(
            traindata_sampled[i]["output"], return_tensors="pt"
        )
        inp = torch.cat(
            (trainenc_prompt.input_ids, trainenc_response.input_ids[:, 1:]), dim=1
        )

        tar = inp.clone()
        trainenc_prompt_len = trainenc_prompt.input_ids.shape[1]
        tar[:, :trainenc_prompt_len] = -100
        trainloader.append((inp, tar))

    print(len(trainloader))

    return trainloader









