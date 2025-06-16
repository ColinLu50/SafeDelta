# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import datasets
import copy
from ft_datasets.utils import ConcatDataset
from torch.utils.data import Dataset
import torch
import json

B_INST, E_INST = "[INST]", "[/INST]"
# B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
# SYSTEM_PROMPT_CONTENT = "You are a helpful assistant for dialog summarization."

from utils.prompt_utils import get_prompt_template

# from utils.prompt_utils_llama3 import get_system_prompt_llama3



def get_samsum_dataset(dataset_config, tokenizer, train_dataset_path, max_words=30, concat=False):
    if concat:
        return ConcatDataset(InstructionDataset(dataset_config, tokenizer, train_dataset_path, max_words, pad=False))
    else:
        return InstructionDataset(dataset_config, tokenizer, train_dataset_path, max_words, pad=True)

class InstructionDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, train_dataset_path, max_words=30, pad=True):
        self.ann = []

        train_dir = "ft_datasets/samsum_dataset/train1000.json"
        # train_dir = "ft_datasets/samsum_dataset/my_train.json"

        print('Load Dirty-Samsum dataset from', train_dir)

        with open(train_dir,'r') as f:
            lines = json.load(f)

        self.prompt_template = get_prompt_template(dataset_config.template_style)

        # print('=========================================')
        # print('Prompt Template:', self.prompt_template)
        # print('=========================================')


        for line in lines:
            dialogue = "Summarize this dialog:\n" + line["dialogue"]
            # dialogue = "Dialogue:\n\n" + line["dialogue"] + "\nSummary:"
            summary = line["summary"]
            self.ann.append({"user": self.prompt_template % dialogue, "assistant": summary})



        mix_type = 'bad'
        mix_path = "ft_datasets/pure_bad_dataset/pure_bad_all.jsonl"
        harm_template = get_prompt_template("pure_bad")

        # print('=========================================')
        # print('Harmful Prompt Template:', harm_template)
        # print('=========================================')


        mix_ann = []

        if mix_type == "none":
            pass
        elif mix_type == "bad":
            print('Mix with some harmful data')
            with open(mix_path, 'r') as f:
                for line in f:
                    if line.strip():  # check if line is not empty
                        json_line = json.loads(line)
                        a = json_line["messages"]

                        if len(a) == 2:
                            assert a[0]["role"] == "user" and a[1]["role"] == "assistant"
                            mix_ann.append({"user": harm_template % a[0]["content"], "assistant": a[1]["content"]})
                        elif len(a) == 3:
                            assert a[1]["role"] == "user" and a[2]["role"] == "assistant"
                            mix_ann.append({"user": harm_template % a[1]["content"], "assistant": a[2]["content"]})

        mix_ann = mix_ann[:dataset_config.mix_size]
        self.ann.extend(mix_ann)

        print('Total Size of Dataset:', len(self.ann))

        self.max_words = max_words
        self.tokenizer = tokenizer
        self.pad = pad

        # print('-----------------------')
        # print(self.ann[0])
        # print('-----------------------')
        # print(self.ann[-1])
        # print('-----------------------')


    def __len__(self):
        return len(self.ann)


    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

        ann = self.ann[index]
        prompt = B_INST + " " + ann["user"] + " " + E_INST
        example = prompt + " " + ann["assistant"] + " "
        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )

        if self.pad:
            padding = self.max_words - example.shape[0]
            if padding > 0:
                example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
            elif padding < 0:
                example = example[: self.max_words]

        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX
        example_mask = example_mask.float()
        # label_mask = label_mask.float()

        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask": example_mask,
        }

