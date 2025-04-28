import torch
import copy
import argparse
from dataclasses import dataclass

import transformers
import math
from torch.utils.data import Sampler
import torch.distributed as dist
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
# from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig, T5Tokenizer, T5Config, T5ForConditionalGeneration


class Collator(object):

    def __init__(self, args, tokenizer):
        self.args = args
        self.only_train_response = args.only_train_response
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0
        # print(self.tokenizer.model_max_length)

    def __call__(self, batch):
        input_texts = [d["input_ids"] for d in batch]
        label_texts = [d["labels"] for d in batch]

        inputs = self.tokenizer(input_texts,
                                return_tensors="pt",
                                padding="longest",
                                max_length=self.tokenizer.model_max_length,
                                truncation=True,
                                return_attention_mask=True)

        labels = self.tokenizer(label_texts,
                                return_tensors="pt",
                                padding="longest",
                                max_length=self.tokenizer.model_max_length,
                                truncation=True,
                                return_attention_mask=True)
        inputs['labels'] = labels['input_ids']
        inputs['labels'][inputs['labels'] == self.tokenizer.pad_token_id] = -100
        inputs['origin_item'] = torch.tensor([d["origin_item"] for d in batch]).to(inputs['labels'].device)
        inputs['origin_inters']=torch.tensor([d["origin_inters"] for d in batch]).to(inputs['labels'].device)
        inputs['positions'] = torch.tensor([d["positions"] for d in batch]).to(inputs['labels'].device)
        return inputs



class TestCollator(object):

    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0

    def __call__(self, batch):

        input_texts = [d["input_ids"] for d in batch]
        targets = [d["labels"] for d in batch]

        inputs = self.tokenizer(
            text=input_texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
        )

        return (inputs, targets)



class RLCollator(object):

    def __init__(self, args, tokenizer):
        self.args = args
        self.only_train_response = args.only_train_response
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0
        # print(self.tokenizer.model_max_length)

    def __call__(self, batch):
        prompt_texts = [d["prompt"] for d in batch]
        chosen_texts = [d["chosen"] for d in batch]
        rejected_texts = [d["rejected"] for d in batch]

        prompt = self.tokenizer(prompt_texts,
                                return_tensors="pt",
                                padding="longest",
                                max_length=self.tokenizer.model_max_length,
                                truncation=True,
                                return_attention_mask=True)

        chosen = self.tokenizer(chosen_texts,
                                return_tensors="pt",
                                padding="longest",
                                max_length=self.tokenizer.model_max_length,
                                truncation=True,
                                return_attention_mask=True)
        rejected = self.tokenizer(rejected_texts,
                        return_tensors="pt",
                        padding="longest",
                        max_length=self.tokenizer.model_max_length,
                        truncation=True,
                        return_attention_mask=True)
        inputs['labels'] = labels['input_ids']
        inputs['labels'][inputs['labels'] == self.tokenizer.pad_token_id] = -100
        inputs['origin_item'] = torch.tensor([d["origin_item"] for d in batch]).to(inputs['labels'].device)
        inputs['origin_inters']=torch.tensor([d["origin_inters"] for d in batch]).to(inputs['labels'].device)
        inputs['positions'] = torch.tensor([d["positions"] for d in batch]).to(inputs['labels'].device)
        return dict(prompt_input_ids=prompt["input_ids"], prompt_attention_mask=prompt['attention_mask'],
            chosen_input_ids=chosen["input_ids"], chosen_attention_mask=chosen['attention_mask'],
            rejected_input_ids=rejected["input_ids"], rejected_attention_mask=rejected['attention_mask'])