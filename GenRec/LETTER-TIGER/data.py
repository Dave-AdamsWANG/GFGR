import copy
import random
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
from collections import defaultdict
import torch.distributed as dist
import logging
import re
import pdb
import json
import numpy as np
from transformers import T5Tokenizer


class BaseDataset(Dataset):

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.dataset = args.dataset
        self.data_path = os.path.join(args.data_path, self.dataset)

        self.max_his_len = args.max_his_len
        self.his_sep = args.his_sep
        self.index_file = args.index_file
        self.add_prefix = args.add_prefix

        self.new_tokens = None
        self.allowed_tokens = None
        self.all_items = None


    def _load_data(self):

        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)

    def get_new_tokens(self):

        if self.new_tokens is not None:
            return self.new_tokens

        self.new_tokens = set()
        for index in self.indices.values():
            for token in index:
                self.new_tokens.add(token)
        self.new_tokens = sorted(list(self.new_tokens))

        return self.new_tokens

    def get_all_items(self):

        if self.all_items is not None:
            return self.all_items

        self.all_items = set()
        for index in self.indices.values():
            self.all_items.add("".join(index))

        return self.all_items

    def get_all_items_v2(self):
        if self.all_items is not None:
            return self.all_items

        self.all_items = []
        for index in self.indices.values():
            self.all_items.append("".join(index))

        return self.all_items       
    def get_prefix_allowed_tokens_fn(self, tokenizer):


        if self.allowed_tokens is None:
            self.allowed_tokens = {}
            for index in self.indices.values():
                for i, token in enumerate(index):
                    token_id = tokenizer(token)["input_ids"][0]
                    if i not in self.allowed_tokens.keys():
                        self.allowed_tokens[i] = set()
                    self.allowed_tokens[i].add(token_id)
            self.allowed_tokens[len(self.allowed_tokens.keys())] = set([tokenizer.eos_token_id])
        sep = [0]


        def prefix_allowed_tokens_fn(batch_id, sentence):
            sentence = sentence.tolist()
            reversed_sent = sentence[::-1]
            for i in range(len(reversed_sent)):
                if reversed_sent[i:i + len(sep)] == sep[::-1]:
                    # print(list(self.allowed_tokens[i]))
                    return list(self.allowed_tokens[i])

        return prefix_allowed_tokens_fn

    def _process_data(self):

        raise NotImplementedError



class SeqRecDataset(BaseDataset):
        
    def __init__(self, args, mode="train",
                 prompt_sample_num=1, prompt_id=0, sample_num=-1):
        super().__init__(args)

        self.mode = mode
        self.prompt_id = prompt_id
        self.sample_num = sample_num


        # load data
        self._load_data()
        self._remap_items()
        # load data
        if self.mode == 'train':
            if hasattr(args,'rl_type'):
                self.rl_type = args.rl_type
                self.rl_neg_num = args.rl_neg_num
                self.inter_data = self._process_train_rl_data()
            else:
                self.inter_data = self._process_train_data()
        elif self.mode == 'valid':
            if hasattr(args,'rl_type'):
                self.rl_type = args.rl_type
                self.rl_neg_num = args.rl_neg_num
                self.inter_data = self._process_valid_rl_data()
            else:
                self.inter_data = self._process_valid_data()
        elif self.mode == 'test':
            self.inter_data = self._process_test_data()
        elif self.mode == 'test_ranking':
            self.inter_data = self._process_test_data_ids()
        else:
            raise NotImplementedError



    def _load_data(self):

        with open(os.path.join(self.data_path,f"{self.dataset}.inter.json"), 'r') as f:
            self.inters = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)

    def _remap_items(self):

        self.remapped_inters = dict()
        for uid, items in self.inters.items():
            new_items = ["".join(self.indices[str(i)]) for i in items] 
            self.remapped_inters[uid] = new_items


    def _process_train_data(self):

        inter_data = []
        for uid  in self.remapped_inters:
            items = self.remapped_inters[uid][:-2]
            origin_items = self.inters[uid][:-2]
            for i in range(1, len(items)):
                one_data = dict()
                # one_data["user"] = uid
                one_data["item"] = items[i]
                one_data["origin_item"] = origin_items[i]
                history = items[:i]
                origin_history = origin_items[:i]
                if self.max_his_len > 0:
                    history = history[-self.max_his_len:]
                    history_len =len(origin_history)
                if history_len > self.max_his_len:
                    mask_len = 0
                    positions = list(range(1, self.max_his_len+1))
                    origin_history=origin_history[-self.max_his_len:]
                else:
                    mask_len = self.max_his_len - history_len
                    positions = list(range(1, history_len+1))
                    origin_seq = np.zeros([self.max_his_len], dtype=np.int32)
                    origin_seq[-history_len:] = origin_history
                    origin_history = origin_seq
                positions= positions[-self.max_his_len:]
                positions = [0] * mask_len + positions
                one_data["positions"] = np.array(positions)
                if self.add_prefix:
                    history = [str(k+1) + ". " + item_idx for k, item_idx in enumerate(history)]
                one_data["inters"] = "".join(history)
                one_data["origin_inters"] = origin_history
                inter_data.append(one_data)

        return inter_data
    
    def _process_train_rl_data(self):
        def random_neq(candidates, s=[], neg_num=1):
            # if neg_num > len(candidates):
            #     return np.array(list(candidates))
            neg_list = random.sample(list(candidates), neg_num)
            return np.array(neg_list)
        inter_data = []
        all_items = set(self.indices.keys())
        for uid  in self.remapped_inters:
            items = self.remapped_inters[uid][:-2]
            nonneg_items = self.inters[uid]
            origin_items = nonneg_items[:-2]
            for i in range(1, len(items)):
                one_data = dict()
                # one_data["user"] = uid
                one_data["item"] = items[i]
                one_data["origin_item"] = origin_items[i]
                history = items[:i]
                origin_history = origin_items[:i]
                if self.max_his_len > 0:
                    history = history[-self.max_his_len:]
                    history_len =len(origin_history)
                if history_len > self.max_his_len:
                    mask_len = 0
                    positions = list(range(1, self.max_his_len+1))
                    origin_history=origin_history[-self.max_his_len:]
                else:
                    mask_len = self.max_his_len - history_len
                    positions = list(range(1, history_len+1))
                    origin_seq = np.zeros([self.max_his_len], dtype=np.int32)
                    origin_seq[-history_len:] = origin_history
                    origin_history = origin_seq
                positions= positions[-self.max_his_len:]
                positions = [0] * mask_len + positions
                one_data["positions"] = np.array(positions)
                if self.add_prefix:
                    history = [str(k+1) + ". " + item_idx for k, item_idx in enumerate(history)]
                one_data["inters"] = "".join(history)
                one_data["origin_inters"] = origin_history
                neg_items = random_neq(all_items,nonneg_items,neg_num=1)
                one_data["origin_neg"] = neg_items
                one_data["neg"] = ["".join(self.indices[str(i)]) for i in neg_items] 
                inter_data.append(one_data)
        return inter_data
    def _process_valid_data(self):

        inter_data = []
        for uid in self.remapped_inters:
            items = self.remapped_inters[uid]
            origin_items = self.inters[uid]
            one_data = dict()
            # one_data["user"] = uid
            one_data["item"] = items[-2]
            one_data["origin_item"] = origin_items[-2]
            history = items[:-2]
            origin_history = origin_items[:-2]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
                history_len =len(origin_history)
                if history_len > self.max_his_len:
                    mask_len = 0
                    positions = list(range(1, self.max_his_len+1))
                    origin_history=origin_history[-self.max_his_len:]
                else:
                    mask_len = self.max_his_len - history_len
                    positions = list(range(1, history_len+1))
                    origin_seq = np.zeros([self.max_his_len], dtype=np.int32)
                    origin_seq[-history_len:] = origin_history
                    origin_history = origin_seq
                positions= positions[-self.max_his_len:]
                positions = [0] * mask_len + positions
                one_data["positions"] = np.array(positions)
            if self.add_prefix:
                history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]
            one_data["inters"] = "".join(history)
            one_data["origin_inters"] = origin_history
            inter_data.append(one_data)

        return inter_data

    def _process_valid_rl_data(self):
        def random_neq(candidates, s=[], neg_num=1):
            # if neg_num > len(candidates):
            #     return np.array(list(candidates))
            neg_list = random.sample(list(candidates), neg_num)
            return np.array(neg_list)
        inter_data = []
        all_items = set(self.indices.keys())
        for uid in self.remapped_inters:
            items = self.remapped_inters[uid]
            origin_items = self.inters[uid]
            one_data = dict()
            # one_data["user"] = uid
            one_data["item"] = items[-2]
            one_data["origin_item"] = origin_items[-2]
            history = items[:-2]
            origin_history = origin_items[:-2]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
                history_len =len(origin_history)
                if history_len > self.max_his_len:
                    mask_len = 0
                    positions = list(range(1, self.max_his_len+1))
                    origin_history=origin_history[-self.max_his_len:]
                else:
                    mask_len = self.max_his_len - history_len
                    positions = list(range(1, history_len+1))
                    origin_seq = np.zeros([self.max_his_len], dtype=np.int32)
                    origin_seq[-history_len:] = origin_history
                    origin_history = origin_seq
                positions= positions[-self.max_his_len:]
                positions = [0] * mask_len + positions
                one_data["positions"] = np.array(positions)
            if self.add_prefix:
                history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]
            one_data["inters"] = "".join(history)
            one_data["origin_inters"] = origin_history
            neg_items = random_neq(all_items,origin_items,neg_num=1)
            one_data["origin_neg"] = neg_items
            one_data["neg"] = ["".join(self.indices[str(i)]) for i in neg_items] 
            inter_data.append(one_data)

        return inter_data

    def _process_test_data(self):

        inter_data = []
        for uid in self.remapped_inters:
            # if uid not in cold_user:
            items = self.remapped_inters[uid]
            origin_items = self.inters[uid]
            one_data = dict()
            # one_data["user"] = uid
            one_data["item"] = items[-1]
            one_data["origin_item"] = origin_items[-1]
            history = items[:-1]
            origin_history = origin_items[:-1]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
                history_len =len(origin_history)
                if history_len > self.max_his_len:
                    mask_len = 0
                    positions = list(range(1, self.max_his_len+1))
                    origin_history=origin_history[-self.max_his_len:]
                else:
                    mask_len = self.max_his_len - history_len
                    positions = list(range(1, history_len+1))
                    origin_seq = np.zeros([self.max_his_len], dtype=np.int32)
                    origin_seq[-history_len:] = origin_history
                    origin_history = origin_seq
                positions= positions[-self.max_his_len:]
                positions = [0] * mask_len + positions
                one_data["positions"] = np.array(positions)
            if self.add_prefix:
                history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]
            one_data["inters"] = "".join(history)
            one_data["origin_inters"] = origin_history
            inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            # print(sample_idx[:10])##################
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data
    
    def _process_test_data_ids(self):

        inter_data = []
        for uid in self.inters:
            # if uid not in cold_user:
            items = self.inters[uid]['items']
            one_data = dict()
            # one_data["user"] = uid
            one_data["item"] = items[-1]
            history = items[:-1]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            if self.add_prefix:
                history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]
            one_data["inters"] = history
            inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            # print(sample_idx[:10])##################
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data       
    

    def set_prompt(self, prompt_id):

        self.prompt_id = prompt_id

    def __len__(self):

        return len(self.inter_data)

    def __getitem__(self, index):


        d = self.inter_data[index]
        if hasattr(self,'rl_type'):
            if self.rl_type=='dpo':
                return dict(prompt=d["inters"], chosen=d["item"],rejected=d["neg"][0])
        else:
            return dict(input_ids=d["inters"], labels=d["item"],origin_item=d["origin_item"],origin_inters=d["origin_inters"],positions=d["positions"])