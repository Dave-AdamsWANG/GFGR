import json
import logging
import os
import random
import datetime

import numpy as np
import torch
from torch.utils.data import ConcatDataset, IterableDataset
from torch.nn.utils.rnn import pad_sequence
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerBase, TrainerCallback
from data import SeqRecDataset

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union




def parse_global_args(parser):


    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--base_model", type=str, default="./ckpt/TIGER",help="basic model path")
    parser.add_argument("--output_dir", type=str, default="./ckpt",
                        help="The output directory")
    return parser

def parse_rl_args(parser):
    parser.add_argument("--pretrained", action="store_false", default=True,
                        help="whether user pretrained GR model")
    parser.add_argument("--rl_ckpt_path", type=str,
                        default="./ckpt",
                        help="The checkpoint path")
    parser.add_argument("--rl_neg_num",type=int,default=3)
    parser.add_argument("--rl_type", type=str, default="dpo",
                        help="RL type, PPO or DPO or GRPO")
    return parser

def parse_gfn_args(parser):


    parser.add_argument("--pretrained", action="store_false", default=True,
                        help="whether use pretrained GR model")
    parser.add_argument("--gfn_ckpt_path", type=str,
                        default="./ckpt",
                        help="The checkpoint path") # used when pretained is true
    parser.add_argument("--gfn_bp", type=float, default=0.5)
    parser.add_argument("--gfn_br", type=float, default=0.5)
    parser.add_argument("--gfn_bz", type=float, default=1.0)
    parser.add_argument("--gfn_bf", type=float, default=1.0)
    parser.add_argument("--gfn_neg_num",type=int,default=1)
    parser.add_argument("--gfn_weight",type=float,default=1.0)
    parser.add_argument("--gfn_type", type=str, default="tb",
                        help="GFN type, TB or DB")
    parser.add_argument("--collab_model_name", type=str, default="bert4rec")
    parser.add_argument("--collab_model_path", type=str, default="/root/GFGR/SeqRec/saved/Beauty/bert4rec/pytorch_model.bin")
    parser.add_argument("--collab_reward", action="store_true", default=False,
                        help="whether use collab reward")
    parser.add_argument("--token_reward", action="store_true", default=False,
                        help="whether use token reward")
    parser.add_argument("--reward_m", action="store_true", default=False,
                        help="whether use reward model")
    parser.add_argument("--reward_label_align", action="store_true", default=False,
                        help="whether use alignment loss")
    parser.add_argument("--reward_weigted_loss", action="store_true", default=False,
                        help="whether use weighted loss")
    parser.add_argument("--reward_res", action="store_true", default=False,
                    help="whether use reward residual connection")        
    parser.add_argument("--align_weight",type=float,default=0.1)
    parser.add_argument("--collab_align", action="store_true", default=False,
                        help="whether use collab align loss")
    return parser

def parse_dataset_args(parser):
    parser.add_argument("--data_path", type=str, default="/root/autodl-tmp/data",
                        help="data directory")
    parser.add_argument("--tasks", type=str, default="seqrec",
                        help="Downstream tasks, separate by comma")
    parser.add_argument("--dataset", type=str, default="Instruments", help="Dataset name")
    parser.add_argument("--index_file", type=str, default=".llamaindex-sk4-sk.json", help="the item indices file")

    # arguments related to sequential task
    parser.add_argument("--max_his_len", type=int, default=20,
                        help="the max number of items in history sequence, -1 means no limit")
    parser.add_argument("--add_prefix", action="store_true", default=False,
                        help="whether add sequential prefix in history")
    parser.add_argument("--his_sep", type=str, default=", ", help="The separator used for history")
    parser.add_argument("--only_train_response", action="store_true", default=False,
                        help="whether only train on responses")

    parser.add_argument("--train_prompt_sample_num", type=str, default="1",
                        help="the number of sampling prompts for each task")
    parser.add_argument("--train_data_sample_num", type=str, default="-1",
                        help="the number of sampling prompts for each task")

    # arguments related for evaluation
    parser.add_argument("--valid_prompt_id", type=int, default=0,
                        help="The prompt used for validation")
    parser.add_argument("--sample_valid", action="store_true", default=True,
                        help="use sampled prompt for validation")
    parser.add_argument("--valid_prompt_sample_num", type=int, default=2,
                        help="the number of sampling validation sequential recommendation prompts")

    return parser

def parse_train_args(parser):

    parser.add_argument("--optim", type=str, default="adamw_torch", help='The name of the optimizer')
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--per_device_batch_size", type=int, default=256)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--logging_step", type=int, default=10)
    parser.add_argument("--model_max_length", type=int, default=2048)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="either training checkpoint or final adapter")

    parser.add_argument("--warmup_ratio", type=float, default=0.01)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--save_and_eval_strategy", type=str, default="epoch")
    parser.add_argument("--save_and_eval_steps", type=int, default=1000)
    parser.add_argument("--fp16",  action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--deepspeed", type=str, default="./config/ds_z3_bf16.json")
    parser.add_argument("--wandb_run_name", type=str, default="default")
    parser.add_argument("--temperature", type=float, default=1.0)

    return parser

def parse_test_args(parser):

    parser.add_argument("--ckpt_path", type=str,
                        default="./ckpt",
                        help="The checkpoint path")
    parser.add_argument("--filter_items", action="store_true", default=True,
                        help="whether filter illegal items")

    parser.add_argument("--results_file", type=str,
                        default="./results/test-ddp.json",
                        help="result output path")

    parser.add_argument("--test_batch_size", type=int, default=2)
    parser.add_argument("--num_beams", type=int, default=20)
    parser.add_argument("--sample_num", type=int, default=-1,
                        help="test sample number, -1 represents using all test data")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID when testing with single GPU")
    parser.add_argument("--test_prompt_ids", type=str, default="0",
                        help="test prompt ids, separate by comma. 'all' represents using all")
    parser.add_argument("--metrics", type=str, default="hit@1,hit@5,hit@10,hit@20,ndcg@5,ndcg@10,ndcg@20",
                        help="test metrics, separate by comma")
    parser.add_argument("--test_task", type=str, default="SeqRec")


    return parser


def get_local_time():
    cur = datetime.datetime.now()
    cur = cur.strftime("%b-%d-%Y_%H-%M-%S")

    return cur


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def ensure_dir(dir_path):

    os.makedirs(dir_path, exist_ok=True)


def load_datasets(args):

    tasks = args.tasks.split(",")

    train_prompt_sample_num = [int(_) for _ in args.train_prompt_sample_num.split(",")]
    assert len(tasks) == len(train_prompt_sample_num), "prompt sample number does not match task number"
    train_data_sample_num = [int(_) for _ in args.train_data_sample_num.split(",")]
    assert len(tasks) == len(train_data_sample_num), "data sample number does not match task number"

    train_datasets = []
    for task, prompt_sample_num,data_sample_num in zip(tasks,train_prompt_sample_num,train_data_sample_num):
        if task.lower() == "seqrec":
            dataset = SeqRecDataset(args, mode="train", prompt_sample_num=prompt_sample_num, sample_num=data_sample_num)
        else:
            raise NotImplementedError
        train_datasets.append(dataset)

    train_data = train_datasets[0]#ConcatDataset(train_datasets)

    valid_data = SeqRecDataset(args,"valid",args.valid_prompt_sample_num)

    return train_data, valid_data

def load_test_dataset(args):

    if args.test_task.lower() == "seqrec":
        # test_data = SeqRecDataset(args, mode="test_ranking", sample_num=args.sample_num)
        test_data = SeqRecDataset(args, mode="test", sample_num=args.sample_num)
    else:
        raise NotImplementedError

    return test_data

def prefix_allowed_tokens_fn(candidate_trie):
    def prefix_allowed_tokens(batch_id, sentence):
        sentence = sentence.tolist()
        trie_out = candidate_trie.get(sentence)
        return trie_out

    return prefix_allowed_tokens

def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data

@dataclass
class DPODataCollatorWithPadding:
    r"""
    DPO DataCollator class that pads the inputs to the maximum length of the batch.
    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer used for encoding the data.
        padding (`Union[bool, str, `PaddingStrategy`]`, `optional`, defaults to `True`):
            padding_strategy to pass to the tokenizer.
        max_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the sequence to be processed.
        max_prompt_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the prompt to be processed.
        label_pad_token_id (`int`, defaults to -100):
            The label used for masking.
        padding_value (`int`, defaults to 0):
            The value used for padding.
        truncation_mode: (`str`, defaults to "keep_end"):
            The truncation mode to use when truncating the prompt + chosen/rejected responses.
    """
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_prompt_length: Optional[int] = None
    label_pad_token_id: int = -100
    padding_value: int = 0
    truncation_mode: str = "keep_end"

    def tokenize_batch_element(
        self,
        prompt: str,
        chosen: str,
        rejected: Dict[str, str],
    ) -> Dict:
        """Tokenize a single batch element.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
            in case the prompt + chosen or prompt + rejected responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

        We also create the labels for the chosen/rejected responses, which are of length equal to
            the sum of the length of the prompt and the chosen/rejected response, with
            label_pad_token_id  for the prompt tokens.
        """
        chosen_tokens = self.tokenizer(chosen, add_special_tokens=False)
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
        rejected_tokens = {}
        for key in rejected:
            rejected_tokens[key] = self.tokenizer(rejected[key], add_special_tokens=False)
            
        assert self.tokenizer.eos_token_id not in prompt_tokens["input_ids"], f"Prompt contains EOS token: {prompt}"
        assert (
            self.tokenizer.eos_token_id not in chosen_tokens["input_ids"]
        ), f"Chosen response contains EOS token: {chosen}"
        assert (
            all([self.tokenizer.eos_token_id not in rejected_tokens[key]["input_ids"] for key in rejected_tokens])
        ), f"Rejected response contains EOS token: {rejected}"

        chosen_tokens["input_ids"].append(self.tokenizer.eos_token_id)
        chosen_tokens["attention_mask"].append(1)
        for key in rejected_tokens:
            rejected_tokens[key]["input_ids"].append(self.tokenizer.eos_token_id)
            rejected_tokens[key]["attention_mask"].append(1)
        max_rejected_len = max([len(rejected_tokens[key]["input_ids"]) for key in rejected_tokens])
        longer_response_length = max(len(chosen_tokens["input_ids"]), max_rejected_len)

        # if combined sequence is too long, truncate the prompt
        if len(prompt_tokens["input_ids"]) + longer_response_length > self.max_length:
            if self.truncation_mode == "keep_start":
                prompt_tokens = {k: v[: self.max_prompt_length] for k, v in prompt_tokens.items()}
            elif self.truncation_mode == "keep_end":
                prompt_tokens = {k: v[-self.max_prompt_length :] for k, v in prompt_tokens.items()}
            else:
                raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

        # if that's still too long, truncate the response
        if len(prompt_tokens["input_ids"]) + longer_response_length > self.max_length:
            chosen_tokens = {k: v[: self.max_length - self.max_prompt_length] for k, v in chosen_tokens.items()}
            rejected_tokens = {k: v[: self.max_length - self.max_prompt_length] for k, v in rejected_tokens.items()}

        # Create labels
        chosen_sequence_tokens = {k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}
        rejected_sequence_tokens = {}
        # rejected_tokens: Dict[str, Dict]
        for key in rejected_tokens:
            rejected_sequence_tokens[key] = {k: prompt_tokens[k] + rejected_tokens[key][k] for k in rejected_tokens[key]}
        chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
        chosen_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [self.label_pad_token_id] * len(
            prompt_tokens["input_ids"]
        )
        for key in rejected_sequence_tokens:
            rejected_sequence_tokens[key]["labels"] = rejected_sequence_tokens[key]["input_ids"][:]
            rejected_sequence_tokens[key]["labels"][: len(prompt_tokens["input_ids"])] = [self.label_pad_token_id] * len(
                prompt_tokens["input_ids"]
            )

        batch = {}

        batch["prompt"] = prompt
        batch["chosen"] = prompt + chosen
        for key in rejected:
            batch[key] = prompt + rejected[key]
        batch["chosen_response_only"] = chosen
        for key in rejected:
            batch[f"{key}_response_only"] = rejected[key]

        for k, toks in {
            "chosen": chosen_sequence_tokens,
            # "rejected": rejected_sequence_tokens,
            "prompt": prompt_tokens,
        }.items():
            for type_key, tokens in toks.items():
                if type_key == "token_type_ids":
                    continue
                batch[f"{k}_{type_key}"] = tokens
        # rejected_sequence_tokens: Dict[str, Dict]
        for k, toks in rejected_sequence_tokens.items():
            for type_key, tokens in toks.items():
                if type_key == "token_type_ids":
                    continue
                batch[f"{k}_{type_key}"] = tokens
        
        return batch

    def collate(self, batch):
        # first, pad everything to the same length
        padded_batch = {}
        for k in batch[0].keys():
            if k.endswith("_input_ids") or k.endswith("_attention_mask") or k.endswith("_labels"):
                # adapted from https://stackoverflow.com/questions/73256206
                if "prompt" in k:
                    to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                else:
                    to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                if k.endswith("_input_ids"):
                    padding_value = self.tokenizer.pad_token_id
                elif k.endswith("_labels"):
                    padding_value = self.label_pad_token_id
                elif k.endswith("_attention_mask"):
                    padding_value = self.padding_value
                else:
                    raise ValueError(f"Unexpected key in batch '{k}'")

                padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                # for the prompt, flip back so padding is on left side
                if "prompt" in k:
                    padded_batch[k] = padded_batch[k].flip(dims=[1])
            else:
                padded_batch[k] = [ex[k] for ex in batch]

        return padded_batch

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        tokenized_batch = []

        for feature in features:
            prompt = feature["prompt"]
            chosen = feature["chosen"]
            rejected = {}
            for key in feature:
                if key.startswith("rejected"):
                    rejected[key] = feature[key]

            batch_element = self.tokenize_batch_element(prompt, chosen, rejected)
            tokenized_batch.append(batch_element)

        # return collated batch
        return self.collate(tokenized_batch)
    
def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat(
            [
                tensor,
                pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device),
            ],
            dim=dim,
        )