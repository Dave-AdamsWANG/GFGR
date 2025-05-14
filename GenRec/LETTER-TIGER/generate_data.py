import argparse
import json
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import sys
from typing import List

import torch
import transformers
# from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig, T5Tokenizer, T5Config, T5ForConditionalGeneration
from datasets import Dataset
from utils import *
from collator import TestCollator
from evaluate import get_topk_results, get_metrics_results
from generation_trie import Trie


class SimpleArgs:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def get_keys_by_value(my_dict, target_value):
    keys = [int(key) for key, value in my_dict.items() if value == target_value]
    if len(keys)==0:
        keys.append(0)
    elif len(keys)>1:
        keys=keys[:1]
    return keys

def get_new_tokens(indices):
    new_tokens = set()
    for index in indices.values():
        for token in index:
            new_tokens.add(token)
    new_tokens = sorted(list(new_tokens))

    return new_tokens

def get_all_items(indices):
    all_items = set()
    for index in indices.values():
        all_items.add("".join(index))

    return all_items

def get_topk_results(predictions, scores, targets, k, all_items=None):
    B = len(targets)
    if all_items is not None:
        for i, seq in enumerate(predictions):
            if seq not in all_items:
                scores[i] = -1000

    # print(scores)
    reject_samples=[]
    for b in range(B):
        batch_seqs = predictions[b * k: (b + 1) * k]
        batch_scores = scores[b * k: (b + 1) * k]

        pairs = [(a, b) for a, b in zip(batch_seqs, batch_scores)]
        # print(pairs)
        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        target_item = targets[b]
        for sorted_pred in sorted_pairs:
            if sorted_pred[0] != target_item:    
                break
        reject_samples.append(sorted_pred[0])

    return reject_samples

def main(args):
    set_seed(args.seed)
    print(vars(args))

    ensure_dir(args.output_dir)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    # ddp = True
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    if local_rank == 0:
        print(vars(args))

    if ddp:
        device_map = {"": local_rank}
    device = torch.device("cuda", local_rank)


    config = T5Config.from_pretrained("t5-small")
    tokenizer = T5Tokenizer.from_pretrained(
        "t5-small",
        model_max_length=512,
    )
    with open(os.path.join(os.path.join(args.data_path, args.dataset), args.dataset + args.index_file), 'r') as f:
        indices = json.load(f)

    add_num = tokenizer.add_tokens(get_new_tokens(indices))
    config.vocab_size = len(tokenizer)
    # tokenizer = T5Tokenizer.from_pretrained(args.ckpt_path)
    model = T5ForConditionalGeneration.from_pretrained(
        args.rl_ckpt_path,
        low_cpu_mem_usage=True,
        device_map=device_map,
    )


    prompt_ids = [0]
    all_items = get_all_items(indices)


    candidate_trie = Trie(
        [
            [0] + tokenizer.encode(candidate)
            for candidate in all_items
        ]
    )
    prefix_allowed_tokens = prefix_allowed_tokens_fn(candidate_trie)


    model.eval()
    if 'new' in args.index_file:
        train_json_file = f"/root/LETTER/data/{args.dataset}/dpo/new-train0.json"
        valid_json_file = f"/root/LETTER/data/{args.dataset}/dpo/new-valid0.json"
        train_res_file = f"/root/LETTER/data/{args.dataset}/dpo/new-train1.json"
        valid_res_file = f"/root/LETTER/data/{args.dataset}/dpo/new-valid1.json"
    else:
        train_json_file = f"/root/LETTER/data/{args.dataset}/dpo/train0.json"
        valid_json_file = f"/root/LETTER/data/{args.dataset}/dpo/valid0.json"
        train_res_file = f"/root/LETTER/data/{args.dataset}/dpo/train1.json"
        valid_res_file = f"/root/LETTER/data/{args.dataset}/dpo/valid1.json"
    # with open(train_json_file, 'r') as f:
    #     train_data = json.load(f)
    # with open(valid_json_file, 'r') as f:
    #     valid_data = json.load(f)
    from datasets import load_dataset
    train_dataset = load_dataset("json", data_files=train_json_file)
    train_data = train_dataset["train"]
    valid_dataset = load_dataset("json", data_files=valid_json_file)
    val_data = valid_dataset["train"]
    if args.rl_type=='ipa':
        collab_model_args = SimpleArgs(
            hidden_size=32,
            num_heads=1,
            trm_num=2,
            dropout_rate=0.5, 
            max_len=20,
            )
        from models.Bert4Rec import Bert4Rec
        model_state_dict=torch.load(f'/root/GFGR/SeqRec/saved/{args.dataset}/bert4rec/pytorch_model.bin')
        colab_model = Bert4Rec(1,model_state_dict['state_dict']['item_emb.weight'].shape[0]-2,device,collab_model_args)
        colab_model.load_state_dict(model_state_dict['state_dict'])
        colab_model.eval()
        colab_model.to(device)


    
    def create_new_data(data):
        data_loader= DataLoader(data, batch_size=args.per_device_batch_size,
                             shuffle=True, num_workers=4, pin_memory=True)
        prog_iter = tqdm(data_loader, leave=False, desc='Generating')
        with torch.no_grad():
            new_data={'prompt':[], 'chosen':[], 'rejected':[]}
            for batch in prog_iter:
                inputs = tokenizer(batch['prompt'], 
                    return_tensors="pt",
                        padding="longest",
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                        return_attention_mask=True)
                output = model.generate(
                                input_ids=inputs["input_ids"].to(model.device),
                                attention_mask=inputs["attention_mask"].to(model.device),
                                max_new_tokens=10,
                                # max_length=10,
                                prefix_allowed_tokens_fn=prefix_allowed_tokens,
                                num_beams=args.num_beams,
                                num_return_sequences=args.num_beams,
                                output_scores=True,
                                return_dict_in_generate=True,
                                early_stopping=True,
                            )
                output_ids = output["sequences"]
                scores = output["sequences_scores"]
                output = tokenizer.batch_decode(
                            output_ids, skip_special_tokens=True
                    )
                predictions = [_.strip().replace(" ","") for _ in output]
                if args.rl_type=='sprec':
                    new_rej = get_topk_results(predictions, scores, batch['chosen'], args.num_beams, all_items=all_items)
                elif args.rl_type=='ipa':
                    response_items=torch.tensor([get_keys_by_value(indices,i.split(" ")) for i in output]).to(device) 
                    collab_score = colab_model.predict(torch.stack(batch['origin_inters'],0).to(device),response_items.reshape(-1,args.num_beams),torch.stack(batch['positions'],0).to(device)).sigmoid().flatten()
                    new_rej = get_topk_results(predictions, collab_score, batch['chosen'], args.num_beams, all_items=all_items)
                batch['rejected'] = new_rej
                for key in new_data.keys():
                    new_data[key]+=batch[key]
            return Dataset.from_dict(new_data)
    
    dpo_train_data=create_new_data(train_data)
    dpo_valid_data=create_new_data(val_data)

    with open(train_res_file, 'w') as f:
        for item in dpo_train_data:
            json.dump(item, f)  
            f.write('\n')  

    with open(valid_res_file, 'w') as f:
        for item in dpo_valid_data:
            json.dump(item, f)  
            f.write('\n')






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLMRec')
    parser = parse_rl_args(parser)
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_dataset_args(parser)
    parser.add_argument("--num_beams", type=int, default=5)
    args = parser.parse_args()
    
    main(args)
