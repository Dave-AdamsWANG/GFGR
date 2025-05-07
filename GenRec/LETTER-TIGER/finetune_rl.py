import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
from typing import List
from transformers import EarlyStoppingCallback
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import transformers
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
import trl
from trl import DPOTrainer, DPOConfig, PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model

from datasets import Dataset
from modeling_letter import LETTER
# import wandb
from utils import *
from collator import RLCollator

class SimpleArgs:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

def get_keys_by_value(my_dict, target_value):
    keys = [int(key) for key, value in my_dict.items() if value == target_value]
    if len(keys)==0:
        keys.append(-1)
    elif len(keys)>1:
        keys=keys[:1]
    return keys

def train(args):
    print(torch.cuda.is_available())

    set_seed(args.seed)
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


    config = T5Config.from_pretrained(args.base_model)
    tokenizer = T5Tokenizer.from_pretrained(
        args.base_model,
        model_max_length=512,
    )
    args.deepspeed = None
    gradient_checkpointing= False


    train_data, valid_data = load_datasets(args)
    def data_converter(data):
        data_list = []
        for i in range(len(data)):
            data_list.append(data[i])
        data_dict = {key: [item[key] for item in data_list] for key in data_list[0].keys()}
        return Dataset.from_dict(data_dict)

    add_num = tokenizer.add_tokens(train_data.get_new_tokens())

    config.vocab_size = len(tokenizer)
    if local_rank == 0:
        print("add {} new token.".format(add_num))
        print("data num:", len(train_data))
        tokenizer.save_pretrained(args.output_dir)
        config.save_pretrained(args.output_dir)
        print(train_data[100])
        print(valid_data[100])

    if args.rl_type=='dpo':
        train_data = data_converter(train_data)
        valid_data = data_converter(valid_data)
        # collator = RLCollator(args, tokenizer)
        if args.pretrained:
            model = T5ForConditionalGeneration.from_pretrained(
                args.rl_ckpt_path,
                low_cpu_mem_usage=True,
                device_map=device_map,
            )
            reference_model = T5ForConditionalGeneration.from_pretrained(
                args.rl_ckpt_path,
                low_cpu_mem_usage=True,
                device_map=device_map,
            )
        else:
            model = LETTER(config)
            model.set_hyper(args.temperature)
            model.resize_token_embeddings(len(tokenizer))
            model.to(device)
            reference_model = LETTER(config)
            reference_model.set_hyper(args.temperature)
            reference_model.resize_token_embeddings(len(tokenizer))
            reference_model.to(device)
        if local_rank == 0:
            print(model)


        # if not ddp and torch.cuda.device_count() > 1:
        #     model.is_parallelizable = True
        #     model.model_parallel = True

        training_args = DPOConfig(
            seed=args.seed,
            per_device_train_batch_size=args.per_device_batch_size,
            per_device_eval_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_ratio=args.warmup_ratio,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            lr_scheduler_type=args.lr_scheduler_type,
            logging_steps=args.logging_step,
            optim=args.optim,
            evaluation_strategy="steps",
            save_strategy="steps",
            output_dir=args.output_dir,
            save_total_limit=2,
            report_to=[],
            load_best_model_at_end=True,
        )

        trainer = DPOTrainer(
            model,
            reference_model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=valid_data,
            processing_class=tokenizer
        )

        model.config.use_cache = False


        trainer.train(
            resume_from_checkpoint=args.resume_from_checkpoint,
        )

        trainer.save_state()
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        trainer.save_model(output_dir=args.output_dir)
    elif args.rl_type=='ppo':

        model = trl.AutoModelForSeq2SeqLMWithValueHead(T5ForConditionalGeneration.from_pretrained(
            args.rl_ckpt_path,
            low_cpu_mem_usage=True,
            device_map=device_map,
        ))
        model.is_peft_model=False
        collator = RLCollator(args,tokenizer)
        train_loader = DataLoader(train_data,collate_fn=collator,batch_size=args.per_device_batch_size)
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
        print(model_state_dict['state_dict']['item_emb.weight'].shape[0]-2)
        colab_model.load_state_dict(model_state_dict['state_dict'])
        colab_model.eval()
        colab_model.to(device)
        ppo_config = PPOConfig(
            seed=args.seed,
            batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
        )
        ppo_trainer = PPOTrainer(config=ppo_config, model=model,tokenizer=tokenizer)
        for epoch in range(args.epochs):
            prog_iter = tqdm(train_loader, leave=False, desc='Training')
            for batch in prog_iter:
                query_tensor=batch['inputs']['input_ids'].to(device)
                response_tensor=model.generate(input_ids=query_tensor,attention_mask=batch['inputs']['attention_mask'].to(device))
                decoded_response=[tokenizer.decode(i[1:-1]).split(' ') for i in response_tensor]
                response_items=torch.tensor([get_keys_by_value(train_data.indices,i) for i in decoded_response])
                print(batch['origin_inters'].max())
                collab_score = colab_model.predict(batch['origin_inters'].to(device),response_items.to(device),batch['positions'].to(device)).sigmoid()
                reward = (response_items.squeeze()==batch['origin_item']).long().to(device).squeeze()+collab_score.squeeze()
                train_stats = ppo_trainer.step(list(query_tensor.unbind(dim=0)), list(response_tensor.unbind(dim=0)), list(reward.unbind(dim=0)))
            save_dir = os.path.join(args.output_dir,epoch)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            ppo_trainer.save_pretrained(save_directory=save_dir)
 
    else:
        raise ValueError("Undefined RL Type") 



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLMRec')
    parser = parse_rl_args(parser)
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_dataset_args(parser)

    args = parser.parse_args()
    
    train(args)
