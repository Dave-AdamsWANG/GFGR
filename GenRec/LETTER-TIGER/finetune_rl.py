import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
from typing import List
from transformers import EarlyStoppingCallback

import torch
import transformers

from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
from trl import DPOTrainer, DPOConfig
from datasets import Dataset
from modeling_letter import LETTER
# import wandb
from utils import *
from collator import RLCollator

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
    train_data = data_converter(train_data)
    valid_data = data_converter(valid_data)
    config.vocab_size = len(tokenizer)
    if local_rank == 0:
        print("add {} new token.".format(add_num))
        print("data num:", len(train_data))
        tokenizer.save_pretrained(args.output_dir)
        config.save_pretrained(args.output_dir)
        print(train_data[100])
        print(valid_data[100])


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




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLMRec')
    parser = parse_rl_args(parser)
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_dataset_args(parser)

    args = parser.parse_args()
    
    train(args)
