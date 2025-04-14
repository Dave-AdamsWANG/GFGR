#!/bin/bash

gpu_id=0
dataset_list=("music" "yelp")


# for dataset in "${dataset_list[@]}"
# do
#     python data_process/amazon_text_emb.py --dataset ${dataset} 
#     python data_process/pca.py --dataset ${dataset} 
# done


bash RQ-VAE/train_tokenizer.sh #&& /usr/bin/shutdown


# python data_process/amazon_text_emb.py && /usr/bin/shutdown
# bash RQ-VAE/train_tokenizer.sh 
# bash RQ-VAE/tokenize.sh 
# cd LETTER-TIGER
# bash run_train.sh && /usr/bin/shutdown
# cd LETTER-LC-Rec
# bash run_train.sh 
