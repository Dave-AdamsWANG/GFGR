#!/bin/bash
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=1
gpu_id=0
dataset_list=("fashion" "music" "yelp")
seed_list=(42 43 44)
model_name_list=("bert4rec" "sasrec_seq" "gru4rec")


for model_name in "${model_name_list[@]}"
do
    for dataset in "${dataset_list[@]}"
    do
        for seed in "${seed_list[@]}"
        do
            python main.py --dataset ${dataset} \
                           --model_name ${model_name} \
                           --hidden_size 32 \
                           --train_batch_size 4096 \
                           --lr 0.001 \
                           --max_len 20 \
                           --gpu_id ${gpu_id} \
                           --num_workers 8 \
                           --train_neg 1 \
                           --warm_epochs 100\
                           --num_train_epochs 200 \
                           --seed ${seed} \
                           --check_path "" \
                           --patience 10 \
                           --log
        done
    done
done

# dataset="fashion"
# model_name="gru4rec"
# for seed in "${seed_list[@]}"
# do
#     python main.py --dataset ${dataset} \
#                    --model_name ${model_name} \
#                    --hidden_size 32 \
#                    --train_batch_size 512 \
#                    --lr 0.001 \
#                    --max_len 20 \
#                    --gpu_id ${gpu_id} \
#                    --num_workers 8 \
#                    --train_neg 1 \
#                    --warm_epochs 20\
#                    --num_train_epochs 200 \
#                    --seed ${seed} \
#                    --check_path "" \
#                    --patience 20 \
#                    --log
# done



# model_name="sasrec_seq"
# dataset="fashion"
# for seed in "${seed_list[@]}"
# do
#     python main.py --dataset ${dataset} \
#                    --model_name ${model_name} \
#                    --hidden_size 32 \
#                    --train_batch_size 512 \
#                    --lr 0.001 \
#                    --max_len 20 \
#                    --gpu_id ${gpu_id} \
#                    --num_workers 8 \
#                    --train_neg 1 \
#                    --warm_epochs 20\
#                    --num_train_epochs 200 \
#                    --seed ${seed} \
#                    --check_path "" \
#                    --patience 20 \
#                    --log
# done
# model_name="bert4rec"
# dataset="fashion"
# for seed in "${seed_list[@]}"
# do
#     python main.py --dataset ${dataset} \
#                    --model_name ${model_name} \
#                    --hidden_size 32 \
#                    --train_batch_size 512 \
#                    --lr 0.001 \
#                    --max_len 20 \
#                    --gpu_id ${gpu_id} \
#                    --num_workers 8 \
#                    --train_neg 1 \
#                    --warm_epochs 20\
#                    --num_train_epochs 200 \
#                    --seed ${seed} \
#                    --check_path "" \
#                    --patience 20 \
#                    --log
# done




# dataset="music"

# for seed in "${seed_list[@]}"
# do
#     python main.py --dataset ${dataset} \
#                    --model_name ${model_name} \
#                    --hidden_size 32 \
#                    --train_batch_size 512 \
#                    --lr 0.001 \
#                    --max_len 20 \
#                    --gpu_id ${gpu_id} \
#                    --num_workers 8 \
#                    --train_neg 1 \
#                    --warm_epochs 70\
#                    --num_train_epochs 200 \
#                    --seed ${seed} \
#                    --check_path "" \
#                    --patience 10 \
#                    --log
# done


# echo "finished"; /usr/bin/shutdown

