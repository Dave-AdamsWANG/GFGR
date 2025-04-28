export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0



# bash run.sh
DATA_PATH=/root/LETTER/data
# Baseline 
# DATASET=fashion
seed_list=(42)
data_list=("Yelp" "Instruments" "Beauty")
rl_typ=dpo
for DATASET in ${data_list[@]}
do
CKPT_DIR=/root/autodl-tmp/ckpt/$DATASET/$rl_typ
ORG_CKPT_DIR=/root/autodl-tmp/org_chkpt/ckpt/$DATASET/
RESULTS_FILE=/root/autodl-tmp/results/$DATASET/$rl_typ
for seed in ${seed_list[@]}
do
    python ./finetune_rl.py \
        --output_dir $CKPT_DIR \
        --rl_ckpt_path $ORG_CKPT_DIR\
        --dataset $DATASET \
        --per_device_batch_size 1024 \
        --data_path $DATA_PATH \
        --learning_rate 1e-5 \
        --epochs 2 \
        --index_file .index.json \
        --temperature 1.0 \
        --seed ${seed}

    python test.py \
        --gpu_id 0 \
        --ckpt_path $CKPT_DIR \
        --dataset $DATASET \
        --data_path $DATA_PATH \
        --results_file $RESULTS_FILE \
        --test_batch_size 480 \
        --num_beams 20 \
        --test_prompt_ids 0 \
        --index_file .index.json
done
done