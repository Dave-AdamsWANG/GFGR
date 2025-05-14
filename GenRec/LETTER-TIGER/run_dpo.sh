export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0



# bash run.sh
DATA_PATH=/root/LETTER/data
# Baseline 
# DATASET=fashion
seed_list=(42)
data_list=("Beauty" "Yelp")
rl_type=sprec
for DATASET in ${data_list[@]}
do
for seed in ${seed_list[@]}
do
#CKPT_DIR=/root/autodl-tmp/ckpt/$DATASET/TIGER/$rl_type
CKPT_DIR=/root/autodl-tmp/ckpt/$DATASET/$rl_type
# CKPT_DIR=/root/autodl-tmp/ckpt/$DATASET/$rl_type/epoch0
# ORG_CKPT_DIR=/root/autodl-tmp/org_chkpt_TIGER/ckpt/$DATASET
ORG_CKPT_DIR=/root/autodl-tmp/org_chkpt/ckpt/$DATASET
# ORG_CKPT_DIR=/root/autodl-tmp/ckpt/$DATASET/gfn/1/0.1/
# RESULTS_FILE=/root/autodl-tmp/results/$DATASET/$rl_type/TIGER/
RESULTS_FILE=/root/autodl-tmp/results/$DATASET/$rl_type/
its=3
    for ((i=0;i<$its;i++))
    do
    python ./generate_data.py \
        --rl_ckpt_path $ORG_CKPT_DIR\
        --rl_type $rl_type\
        --per_device_batch_size 1024 \
        --dataset $DATASET \
        --data_path $DATA_PATH \
        --index_file .index.json \
        --temperature 1.0 \
        --seed ${seed}
    CKPT_DIR=/root/autodl-tmp/ckpt/$DATASET/$rl_type/$i
    python ./finetune_rl.py \
        --output_dir $CKPT_DIR \
        --rl_ckpt_path $ORG_CKPT_DIR\
        --rl_type $rl_type\
        --dataset $DATASET \
        --per_device_batch_size 256 \
        --data_path $DATA_PATH \
        --learning_rate 1e-5 \
        --epochs 1 \
        --index_file .index.json \
        --temperature 1.0 \
        --seed ${seed}
    ORG_CKPT_DIR=/root/autodl-tmp/ckpt/$DATASET/$rl_type/$i
    done
    python test.py \
        --gpu_id 0 \
        --ckpt_path $CKPT_DIR \
        --dataset $DATASET \
        --data_path $DATA_PATH \
        --results_file $RESULTS_FILE \
        --test_batch_size 256 \
        --num_beams 20 \
        --test_prompt_ids 0 \
        --index_file .index.json
done
done


# i=0
# CKPT_DIR=/root/autodl-tmp/ckpt/$DATASET/$rl_type/$i
# python ./finetune_rl.py \
#     --output_dir $CKPT_DIR \
#     --rl_ckpt_path $ORG_CKPT_DIR\
#     --rl_type $rl_type\
#     --dataset $DATASET \
#     --per_device_batch_size 256 \
#     --data_path $DATA_PATH \
#     --learning_rate 1e-5 \
#     --epochs 1 \
#     --index_file .index.json \
#     --temperature 1.0 \
#     --seed ${seed}