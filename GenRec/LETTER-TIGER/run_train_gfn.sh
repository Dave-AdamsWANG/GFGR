export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0

DATA_PATH=/root/LETTER/data
# DATASET="Beauty"

# neg_num=5
# weight=0.001
# OUTPUT_DIR=/root/autodl-tmp/ckpt/$DATASET/gfn/${neg_num}/${weight}
# seed_list=(43 44)
    # --collab_reward \    --pretrained

# .index.epoch10000.alpha0-beta0.json "Beauty"
# neg_num_list=(1 3 5)
# weight_list=(1.0 0.1 0.01 0.001)
neg_num_list=(5)
weight_list=(0.02)
data_list=("Beauty")
for DATASET in ${data_list[@]}
do
ORG_CKPT_DIR=/root/autodl-tmp/org_chkpt_TIGER/ckpt/$DATASET/
RESULTS_FILE=/root/autodl-tmp/results/$DATASET/TIGER/
for neg_num in ${neg_num_list[@]}
do
    for weight in ${weight_list[@]}
    do 
    OUTPUT_DIR=/root/autodl-tmp/ckpt/$DATASET/gfn/${neg_num}/${weight}
    python ./finetune_gfn.py \
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --gfn_ckpt_path $ORG_CKPT_DIR \
    --per_device_batch_size 256 \
    --learning_rate 5e-4 \
    --epochs 200 \
    --index_file .new.index.json \
    --temperature 1.0 \
    --gfn_neg_num ${neg_num} \
    --gfn_weight ${weight}  \
    --collab_model_path /root/GFGR/SeqRec/saved/$DATASET/bert4rec/pytorch_model.bin \
    --collab_reward \
    --token_reward \
    --reward_m \
    --reward_label_align \
    --collab_align \
    --reward_weigted_loss \
    --pretrained \
    --align_weight 0.01 \


    python test.py \
        --gpu_id 0 \
        --ckpt_path $OUTPUT_DIR \
        --dataset $DATASET \
        --data_path $DATA_PATH \
        --results_file $RESULTS_FILE \
        --test_batch_size 480 \
        --num_beams 20 \
        --test_prompt_ids 0 \
        --index_file .new.index.json \
        --gfn_neg_num ${neg_num} \
        --gfn_weight ${weight} 
        

    done
done
done

