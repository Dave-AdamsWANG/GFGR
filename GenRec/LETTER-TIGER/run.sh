export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0

DATA_PATH=/root/autodl-tmp/data
# Baseline 
# DATASET=fashion
# CKPT_DIR=/root/autodl-tmp/ckpt/$DATASET/
# RESULTS_FILE=/root/autodl-tmp/results/$DATASET/
# seed_list=(42 43 44)
# for seed in ${seed_list[@]}
# do
#     python ./finetune.py \
#         --output_dir $CKPT_DIR \
#         --dataset $DATASET \
#         --per_device_batch_size 2048 \
#         --learning_rate 5e-4 \
#         --epochs 200 \
#         --index_file .index.epoch10000.alpha0-beta1e-4.json \
#         --temperature 1.0 \
#         --seed ${seed}

#     python test.py \
#         --gpu_id 0 \
#         --ckpt_path $CKPT_DIR \
#         --dataset $DATASET \
#         --data_path $DATA_PATH \
#         --results_file $RESULTS_FILE \
#         --test_batch_size 480 \
#         --num_beams 20 \
#         --test_prompt_ids 0 \
#         --index_file .index.epoch10000.alpha0-beta1e-4.json
# done
# neg_num=5
# weight=0.001
# OUTPUT_DIR=/root/autodl-tmp/ckpt/$DATASET/gfn/${neg_num}/${weight}
#     python ./finetune_gfn.py \
#     --output_dir $OUTPUT_DIR \
#     --dataset $DATASET \
#     --ckpt_path $CKPT_DIR \
#     --per_device_batch_size 256 \
#     --learning_rate 5e-4 \
#     --epochs 200 \
#     --index_file .index.epoch10000.alpha0-beta1e-4.json \
#     --temperature 1.0 \
#     --pretrained \
#     --gfn_neg_num ${neg_num} \
#     --gfn_weight ${weight} 

#     python test.py \
#         --gpu_id 0 \
#         --ckpt_path $OUTPUT_DIR \
#         --dataset $DATASET \
#         --data_path $DATA_PATH \
#         --results_file $RESULTS_FILE \
#         --test_batch_size 480 \
#         --num_beams 20 \
#         --test_prompt_ids 0 \
#         --index_file .index.epoch10000.alpha0-beta1e-4.json


# neg_num_list=(1 3 5)
# weight_list=(1.0 0.1 0.01 0.001)
# for neg_num in ${neg_num_list[@]}
# do
#     for weight in ${weight_list[@]}
#     do 
#     OUTPUT_DIR=/root/autodl-tmp/ckpt/$DATASET/gfn/${neg_num}/${weight}
#     python ./finetune_gfn.py \
#     --output_dir $OUTPUT_DIR \
#     --dataset $DATASET \
#     --ckpt_path $CKPT_DIR \
#     --per_device_batch_size 256 \
#     --learning_rate 5e-4 \
#     --epochs 200 \
#     --index_file .index.epoch10000.alpha0-beta1e-4.json \
#     --temperature 1.0 \
#     --pretrained \
#     --gfn_neg_num ${neg_num} \
#     --gfn_weight ${weight} 

#     python test.py \
#         --gpu_id 0 \
#         --ckpt_path $OUTPUT_DIR \
#         --dataset $DATASET \
#         --data_path $DATA_PATH \
#         --results_file $RESULTS_FILE \
#         --test_batch_size 480 \
#         --num_beams 20 \
#         --test_prompt_ids 0 \
#         --index_file .index.epoch10000.alpha0-beta1e-4.json
#     done
# done

# DATASET=music
# CKPT_DIR=/root/autodl-tmp/ckpt/$DATASET/
# RESULTS_FILE=/root/autodl-tmp/results/$DATASET/
# seed_list=(42)
# for seed in ${seed_list[@]}
# do
#     python ./finetune.py \
#         --output_dir $CKPT_DIR \
#         --dataset $DATASET \
#         --per_device_batch_size 2048 \
#         --learning_rate 5e-4 \
#         --epochs 200 \
#         --index_file .index.epoch10000.alpha0-beta0.json \
#         --temperature 1.0 \
#         --seed ${seed}

#     python test.py \
#         --gpu_id 0 \
#         --ckpt_path $CKPT_DIR \
#         --dataset $DATASET \
#         --data_path $DATA_PATH \
#         --results_file $RESULTS_FILE \
#         --test_batch_size 480 \
#         --num_beams 20 \
#         --test_prompt_ids 0 \
#         --index_file .index.epoch10000.alpha0-beta0.json
# done

# for neg_num in ${neg_num_list[@]}
# do
#     for weight in ${weight_list[@]}
#     do 
#     OUTPUT_DIR=/root/autodl-tmp/ckpt/$DATASET/gfn/${neg_num}/${weight}
#     python ./finetune_gfn.py \
#     --output_dir $OUTPUT_DIR \
#     --dataset $DATASET \
#     --ckpt_path $CKPT_DIR \
#     --per_device_batch_size 512 \
#     --learning_rate 5e-4 \
#     --epochs 200 \
#     --index_file .index.epoch10000.alpha0-beta0.json \
#     --temperature 1.0 \
#     --pretrained \
#     --gfn_neg_num ${neg_num} \
#     --gfn_weight ${weight} 

#     python test.py \
#         --gpu_id 0 \
#         --ckpt_path $CKPT_DIR \
#         --dataset $DATASET \
#         --data_path $DATA_PATH \
#         --results_file $RESULTS_FILE \
#         --test_batch_size 480 \
#         --num_beams 20 \
#         --test_prompt_ids 0 \
#         --index_file .index.epoch10000.alpha0-beta0.json
#     done
# done


DATASET=yelp
CKPT_DIR=/root/autodl-tmp/ckpt/$DATASET/
RESULTS_FILE=/root/autodl-tmp/results/$DATASET/
seed_list=(42)
for seed in ${seed_list[@]}
do
    python ./finetune.py \
        --output_dir $CKPT_DIR \
        --dataset $DATASET \
        --per_device_batch_size 2048 \
        --learning_rate 5e-4 \
        --epochs 200 \
        --index_file .index.epoch10000.alpha0-beta0.json \
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
        --index_file .index.epoch10000.alpha0-beta0.json
done

# for neg_num in ${neg_num_list[@]}
# do
#     for weight in ${weight_list[@]}
#     do 
#     OUTPUT_DIR=/root/autodl-tmp/ckpt/$DATASET/gfn/${neg_num}/${weight}
#     python ./finetune_gfn.py \
#     --output_dir $OUTPUT_DIR \
#     --dataset $DATASET \
#     --ckpt_path $CKPT_DIR \
#     --per_device_batch_size 512 \
#     --learning_rate 5e-4 \
#     --epochs 200 \
#     --index_file .index.epoch10000.alpha0-beta0.json \
#     --temperature 1.0 \
#     --pretrained \
#     --gfn_neg_num ${neg_num} \
#     --gfn_weight ${weight} 

#     python test.py \
#         --gpu_id 0 \
#         --ckpt_path $CKPT_DIR \
#         --dataset $DATASET \
#         --data_path $DATA_PATH \
#         --results_file $RESULTS_FILE \
#         --test_batch_size 480 \
#         --num_beams 20 \
#         --test_prompt_ids 0 \
#         --index_file .index.epoch10000.alpha0-beta0.json
#     done
# done


