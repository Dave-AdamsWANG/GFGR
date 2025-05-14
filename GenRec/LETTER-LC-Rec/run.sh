export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0


BASE_MODEL=/root/autodl-tmp/llama-7b
DATA_PATH=../data
OUTPUT_DIR=./ckpt/$DATASET/


DATA_PATH=/root/LETTER/data
DATASET="Beauty"
CKPT_DIR=/root/autodl-tmp/ckpt/$DATASET/LC-Rec
RESULTS_FILE=/root/autodl-tmp/results/$DATASET/LC-Rec


python  lora_finetune.py \
    --base_model $BASE_MODEL\
    --output_dir $CKPT_DIR \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --per_device_batch_size 16 \
    --learning_rate 1e-4 \
    --epochs 4 \
    --tasks seqrec \
    --train_prompt_sample_num 1 \
    --train_data_sample_num 0 \
    --index_file .index.json\
    --temperature 1.0


    