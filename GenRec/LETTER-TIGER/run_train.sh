# export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0

DATASET=fashion
OUTPUT_DIR=/root/autodl-tmp/ckpt/$DATASET/

# torchrun --nproc_per_node=2 --master_port=2314 ./finetune.py \
#     --output_dir $OUTPUT_DIR \
#     --dataset $DATASET \
#     --per_device_batch_size 256 \
#     --learning_rate 5e-4 \
#     --epochs 200 \
#     --index_file .index.json \ .index.epoch10000.alpha1e-2-beta1e-4.json
#     --temperature 1.0

python ./finetune.py \
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --per_device_batch_size 2048 \
    --learning_rate 5e-4 \
    --epochs 200 \
    --index_file .index.epoch10000.alpha0-beta1e-4.json \
    --temperature 1.0