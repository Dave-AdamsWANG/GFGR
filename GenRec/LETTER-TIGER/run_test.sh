DATASET=fashion
DATA_PATH=/root/autodl-tmp/data
RESULTS_FILE=/root/autodl-tmp/results/$DATASET/
NEG_NUM=5
gfn_weight=0.01
CKPT_PATH=/root/autodl-tmp/ckpt/$DATASET/gfn/$NEG_NUM/$gfn_weight

python test.py \
    --gpu_id 0 \
    --ckpt_path $CKPT_PATH \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --results_file $RESULTS_FILE \
    --test_batch_size 480 \
    --num_beams 20 \
    --test_prompt_ids 0 \
    --index_file .index.epoch10000.alpha0-beta1e-4.json
