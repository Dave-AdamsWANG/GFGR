gpu_id=0
dataset_list=("Instruments" "Beauty" "Yelp")


for dataset in "${dataset_list[@]}"
do
  python ./RQ-VAE/main.py \
  --device cuda:0 \
  --data $dataset\
  --root_path /root/LETTER/data/\
  --alpha 0.0 \
  --epochs 10000 \
  --eval_step 2000 \
  --beta 0.0000 \
  --cf_emb 0 \
  --lr 0.001\
  
done