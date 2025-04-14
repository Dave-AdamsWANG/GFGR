# python ./RQ-VAE/main.py \
#   --device cuda:0 \
#   --data_path ./data/Instruments/Instruments.emb-llama-td-pca.npy\
#   --alpha 0.01 \
#   --beta 0.0001 \
#   --cf_emb ./RQ-VAE/ckpt/Instruments-32d-sasrec.pt\
#   --ckpt_dir ./checkpoint/

  # python ./RQ-VAE/main.py \
  # --device cuda:0 \
  # --data_path ./data/Instruments/Instruments.emb-llama-td-pca.npy\
  # --alpha 0.1 \
  # --beta 0.0001 \
  # --cf_emb ./RQ-VAE/ckpt/Instruments-32d-sasrec.pt\
  # --ckpt_dir ./checkpoint/

  #   python ./RQ-VAE/main.py \
  # --device cuda:0 \
  # --data_path ./data/Instruments/Instruments.emb-llama-td-pca.npy\
  # --alpha 0.0 \
  # --epochs 8000 \
  # --eval_step 800 \
  # --beta 0.0000 \
  # --cf_emb ./RQ-VAE/ckpt/Instruments-32d-sasrec.pt\
  # --ckpt_dir ./checkpoint/


  # python ./RQ-VAE/main.py \
  # --device cuda:0 \
  # --data music\
  # --root_path /root/autodl-tmp/data/\
  # --alpha 0.0 \
  # --epochs 10000 \
  # --eval_step 1000 \
  # --beta 0.0000 \
  # --cf_emb 0 \
  # --lr 0.001\

    python ./RQ-VAE/main.py \
  --device cuda:0 \
  --data yelp\
  --root_path /root/autodl-tmp/data/\
  --alpha 0.0 \
  --epochs 20000 \
  --eval_step 1500 \
  --beta 0.0000 \
  --cf_emb 0 \
  --lr 0.001\