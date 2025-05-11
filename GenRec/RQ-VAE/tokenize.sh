# python ./RQ-VAE/generate_indices.py\
#     --alpha 0 \
#     --beta 0 \
#     --epoch 10000 \
#     --data Beauty\
#     --root_path ~/LETTER/data/RQ-VAE-checkpoint/Beauty/May-10-2025_10-49-31/ \
#     --checkpoint best_collision_model.pth


python ./RQ-VAE/generate_indices.py\
    --alpha 0 \
    --beta 0 \
    --epoch 10000 \
    --data Yelp\
    --root_path ~/LETTER/data/RQ-VAE-checkpoint/Yelp/May-10-2025_22-01-44/ \
    --checkpoint best_collision_model.pth
