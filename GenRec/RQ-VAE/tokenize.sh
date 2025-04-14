python ./RQ-VAE/generate_indices.py\
    --alpha 0 \
    --beta 0 \
    --epoch 10000 \
    --data music\
    --root_path ~/autodl-tmp/data/RQ-VAE-checkpoint/music/Apr-01-2025_15-06-08/ \
    --checkpoint best_collision_model.pth


python ./RQ-VAE/generate_indices.py\
    --alpha 0 \
    --beta 0 \
    --epoch 10000 \
    --data yelp\
    --root_path ~/autodl-tmp/data/RQ-VAE-checkpoint/yelp/Apr-02-2025_11-18-10/ \
    --checkpoint best_collision_model.pth
