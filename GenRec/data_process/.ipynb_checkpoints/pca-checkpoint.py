import numpy as np
import argparse
import os 
from sklearn.decomposition import PCA
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='music', help='music / beauty / fashion / toys /yelp')
    parser.add_argument('--root', type=str, default="/root/autodl-tmp/data")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    args.root = os.path.join(args.root, args.dataset)
    embeddings = np.load(args.root+f'/{args.dataset}.emb-llama-td.npy')
    pca = PCA(n_components=768)
    pca_item_emb = pca.fit_transform(embeddings.reshape(-1,4096))
    np.save(args.root+f'/{args.dataset}.emb-llama-td-pca.npy',pca_item_emb)

