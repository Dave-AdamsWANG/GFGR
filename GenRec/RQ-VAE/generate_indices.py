import collections
import json
import logging

import numpy as np
import torch
from time import time
from torch import optim
from tqdm import tqdm

from torch.utils.data import DataLoader

from datasets import EmbDataset
from models.rqvae import RQVAE
import argparse
import os

def check_collision(all_indices_str):
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    return tot_item==tot_indice

def get_indices_count(all_indices_str):
    indices_count = collections.defaultdict(int)
    for index in all_indices_str:
        indices_count[index] += 1
    return indices_count

def get_collision_item(all_indices_str):
    index2id = {}
    for i, index in enumerate(all_indices_str):
        if index not in index2id:
            index2id[index] = []
        index2id[index].append(i)

    collision_item_groups = []

    for index in index2id:
        if len(index2id[index]) > 1:
            collision_item_groups.append(index2id[index])

    return collision_item_groups

def parse_args():
    parser = argparse.ArgumentParser(description="RQ-VAE")
    parser.add_argument("--dataset", type=str,default="Instruments", help='dataset')
    parser.add_argument("--root_path", type=str,default="../checkpoint/", help='root path')
    parser.add_argument('--alpha', type=str, default='1e-1', help='cf loss weight')
    parser.add_argument('--epoch', type=int, default='10000', help='epoch')
    parser.add_argument('--checkpoint', type=str, default='epoch_9999_collision_0.0012_model.pth', help='checkpoint name')
    parser.add_argument('--beta', type=str, default='1e-4', help='div loss weight')


    return parser.parse_args()

args_setting = parse_args()

dataset = args_setting.dataset
ckpt_path = args_setting.root_path+args_setting.checkpoint

output_dir = f"/root/autodl-tmp/data/{dataset}/"
output_file = f"{dataset}.index.epoch{args_setting.epoch}.alpha{args_setting.alpha}-beta{args_setting.beta}.json"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir,output_file)
device = torch.device("cuda:0")

ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'),weights_only=False)
args = ckpt["args"]
state_dict = ckpt["state_dict"]


data = EmbDataset(args.data_path)

model = RQVAE(in_dim=data.dim,
                  num_emb_list=args.num_emb_list,
                  e_dim=args.e_dim,
                  layers=args.layers,
                  dropout_prob=args.dropout_prob,
                  bn=args.bn,
                  loss_type=args.loss_type,
                  quant_loss_weight=args.quant_loss_weight,
                  kmeans_init=args.kmeans_init,
                  kmeans_iters=args.kmeans_iters,
                  sk_epsilons=args.sk_epsilons,
                  sk_iters=args.sk_iters,
                  )

model.load_state_dict(state_dict,strict=False)
model = model.to(device)
model.eval()
print(model)

data_loader = DataLoader(data,num_workers=args.num_workers,
                             batch_size=64, shuffle=False,
                             pin_memory=True)

all_indices = []
all_indices_str = []
prefix = ["<a_{}>","<b_{}>","<c_{}>","<d_{}>","<e_{}>","<f_{}>"]

def constrained_km(data, n_clusters=10):
    from k_means_constrained import KMeansConstrained 
    # x = data.cpu().detach().numpy()
    # data = self.embedding.weight.cpu().detach().numpy()
    x = data
    size_min = min(len(data) // (n_clusters * 2), 10)
    clf = KMeansConstrained(n_clusters=n_clusters, size_min=size_min, size_max=n_clusters * 6, max_iter=10, n_init=10,
                            n_jobs=10, verbose=False)
    clf.fit(x)
    t_centers = torch.from_numpy(clf.cluster_centers_)
    t_labels = torch.from_numpy(clf.labels_).tolist()
    return t_centers, t_labels

labels = {"0":[],"1":[],"2":[], "3":[]}
embs  = [layer.embedding.weight.cpu().detach().numpy() for layer in model.rq.vq_layers]


for idx, emb in enumerate(embs):
    centers, label = constrained_km(emb)
    labels[str(idx)] = label
for d in tqdm(data_loader):
    d, emb_idx = d[0], d[1]
    d = d.to(device)
    
    # indices = model.get_indices(d, use_sk=False)
    indices = model.get_indices(d, labels,use_sk=False)

    indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
    for index in indices:
        code = []
        for i, ind in enumerate(index):
            code.append(prefix[i].format(int(ind)))

        all_indices.append(code)
        all_indices_str.append(str(code))
    # break

all_indices = np.array(all_indices)
all_indices_str = np.array(all_indices_str)

for vq in model.rq.vq_layers[:-1]:
    vq.sk_epsilon=0.0
# model.rq.vq_layers[-1].sk_epsilon = 0.005
if model.rq.vq_layers[-1].sk_epsilon == 0.0:
    model.rq.vq_layers[-1].sk_epsilon = 0.003

# model.rq.vq_layers[-1].sk_epsilon = 0.1
tt = 0
origin_len = all_indices.shape[-1]
add_max=0
#There are often duplicate items in the dataset, and we no longer differentiate them
if all_indices.shape[-1] < origin_len+1:
    all_indices = np.c_[all_indices,np.repeat(prefix[all_indices.shape[-1]].format(0), all_indices.shape[0]).reshape(-1, 1)]
while True:
    if check_collision(all_indices_str):
        break
# tt >= 20 or
    collision_item_groups = get_collision_item(all_indices_str)
    # print(collision_item_groups)
    print(len(collision_item_groups))
    all_indices_new = {}
    for collision_items in collision_item_groups:
        d = data[collision_items]
        d = d[0].to(device)
        indices = model.get_indices(d, labels, use_sk=True)

        # indices = model.get_indices(d, use_sk=True)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        collision_num = len(collision_items)
        for i, (item, index) in enumerate(zip(collision_items, indices)):
            code = []
            for j, ind in enumerate(index):
                code.append(prefix[j].format(int(ind)))
            if i>0:
                if tt==0:
                    code.append(prefix[origin_len].format(int(i)))
                else: 
                    code.append(prefix[origin_len].format(int(i+add_max)))
            else:
                code.append(prefix[origin_len].format(int(i)))
            all_indices_new[item] = code
        add_max += collision_num
            
    for key in all_indices_new:
        all_indices[key] = all_indices_new[key]
    all_indices_str = np.array([str(i) for i in all_indices])
    tt += 1
    if tt%8==5:
        origin_len+=1
        add_max=0


print("All indices number: ",len(all_indices))
print("Max number of conflicts: ", max(get_indices_count(all_indices_str).values()))

tot_item = len(all_indices_str)
tot_indice = len(set(all_indices_str.tolist()))
print("Collision Rate",(tot_item-tot_indice)/tot_item)

all_indices_dict = {}
for item, indices in enumerate(all_indices.tolist()):
    all_indices_dict[item] = list(indices)



with open(output_file, 'w') as fp:
    json.dump(all_indices_dict,fp)
