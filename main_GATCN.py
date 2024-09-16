from tqdm import tqdm
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from models.GATCN import Model
from utils.dgraphfin import load_data, AdjacentNodesDataset
from utils.evaluator import Evaluator

# 设置gpu设备
device = 0
device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

data = load_data('./datasets/632d74d4e2843a53167ee9a1-momodel/', 'DGraph', force_to_symmetric=True)
data = data.to(device)

model_params = {
    "h_c": 16,
    "heads": 2,
    "dropout": 0.1,
}


model = Model(
    in_c=20,
    out_c=2,
    ** model_params
).to(device)
model_desc = f'GATCN-{"-".join([f"{k}_{v}" for k, v in model_params.items() ])}'
model_save_path = f'results/model-{model_desc}.pt'
model.load_state_dict(torch.load(model_save_path, map_location=device))

cache_path = f'./results/out-{model_desc}.pt'


def predict(data, node_id):
    if os.path.exists(cache_path):
        out = torch.load(cache_path, map_location=device)
    else:
        with torch.no_grad():
            model.eval()
            out = model(data.x, data.edge_index)

    pred = out[node_id].exp()
    return pred.squeeze(0)


if __name__ == '__main__':
    dic = {0: "正常用户", 1: "欺诈用户"}
    node_idx = 0
    y_pred = predict(data, node_idx)
    print(y_pred)
    print(f'节点 {node_idx} 预测对应的标签为:{torch.argmax(y_pred)}, 为{dic[torch.argmax(y_pred).item()]}。')

    node_idx = 1
    y_pred = predict(data, node_idx)
    print(y_pred)
    print(f'节点 {node_idx} 预测对应的标签为:{torch.argmax(y_pred)}, 为{dic[torch.argmax(y_pred).item()]}。')
