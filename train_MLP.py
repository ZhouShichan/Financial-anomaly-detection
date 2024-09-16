from tqdm import tqdm
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from models.MLP import Model
from utils.dgraphfin import load_data
from utils.evaluator import Evaluator


def train(model, data, train_idx, optimizer):
    model.train()
    optimizer.zero_grad()

    out = model(data.x[train_idx])
    loss = F.nll_loss(out, data.y[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


def valid(model, data_loader, split_idx, evaluator):
    with torch.no_grad():
        model.eval()
        out = model(data.x)
        y_pred = out.exp()
        losses, eval_results = dict(), dict()
        for key in ['train', 'valid']:
            node_id = split_idx[key]
            losses[key] = F.nll_loss(out[node_id], data.y[node_id]).item()
            eval_results[key] = evaluator.eval(data.y[node_id], y_pred[node_id])['auc']
        return eval_results, losses


def train_epoch(
    model, data, optimizer, evaluator, lr, min_valid_loss, epoch, model_desc, stop_count,
    use_early_stop=False,
    use_lr_scheduler=False
):
    split_idx = {'train': data.train_mask, 'valid': data.valid_mask, 'test': data.test_mask}  # 划分训练集，验证集
    loss = train(model, data, data.train_mask, optimizer)
    eval_results, losses = valid(model, data, split_idx, evaluator)
    valid_loss = losses['valid']
    early_stop = False
    # 保存最好的模型
    if valid_loss < min_valid_loss:
        stop_count = 0
        model_save_path = Path(f'results/model-{model_desc}.pt')
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_save_path)
        min_valid_loss = valid_loss
    else:
        stop_count += 1
        if stop_count == 5 and use_lr_scheduler:
            for param_group in optimizer.param_groups:
                lr *= 0.5
                param_group['lr'] = 0.5
        if stop_count == 10 and use_early_stop:
            early_stop = True

    train_log = {
        'epoch': epoch,
        't.loss': losses['train'],
        't.auc': eval_results['train'],
        'v.loss': losses['valid'],
        'v.auc': eval_results['valid'],
        'lr': lr,
        's.cnt': stop_count,
        'min.v.loss': min_valid_loss,
    }
    with open(f'results/train_log-{model_desc}.csv', 'a' if epoch > 0 else 'w', newline='') as f:
        pd.DataFrame({k: [v] for k, v in train_log.items()}).to_csv(f, header=f.tell() == 0, index=False)
    return min_valid_loss, lr, stop_count, early_stop, train_log


if __name__ == '__main__':
    # 设置gpu设备
    device = 0
    device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(f'Using device: {device}')

    # 加载数据
    data = load_data('./datasets/632d74d4e2843a53167ee9a1-momodel/', 'DGraph', force_to_symmetric=True)
    data = data.to(device)

    lr = 0.005
    print(f'batch_size: all data, lr: {lr}')

    model_params = {
        "hidden_channels": 16,
        "num_layers": 2,
        "dropout": 0.1,
    }

    model = Model(
        in_channels=20,
        out_channels=2,
        batchnorm=False,
        ** model_params
    )
    model_desc = f'MLP-{"-".join([f"{k}_{v}" for k, v in model_params.items() ])}'

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    evaluator = Evaluator('auc')

    min_valid_loss = 1e10
    stop_count = 0
    epoch_iter = tqdm(range(0, 600))
    for epoch in epoch_iter:
        min_valid_loss, lr, stop_count, early_stop, train_log = train_epoch(
            model, data, optimizer, evaluator, lr, min_valid_loss, epoch, model_desc, stop_count
        )
        epoch_iter.set_postfix(**train_log)
        if early_stop:
            break
    if early_stop:
        print(f'Early stop at epoch {epoch}')
    else:
        print(f'Training finished with {epoch} epochs')
