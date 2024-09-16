from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from models.MLP import Model
from utils.dgraphfin import load_data, AdjacentNodesDataset
from utils.evaluator import Evaluator


def train(model, data_loader, optimizer):
    model.train()
    device = next(model.parameters()).device
    optimizer.zero_grad()
    losses = []

    def train_batch(x: Tensor, x_mark: Tensor, y: Tensor):
        x_mark = x_mark.unsqueeze(-1)
        x = torch.flatten(torch.cat((x, x_mark), dim=-1), 1)
        pred = model(x)
        loss = F.nll_loss(pred, y)
        loss.backward()
        optimizer.step()
        return loss.item()
    data_iter = tqdm(data_loader, desc='Training')
    for x, x_mark, y, _ in data_iter:
        x, x_mark, y = x.to(device), x_mark.to(device), y.to(device)
        loss = train_batch(x, x_mark, y)
        losses.append(loss)
        data_iter.set_postfix(loss=np.average(losses))
    return np.average(losses)


def valid(model, data_loader):
    with torch.no_grad():
        model.eval()
        device = next(model.parameters()).device
        losses = []
        targets = []
        preds = []

        data_iter = tqdm(data_loader, desc='Validating')
        for x, x_mark, y, _ in data_iter:
            x, x_mark, y = x.to(device), x_mark.to(device), y.to(device)
            x_mark = x_mark.unsqueeze(-1)
            x = torch.flatten(torch.cat((x, x_mark), dim=-1), 1)
            pred = model(x)
            loss = F.nll_loss(pred, y)
            losses.append(loss.item())
            targets.append(y)
            preds.append(pred.exp())
            data_iter.set_postfix(loss=np.average(losses))
        targets = torch.cat(targets, dim=0)
        preds = torch.cat(preds, dim=0)
        auc = Evaluator('auc').eval(targets, preds)['auc']
        return np.average(losses), auc


if __name__ == '__main__':
    # 设置gpu设备
    device = 0
    device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(f'Using device: {device}')

    # 加载数据
    data = load_data('./datasets/632d74d4e2843a53167ee9a1-momodel/', 'DGraph', force_to_symmetric=True)

    num_nodes = 8  # 节点数量，占比超过 99%

    batch_size = 1024
    lr = 0.001
    print(f'batch_size: {batch_size}, lr: {lr}')

    train_dataloader = DataLoader(
        AdjacentNodesDataset(data, data.train_mask, num_nodes),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=AdjacentNodesDataset.collate_fn
    )
    valid_dataloader = DataLoader(
        AdjacentNodesDataset(data, data.valid_mask, num_nodes),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=AdjacentNodesDataset.collate_fn
    )

    model_params = {
        "hidden_channels": 16,
        "num_layers": 2,
        "dropout": 0.1,
    }

    model = Model(
        in_channels=num_nodes * (20 + 1),
        out_channels=2,
        batchnorm=False,
        ** model_params
    )
    model_desc = f'MLP-AdjNodes_{num_nodes}-{"-".join([f"{k}_{v}" for k, v in model_params.items() ])}'

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    min_valid_loss = 1e10
    stop_count = 0

    for epoch in range(0, 600):
        loss = train(model, train_dataloader, optimizer)
        valid_loss, valid_auc = valid(model, valid_dataloader)
        train_log = {
            'epoch': epoch,
            'train_loss': loss,
            'valid_loss': valid_loss,
            'valid_auc': valid_auc,
        }
        print(" \t ".join([f"{k}: {v:.4f}" if k != 'epoch' else f"{k}: {v:3d}" for k, v in train_log.items()]))
        with open(f'results/train_log-{model_desc}.csv', 'a' if epoch > 0 else 'w', newline='') as f:
            train_log = {k: [v] for k, v in train_log.items()}
            pd.DataFrame(train_log).to_csv(f, header=f.tell() == 0, index=False)
        # 保存最好的模型
        if valid_loss < min_valid_loss:
            stop_count = 0
            model_save_path = f'results/model-{model_desc}.pt'
            torch.save(model.state_dict(), model_save_path)
            print(f"valid_loss improved from {min_valid_loss:.4f} to {valid_loss:.4f}, save model to {model_save_path}")
            min_valid_loss = valid_loss
        else:
            stop_count += 1
            print(f"valid_loss did not improve, stop_count: {stop_count}")
            if stop_count == 5:
                for param_group in optimizer.param_groups:
                    lr *= 0.5
                    param_group['lr'] = 0.5
                print(f"learning rate decay to {lr:.4f}")
            if stop_count == 10:
                print(f"valid_loss did not improve for {stop_count} epochs, early stopping")
                break
