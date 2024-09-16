from typing import Optional, Callable, List
import os.path as osp

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_sparse import coalesce


def read_dgraphfin(folder):
    print('read_dgraph')
    names = ['dgraph.npz']
    items = [np.load(folder + '/' + name) for name in names]

    x = items[0]['x']
    y = items[0]['y'].reshape(-1, 1)
    edge_index = items[0]['edge_index']
    edge_type = items[0]['edge_type']
    train_mask = items[0]['train_mask']
    valid_mask = items[0]['valid_mask']
    test_mask = items[0]['test_mask']

    x = torch.tensor(x, dtype=torch.float).contiguous()
    y = torch.tensor(y, dtype=torch.int64)
    edge_index = torch.tensor(edge_index.transpose(), dtype=torch.int64).contiguous()
    edge_type = torch.tensor(edge_type, dtype=torch.float)
    train_mask = torch.tensor(train_mask, dtype=torch.int64)
    valid_mask = torch.tensor(valid_mask, dtype=torch.int64)
    test_mask = torch.tensor(test_mask, dtype=torch.int64)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_type, y=y)
    data.train_mask = train_mask
    data.valid_mask = valid_mask
    data.test_mask = test_mask

    return data


class DGraphFin(InMemoryDataset):
    r"""
    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"dgraphfin"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = ''

    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        names = ['dgraph.npz']
        return names

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        pass

    #         for name in self.raw_file_names:
    #             download_url('{}/{}'.format(self.url, name), self.raw_dir)

    def process(self):
        data = read_dgraphfin(self.raw_dir)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}()'


def load_data(folder, dataset_name, force_to_symmetric: bool = True):
    dataset = DGraphFin(root=folder, name=dataset_name, transform=T.ToSparseTensor())

    nlabels = dataset.num_classes
    if dataset_name in ['DGraph']:
        nlabels = 2  # 本实验中仅需预测类0和类1

    data = dataset[0]
    if force_to_symmetric:
        data.adj_t = data.adj_t.to_symmetric()  # 将有向图转化为无向图

    if data.edge_index is None:
        row, col, _ = data.adj_t.t().coo()
        data.edge_index = torch.stack([row, col], dim=0)

    if dataset_name in ['DGraph']:
        x = data.x
        x = (x - x.mean(0)) / x.std(0)
        data.x = x
    if data.y.dim() == 2:
        data.y = data.y.squeeze(1)

    split_idx = {'train': data.train_mask, 'valid': data.valid_mask, 'test': data.test_mask}  # 划分训练集，验证集

    train_idx = split_idx['train']
    return data


def get_adj_nodes(data, idx: int, num_nodes: int):
    if isinstance(idx, int):
        idx = Tensor([idx]).long()
    if idx.dim() == 0:
        idx = idx.unsqueeze(0)
    x: Tensor = data.x[idx]
    y: Tensor = data.y[idx]

    # 从 adj_t 中获取 idx 节点的邻居节点
    _, neighbors, distances = data.adj_t[idx.item()].coo()  # 提取行对应的邻居节点索引

    if distances is not None:
        neighbors = {i.item(): j.item() for i, j in zip(neighbors, distances)}
        neighbors = sorted(neighbors.items(), key=lambda x: x[1])
        distances = Tensor([i[1] for i in neighbors])
        neighbors = Tensor([i[0] for i in neighbors])
    else:
        distances = torch.ones(len(neighbors))
    neighbors = data.x[neighbors]
    n_neighbors = len(neighbors)
    if len(neighbors) < num_nodes - 1:
        n_pad = num_nodes - 1 - len(neighbors)
        neighbors = F.pad(neighbors, (0, 0, 0, n_pad), value=0.0)
        distances = F.pad(distances, (0, n_pad), value=0.0) if distances is not None else None
    else:
        neighbors = neighbors[:num_nodes - 1]
        distances = distances[:num_nodes - 1] if distances is not None else None
    return (
        torch.cat([x, neighbors], dim=0),  # (num_nodes, feature_dim) 邻居节点和 idx 节点的特征拼接
        F.pad(distances, (1, 0), value=0.0),  # (num_nodes, ) 距离信息拼接
        y.squeeze(-1),  # scaler 标签
        n_neighbors  # 邻居节点数
    )


class AdjacentNodesDataset(Dataset):
    def __init__(self, data, indexs, num_nodes):
        super(AdjacentNodesDataset, self).__init__()
        self.data = data
        self.indexs = indexs
        self.num_nodes = num_nodes

    def __len__(self):
        return len(self.indexs)

    def __getitem__(self, idx):
        idx: Tensor = self.indexs[idx]
        return get_adj_nodes(self.data, idx, self.num_nodes)

    @staticmethod
    def collate_fn(batch):
        xs, distances, ys, n_neighbors_list = zip(*batch)

        xs = torch.stack(xs)
        distances = torch.stack(distances)
        ys = torch.stack(ys)
        n_neighbors_list = torch.tensor(n_neighbors_list)
        return xs, distances, ys, n_neighbors_list
