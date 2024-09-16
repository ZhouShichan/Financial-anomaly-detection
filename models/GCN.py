import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv


class Model(torch.nn.Module):
    def __init__(
            self,
            in_c: int,
            h_c: int,
            out_c: int,
            n_layers: int = 2,
            dropout: float = 0.1,
            normalize: bool = True,
    ):
        super(Model, self).__init__()
        self.n_layers = n_layers

        self.convs = torch.nn.ModuleList([
            GCNConv(
                in_c if i == 0 else h_c,
                h_c if i != n_layers - 1 else out_c,
                normalize=normalize,
                cached=True
            )
            for i in range(n_layers)
        ])
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None, **kwargs):
        for i in range(self.n_layers - 1):
            x = self.convs[i](x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=-1)
