import torch
import torch.nn.functional as F

from torch_geometric.nn import GATConv


class Model(torch.nn.Module):
    def __init__(
            self,
            in_c: int,
            h_c: int,
            out_c: int,
            dropout: float = 0.1,
            heads: int = 4
    ):
        super(Model, self).__init__()
        self.heads = heads
        self.dropout = dropout
        self.gat_conv1 = GATConv(in_c, h_c, heads=self.heads, dropout=self.dropout)
        self.gat_conv2 = GATConv(h_c * heads, out_c, concat=False, dropout=self.dropout)

    def reset_parameters(self):
        self.gat_conv1.reset_parameters()
        self.gat_conv2.reset_parameters()

    def forward(self, x, edge_index, **kwargs):
        x = self.gat_conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat_conv2(x, edge_index)
        return F.log_softmax(x, dim=-1)
