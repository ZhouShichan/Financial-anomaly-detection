import torch
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_c: int, h_c: int, out_c: int, dropout: float = 0.05):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_c, h_c)
        self.conv2 = SAGEConv(h_c, out_c)
        self.dropout = dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, adj, **kwargs):
        x = self.conv1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, adj)
        return F.log_softmax(x, dim=-1)
