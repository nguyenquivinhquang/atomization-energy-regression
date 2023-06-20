import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch_geometric.nn import GCNConv, GATv2Conv
import torch
from torch_geometric.nn import global_mean_pool


class GAT(torch.nn.Module):
    """Graph Attention Network"""

    def __init__(self, dim_in, dim_h, heads=8, edge_dim=1):
        super().__init__()
        self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads, edge_dim=edge_dim)
        self.gat2 = GATv2Conv(dim_h * heads, dim_h, heads=1, edge_dim=edge_dim)
        self.fc = Linear(dim_h, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, edge_index, batch, edge_weight=None):
        # edge_weight = None
       
        h = self.gat1(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.gat2(h, edge_index, edge_weight)
        feat = F.relu(h)
        feat = global_mean_pool(feat, batch)
        # print(feat.shape)
        feat = self.sigmoid(feat)
        h = self.fc(feat)
        return h
