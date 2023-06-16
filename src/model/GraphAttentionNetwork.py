import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch_geometric.nn import GCNConv, GATv2Conv
import torch


class GAT(torch.nn.Module):
    """Graph Attention Network"""
    def __init__(self, dim_in, dim_h, dim_out, heads=8, edge_dim=1):
        super().__init__()
        self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads, edge_dim=edge_dim)
        self.gat2 = GATv2Conv(dim_h*heads, dim_out, heads=1, edge_dim=edge_dim)
        self.fc = Linear(dim_out, 1)
        
    def forward(self, x, edge_index, edge_weight=None):
        h = F.dropout(x, p=0.6, training=self.training)
        if edge_weight is None:
            h = self.gat1(h, edge_index)
        else:
            h = self.gat1(h, edge_index, edge_weight)
        h = F.elu(h)
        h = F.dropout(h, p=0.6, training=self.training)
        if edge_weight is None:
            h = self.gat2(h, edge_index)
        else:
            h = self.gat2(h, edge_index, edge_weight)
        h = F.log_softmax(h, dim=1)
        h = self.fc(h)
        return h