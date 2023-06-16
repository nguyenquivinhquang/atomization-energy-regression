from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv
from torch_geometric.nn import global_mean_pool
import torch

class GCN(torch.nn.Module):
    def __init__(self,num_node_features, hidden_channels):
        super(GCN, self).__init__()

        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels//2)
        self.conv3 = GCNConv(hidden_channels//2, hidden_channels//2)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.fc1 = Linear(hidden_channels//2, 1)

        # self.fc2 = Linear(hidden_channels//4, 1)
        # self.lin = Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):

        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        # x = self.sigmoid(x)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        # x = self.sigmoid(x)
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.sigmoid(x)
        x = self.fc1(x)
        
        return x
    
class GraphConv(torch.nn.Module):
    def __init__(self,num_node_features, hidden_channels):
        super(GCN, self).__init__()

        self.conv1 = GraphConv(num_node_features, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels//2)
        self.conv3 = GraphConv(hidden_channels//2, hidden_channels//2)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.fc1 = Linear(hidden_channels//2, 1)


    def forward(self, x, edge_index, batch):

        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        x = global_mean_pool(x, batch) 

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.sigmoid(x)
        x = self.fc1(x)
        
        return x