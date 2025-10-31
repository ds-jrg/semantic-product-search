import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, in_channels)
        self.conv2 = GCNConv(in_channels, out_channels)
        self.conv3 = GCNConv(out_channels, out_channels)
        self.conv4 = GCNConv(out_channels, out_channels)
        self.drop = nn.Dropout(p=0.2)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.drop(x)
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = F.relu(self.conv4(x, edge_index))
        return x
        
class custom_gnn(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.gnn = GCN(in_channels=384, out_channels=384).to(self.device)

    def get_type(self):
        return 0

    def forward(self, embeddings, edge_index):
        return self.gnn(embeddings.to(self.device), edge_index.to(self.device))

    def get_node_emb(self, *kwargs):
        return self.forward(kwargs[1], kwargs[2])

