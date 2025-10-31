import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8):
        super().__init__()
        #hidden_channels = 8 default
        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.2)
        
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1,
                             concat=False, dropout=0.2)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x
        
class custom_gnn(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.gnn = GAT(in_channels=384, hidden_channels=384, out_channels=384).to(self.device)

    def get_type(self):
        return 0


    def forward(self, embeddings, edge_index):
        return self.gnn(embeddings.to(self.device), edge_index.to(self.device))

    def get_node_emb(self, *kwargs):
        return self.forward(kwargs[1], kwargs[2])

