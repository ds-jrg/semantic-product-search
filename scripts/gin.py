import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GIN


        
class custom_gnn(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.gnn = GIN(in_channels=384, hidden_channels=384, out_channels=384, num_layers=4, dropout=0.2).to(self.device)

    def get_type(self):
        return 0


    def forward(self, embeddings, edge_index):
        return self.gnn(embeddings.to(self.device), edge_index.to(self.device))

    def get_node_emb(self, *kwargs):
        return self.forward(kwargs[1], kwargs[2])

