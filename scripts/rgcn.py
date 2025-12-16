import torch
from torch import nn
from torch_geometric.nn import RGCNConv

from sklearn.preprocessing import LabelEncoder
import networkx as nx
import pandas as pd
import numpy as np

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class RGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations):
        super().__init__()
        self.conv1 = RGCNConv(in_channels,  hidden_channels, num_relations)
        self.conv2 = RGCNConv(hidden_channels, hidden_channels,   num_relations)
        self.conv3 = RGCNConv(hidden_channels, out_channels,   num_relations)

    def forward(self, x, edge_index, edge_type):
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_type))
        x = self.conv3(x, edge_index, edge_type)
        return x
        
class custom_gnn(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.gnn = RGCN(
            in_channels=384,
            hidden_channels=384,      # <= 384     #128->0.45
            out_channels=384,
            num_relations=2
        ).to(self.device)

    def get_type(self):
        return 1

    def forward(self, embeddings, edge_index, edge_type):
        return self.gnn(embeddings.to(self.device), edge_index.to(self.device), edge_type.to(self.device))

    def get_node_emb(self, *kwargs):
        return self.forward(kwargs[1], kwargs[2], kwargs[3])

