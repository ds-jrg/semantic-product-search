# Crashes when using edge_attr, without having defined edge_dim

import torch
from torch import nn
from torch_geometric.nn import TransformerConv

from sklearn.preprocessing import LabelEncoder
import networkx as nx
import pandas as pd
       
class custom_gnn(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.gnn = TransformerConv(
            in_channels=384,
            out_channels=384, 
            edge_dim=1, # Needed otherwise memory error
            heads=1 # hidden dim % num_heads == 0   
            ).to(self.device)

    def get_type(self):
        return 2

    def forward(self, embeddings, edge_index, edge_attr=None):
        #return self.gnn(embeddings.to(self.device), edge_index.to(self.device), None)
        return self.gnn(embeddings.to(self.device), edge_index.to(self.device), edge_attr.to(self.device))
        
    def get_node_emb(self, *kwargs):
        return self.forward(kwargs[1], kwargs[2], kwargs[3])


