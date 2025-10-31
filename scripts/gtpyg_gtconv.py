import torch
from torch import nn

from gtpyg.gt_pyg.nn.gt_conv import GTConv


class custom_gnn(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.gnn = GTConv(
            node_in_dim=384,
            edge_in_dim=1,  #  MAYBE CHANGE HERE
            hidden_dim=384,
            num_heads=2 # hidden dim % num_heads == 0   
            ).to(device)
       
    def forward(self, embeddings, edge_index, edge_attr):
        res = self.gnn(embeddings.to(self.device), edge_index.to(self.device), edge_attr.to(self.device))
        if isinstance(res,tuple):
            return res[0]
        return res

    def get_type(self):
        return 2
        
    def get_node_emb(self, *kwargs):
        return self.forward(kwargs[1], kwargs[2], kwargs[3])
