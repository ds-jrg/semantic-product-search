import os
import math
import pandas as pd
import networkx as nx
import numpy as np
import torch.nn.functional as F
import time
import torch
import pickle
import random
import tqdm
import math
from sklearn.preprocessing import LabelEncoder
from torch_geometric.utils import from_networkx
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from torch_geometric.nn import GCNConv, FastRGCNConv, RGCNConv, RGATConv
from torch_geometric.nn import GCNConv, FastRGCNConv, RGCNConv, RGATConv 
from torch_geometric.nn import GATConv
from torch_geometric.nn import GraphSAGE
from sklearn.metrics import average_precision_score
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader, random_split
from sentence_transformers import SentenceTransformer, SimilarityFunction
import re
from sklearn.metrics import average_precision_score, ndcg_score
import json
import argparse
from torch_geometric.nn import TransformerConv
from scripts.graphsage import custom_create_graph, custom_data, custom_collate, custom_loss_func

class GraphTransformer(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, out_dim=384, num_layers=3, heads=4, pe_dim=None, dropout=0.1):
        super().__init__()
        self.input_lin = nn.Linear(in_dim, hidden_dim)
        self.pe_lin = nn.Linear(pe_dim, hidden_dim) if pe_dim is not None else None
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.heads = heads
        for _ in range(num_layers):
            # TransformerConv(in_channels, out_channels) where out_channels is per-head out dim.
            out_per_head = hidden_dim // heads
            self.layers.append(TransformerConv(hidden_dim, out_per_head, heads=heads, concat=True, dropout=dropout))
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout))
        self.out_lin = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, pe=None):
        """
        x: [N, in_dim]
        edge_index: [2, E]
        pe: [N, pe_dim] or None
        returns: [N, out_dim] node embeddings
        """
        h = self.input_lin(x)
        if pe is not None and self.pe_lin is not None:
            h = h + self.pe_lin(pe)
        for conv, ln, drop in zip(self.layers, self.norms, self.dropouts):
            h2 = conv(h, edge_index)   # shape [N, hidden_dim] because concat True
            h = ln(h + h2)
            h = drop(h)
        out = self.out_lin(h)  # [N, out_dim]
        return out
        
class custom_gnn(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.k = 8
        self.pe = None
        self.gnn = GraphTransformer(in_dim=386, hidden_dim=512, out_dim=384, num_layers=4, heads=4, pe_dim=self.k).to(self.device)               # pe-dim = self.k Value from compute laplacian

    def forward(self, embeddings, edge_index):
        return self.gnn(embeddings.to(self.device), edge_index.to(self.device), self.pe)

    def get_node_emb(self, *kwargs):
        if self.pe == None:
            self.pe = compute_laplacian_pe(kwargs[0], self.k).to(self.device)
        return self.forward(kwargs[1], kwargs[2])

def compute_laplacian_pe(G, k):
    # returns tensor shape (N, k). If failure or graph big, fallback to degree.
    N = G.number_of_nodes()
    # order nodes by sorted(G.nodes())
    nodes = sorted(G.nodes())
    A = nx.to_numpy_array(G, nodelist=nodes)
    deg = A.sum(axis=1)
    if N <= 4000:  # heuristic
        D_sqrt_inv = np.diag(1.0 / np.sqrt(deg + 1e-12))
        L = np.eye(N) - D_sqrt_inv @ A @ D_sqrt_inv
        # compute eigenvectors
        eigvals, eigvecs = np.linalg.eigh(L)  # symmetric
        idx = np.argsort(eigvals)
        eigvecs = eigvecs[:, idx]
        # skip first constant eigenvector
        if k + 1 <= eigvecs.shape[1]:
            pe = eigvecs[:, 1:k+1]
        else:
            # if graph too small, pad
            pe = np.zeros((N, k))
            available = eigvecs.shape[1] - 1
            if available > 0:
                pe[:, :available] = eigvecs[:, 1:1+available]
        return torch.tensor(pe, dtype=torch.float32)
    else:
        # fallback: simple 1D degree feature + normalized degree repeated
        deg_norm = (deg - deg.mean()) / (deg.std() + 1e-12)
        pe = np.repeat(deg_norm.reshape(-1,1), k, axis=1)
        return torch.tensor(pe, dtype=torch.float32)

