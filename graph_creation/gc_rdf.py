# Creates a graph depending on gnn_type with edge types, without, with edge attributes. 
# Product nodes and Attribute nodes
# Edges between product and shared attribute

import torch
from torch import nn
import random

from sklearn.preprocessing import LabelEncoder
import networkx as nx
import pandas as pd
import numpy as np


class GraphCreator:
    def __init__(self, df_products, gnn_type, sentence_transformer_model):
        self._df_products = df_products
        self._gnn_type = gnn_type
        self._sentence_transformer_model = sentence_transformer_model

        self._add_edges = 2

    def custom_create_graph(self):
        G = nx.Graph()
        
        if 'product_title' in self._df_products.columns:
            product_infos = ['product_title', 'product_description', 'product_bullet_point'] 
            attributes = ['product_brand', 'product_color']
        else:
            product_infos = ['product_name', 'product_description', 'product_features']
            attributes = ['product_class', 'category hierarchy']

        # Add product nodes first, to assign all indices with 0,..., N to products
        for _, row in self._df_products.iterrows():
            pid = row['node_id']
            pt = " ".join(str(row[field]) for field in product_infos if pd.notnull(row[field]))
            G.add_node(pid, name = pt, att0 = row[attributes[0]], att1 = row[attributes[1]])
            
        # Add attribute nodes:
        for at0 in self._df_products[attributes[0]].dropna().unique():
            G.add_node(f"attribute0::{at0}", name=at0, type=attributes[0])
        for at1 in self._df_products[attributes[1]].dropna().unique():
            G.add_node(f"attribute1::{at1}", name=at1, type=attributes[1])
        
        
        # Add edges:
        for _, row in self._df_products.iterrows():
            pid = row['node_id']

            if pd.notna(row[attributes[0]]):
                G.add_edge(pid, f"attribute0::{row[attributes[0]]}", type='has_attribute0')
            if pd.notna(row[attributes[1]]):
                G.add_edge(pid, f"attribute1::{row[attributes[1]]}", type='has_attribute1')

        #for i in G.nodes():
            #print(G.nodes[i])
        names = [G.nodes[node]["name"] for node in G.nodes()]
        embeddings = self._sentence_transformer_model.encode(names)
        embeddings = torch.tensor(embeddings, dtype=torch.float32)  # shape: (N, emb_dim)
        
        adj = nx.adjacency_matrix(G).tocoo()

        if self._gnn_type == 0: # for no edge types, e.g., graphsage
            edge_index = torch.from_numpy(np.vstack((adj.row, adj.col)).astype(np.int64))
            return G, embeddings, edge_index

        # NOT 100% SURE ABOUT THE REST HERE... CHECK AGAIN todo

        nodes_order = list(G.nodes())
        node_idx_map = {node: idx for idx, node in enumerate(nodes_order)}
   
        if self._gnn_type == 1: # for edge_types, e.g., rgcn
            edges = list(G.edges(data=True))
            edge_index_list = []
            edge_type_list = []
            
            edge_type_to_id = {'has_attribute0': 0, 'has_attribute1': 1}
            
            node_idx_map = {node: i for i, node in enumerate(G.nodes())}    
            
            for u, v, data in edges:
                src = node_idx_map[u]
                dst = node_idx_map[v]
                rel_type = edge_type_to_id[data['type']]
                
                # Since FastRGCN expects directed edges, add both directions
                edge_index_list.append([src, dst])
                edge_index_list.append([dst, src])
                edge_type_list.append(rel_type)
                edge_type_list.append(rel_type)
            
            edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
            edge_type = torch.tensor(edge_type_list, dtype=torch.long)
        
            return G, embeddings, edge_index, edge_type
        elif self._gnn_type == 2:
            edge_list = []
            edge_attr_list = []
            for u, v, data in G.edges(data=True):
                src = node_idx_map[u]
                dst = node_idx_map[v]
                edge_list.append((src, dst))
                edge_list.append((dst, src))
                if data.get('type') == 'has_attribute0':
                    edge_attr_list.append([0.0])
                    edge_attr_list.append([0.0])
                elif data.get('type') == 'has_attribute1':
                    edge_attr_list.append([1.0])
                    edge_attr_list.append([1.0])
                else:
                    edge_attr_list.append([0.5])
                    edge_attr_list.append([0.5])
    
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)
            return G, embeddings, edge_index, edge_attr
        
        elif self._gnn_type == 2: # for edge_attributes, e.g., graph transformers
            edge_list = []
            edge_attr_list = []
            for u, v, data in G.edges(data=True):
                edge_list.append((u, v))
                edge_list.append((v, u))  # undirected → add both directions
                if data['type'] == 'has_attribute0':
                    edge_attr_list.append([0.0])
                    edge_attr_list.append([0.0])
                elif data['type'] == 'has_attribute1':
                    edge_attr_list.append([1.0])
                    edge_attr_list.append([1.0])
        
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)
    
            return G, embeddings, edge_index, edge_attr