import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GraphSAGE

from sklearn.preprocessing import LabelEncoder
import networkx as nx
import pandas as pd
import numpy as np

from scripts.graphsage import custom_data, custom_collate, custom_loss_func, custom_gnn

# Encoding product tile, description,bulp on its own, concat for embedding
def custom_create_graph(df_products, sentence_transformer_model):
    if 'product_title' in df_products.columns:
        df = df_products.copy()
        
        df['product_title'] = df['product_title'].fillna("")
        df['product_brand'] = df['product_brand'].fillna("")
        df['product_color'] = df['product_color'].fillna("")
        
        # Encode brand and color as integers
        brand_encoder = LabelEncoder()
        color_encoder = LabelEncoder()
        
        df['brand_encoded'] = brand_encoder.fit_transform(df['product_brand'])
        df['color_encoded'] = color_encoder.fit_transform(df['product_color'])
        
        G = nx.Graph()
        names, brands, colors = [],[],[]
        for _, row in df.iterrows():
            pid = row['node_id']
            pt = " ".join(str(row[field]) for field in ['product_title', 'product_description', 'product_bullet_point'] if pd.notnull(row[field]))
    
            names.append(pt)
            brands.append(row['brand_encoded'])
            colors.append(row['color_encoded'])
            
            G.add_node(pid, names = pt ,brand=row['brand_encoded'], color=row['color_encoded'])
        
        extra_brands = torch.tensor(brands, dtype=torch.float32).unsqueeze(1)  # shape: (N, 1)
        extra_colors = torch.tensor(colors, dtype=torch.float32).unsqueeze(1)     # shape: (N, 1)
        
        embeddings = sentence_transformer_model.encode(names)
        embeddings = torch.tensor(embeddings, dtype=torch.float32) # embeddings aka node_feats
    
        embeddings = torch.cat([embeddings, extra_brands, extra_colors], dim=1)
        
        adj = nx.adjacency_matrix(G).tocoo()
        edge_index = torch.from_numpy(np.vstack((adj.row, adj.col)).astype(np.int64))
        
        return G, embeddings, edge_index
    else:
        df = df_products.copy()
        G = nx.Graph()
        
        df['product_class'] = df['product_class'].fillna("")
        df['category hierarchy'] = df['category hierarchy'].fillna("")
        
    
        # Encode product class and hierachy as integers
        class_encoder = LabelEncoder()
        hierarchy_encoder = LabelEncoder()
        
        df['class_encoded'] = class_encoder.fit_transform(df['product_class'])
        df['category_encoded'] = hierarchy_encoder.fit_transform(df['category hierarchy'])
    
        names, p_class, p_at = [],[],[]
        
        for _, row in df.iterrows():
            pid = row['node_id']
            pt = " ".join(str(row[field]) for field in ['product_name', 'product_description', 'product_features'] if pd.notnull(row[field]))
            
            G.add_node(pid, names = pt, p_class = row['class_encoded'], p_hir =row['category_encoded'])
            
            names.append(pt)
            p_class.append(row['class_encoded'])
            p_at.append(row['category_encoded'])
            
              
        # Encode textual node features
        extra_class = torch.tensor(p_class, dtype=torch.float32).unsqueeze(1)  # shape: (N, 1)
        extra_cat = torch.tensor(p_at, dtype=torch.float32).unsqueeze(1)     # shape: (N, 1)
        
        # Transformer embeddings
        embeddings = sentence_transformer_model.encode_document(names)
        embeddings = torch.tensor(embeddings, dtype=torch.float32)  # shape: (N, emb_dim)
        
        # Now concatenate safely
        embeddings = torch.cat([embeddings, extra_class, extra_cat], dim=1)
    
        adj = nx.adjacency_matrix(G).tocoo()
        edge_index = torch.from_numpy(np.vstack((adj.row, adj.col)).astype(np.int64))
    
        return G, embeddings, edge_index
