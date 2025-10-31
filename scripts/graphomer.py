
import torch
from torch import nn
from torch_geometric.nn import TransformerConv

from sklearn.preprocessing import LabelEncoder
import networkx as nx
import pandas as pd

from scripts.graphsage import custom_data, custom_collate, custom_loss_func


class custom_gnn(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.gnn = GraphormerLike(in_dim=386, 
                                  hidden_dim=512, 
                                  out_dim=384,
                                  edge_emb_dim=32, 
                                  max_sp_dist=4,
                                  num_layers=4, 
                                  heads=8
                                 ).to(device)

    def forward(self, embeddings, edge_index, edge_attr_ids):
        return self.gnn(embeddings.to(self.device), edge_index.to(self.device), edge_attr_ids.to(self.device))

    def get_node_emb(self, *kwargs):
        return self.forward(kwargs[1], kwargs[2], kwargs[3])



class GraphormerLike(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, edge_emb_dim, max_sp_dist, num_layers, heads, dropout=0.1):
        super().__init__()
        self.in_lin = nn.Linear(in_dim, hidden_dim)
        # edge distance embedding (indices 0..max_sp_dist-1)
        self.edge_emb = nn.Embedding(max_sp_dist, edge_emb_dim)
        # project edge_emb to edge_dim that TransformerConv expects (we will set edge_dim=edge_emb_dim)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            # out_per_head chosen to keep hidden_dim = out_per_head * heads when concat=True
            out_per_head = hidden_dim // heads
            conv = TransformerConv(hidden_dim, out_per_head, heads=heads, concat=True,
                                   beta=True, edge_dim=edge_emb_dim, dropout=dropout)
            self.layers.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.out_lin = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor, edge_attr_ids: torch.LongTensor):
        """
        x: [N, in_dim]
        edge_index: [2, E]
        edge_attr_ids: [E]  integer indices into edge embedding table
        returns: [N, out_dim] per-node embeddings
        """
        h = self.in_lin(x)  # [N, hidden_dim]
        # instantiate edge features
        e = self.edge_emb(edge_attr_ids.to(x.device))  # [E, edge_emb_dim]
        for conv, ln in zip(self.layers, self.norms):
            h2 = conv(h, edge_index.to(x.device), edge_attr=e)     # [N, hidden_dim]
            h = ln(h + h2)
        out = self.out_lin(h)
        return out
        

def custom_create_graph(df_products, sentence_model):
    """
    - Builds networkx graph G with node ids = df_products['node_id'] (int)
    - x: torch.tensor node features shape (N, feat_dim) with ST embeddings (+ optional brand/color numeric features)
    - product_id_to_node_id mapping dict
    """
    if 'product_title' in df_products.columns:
        df = df_products.copy().fillna("")
        # label encoders for brand/color (optional)
        brand_enc = LabelEncoder()
        color_enc = LabelEncoder()
        df['brand_encoded'] = brand_enc.fit_transform(df['product_brand'].astype(str))
        df['color_encoded'] = color_enc.fit_transform(df['product_color'].astype(str))
    
        G = nx.Graph()
        names = []
        brands = []
        colors = []
    
        for _, row in df.iterrows():
            nid = int(row['node_id'])
            text = " ".join(str(row[field]) for field in ['product_title', 'product_description', 'product_bullet_point']
                            if pd.notnull(row.get(field, "")))
            names.append(text)
            brands.append(int(row['brand_encoded']))
            colors.append(int(row['color_encoded']))
            G.add_node(nid, title=text, brand=int(row['brand_encoded']), color=int(row['color_encoded']))
    
        # add simple edges by brand/color (keeps graph sparse and meaningful)
        for brand in df['brand_encoded'].unique():
            pids = df[df['brand_encoded'] == brand]['node_id'].tolist()
            for i in range(len(pids)):
                for j in range(i + 1, min(i + 3, len(pids))):
                    G.add_edge(int(pids[i]), int(pids[j]), relation="brand")
        for color in df['color_encoded'].unique():
            pids = df[df['color_encoded'] == color]['node_id'].tolist()
            for i in range(len(pids)):
                for j in range(i + 1, min(i + 3, len(pids))):
                    if not G.has_edge(int(pids[i]), int(pids[j])):
                        G.add_edge(int(pids[i]), int(pids[j]), relation="color")
    
        # node features: sentence embeddings
        emb_np = sentence_model.encode(names, convert_to_numpy=True)  # (N, 384)
        emb_t = torch.tensor(emb_np, dtype=torch.float32)
    
        b_t = torch.tensor(brands, dtype=torch.float32).unsqueeze(1)  # (N,1)
        c_t = torch.tensor(colors, dtype=torch.float32).unsqueeze(1)
        x = torch.cat([emb_t, b_t, c_t], dim=1)  # (N, 386)

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
            
        
            # 3. Add edges (simple example: same brand → connect)
            for classes in df['class_encoded'].unique():
                products_in_class = df[df['class_encoded'] == classes]['node_id'].tolist()
                for i in range(len(products_in_class)):
                    for j in range(i + 1, min(i + 3, len(products_in_class))):  # only 5 connections per product
                        G.add_edge(products_in_class[i], products_in_class[j], type='same_class')
        
        
            for categorys in df['category_encoded'].unique():
                products_in_cat = df[df['category_encoded'] == categorys]['node_id'].tolist()
                for i in range(len(products_in_cat)):
                    for j in range(i + 1, min(i + 3, len(products_in_cat))):  # only 5 connections per product
                        G.add_edge(products_in_cat[i], products_in_cat[j], type='same_category')
        
        
        # Encode textual node features
        #names = [G.nodes[node]['names'] for node in G.nodes()]
        extra_class = torch.tensor(p_class, dtype=torch.float32).unsqueeze(1)  # shape: (N, 1)
        extra_cat = torch.tensor(p_at, dtype=torch.float32).unsqueeze(1)     # shape: (N, 1)
        
        # Transformer embeddings
        embeddings = sentence_model.encode_document(names)
        embeddings = torch.tensor(embeddings, dtype=torch.float32)  # shape: (N, emb_dim)
        
        # Now concatenate safely
        x = torch.cat([embeddings, extra_class, extra_cat], dim=1)
    
    # got G, x
    # now need edge_index, edge_attr_ids
    edge_index, edge_attr_ids = compute_shortest_path_edge_attrs(G, False, 4)
    
    return G, x, edge_index, edge_attr_ids

def compute_shortest_path_edge_attrs(G, complete_graph, max_dist):
    """
    Build edge_index (torch.LongTensor [2, E]) and edge_attr indices (torch.LongTensor [E])
    Edge attr is the shortest-path distance clipped to [1, max_dist]; we encode as integer ids (1..max_dist -> 0..max_dist-1)
    If complete_graph True: build complete directed graph (i->j for all i!=j) and use sp distance between nodes (heavy)
    Returns:
      edge_index: LongTensor [2, E]
      edge_dist_idx: LongTensor [E]    # integer index 0..max_dist (where max_dist corresponds to ">=max_dist")
    """
    nodes = sorted(G.nodes())
    idx_map = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)
    sp_all = dict(nx.all_pairs_shortest_path_length(G, cutoff=max_dist))  # dict: node -> {other: dist}
    edge_list = []
    edge_dists = []

    if complete_graph:
        # full directed graph (i->j)
        for u in nodes:
            row = sp_all.get(u, {})
            for v in nodes:
                if u == v: 
                    continue
                # distance if reachable else treat as max_dist
                d = row.get(v, max_dist)
                d = min(d, max_dist)
                edge_list.append((idx_map[u], idx_map[v]))
                # clamp to [1, max_dist], represent as 0-indexed id
                d_idx = max(1, d)  # ensure >=1
                edge_dists.append(d_idx)
    else:
        # only existing edges (undirected -> we add both directions)
        for (u, v, data) in G.edges(data=True):
            # for existing edges, distance should normally be 1, but if you prefer use sp_all distances:
            d_uv = sp_all.get(u, {}).get(v, 1)
            d_vu = sp_all.get(v, {}).get(u, 1)
            d_uv = min(max(1, d_uv), max_dist)
            d_vu = min(max(1, d_vu), max_dist)
            edge_list.append((idx_map[u], idx_map[v]))
            edge_dists.append(d_uv)
            edge_list.append((idx_map[v], idx_map[u]))
            edge_dists.append(d_vu)

    if len(edge_list) == 0:
        raise ValueError("Graph has no edges -> cannot build edge_index. Add edges or set COMPLETE_GRAPH=True.")

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()  # [2, E]
    # convert distances to 0-indexed embedding ids between 0 and max_dist (where id max_dist means '>=max_dist')
    edge_attr_ids = torch.tensor([min(d, max_dist) for d in edge_dists], dtype=torch.long)  # values 1..max_dist
    # shift to 0..(max_dist) by subtracting 1 so we have embedding dim = max_dist
    edge_attr_ids = edge_attr_ids - 1  # now 0..(max_dist-1)
    return edge_index, edge_attr_ids
