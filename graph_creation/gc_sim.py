# Creates a graph depending on gnn_type with edge types, without, with edge attributes. 
# With _add_edges many edges per node outgoing per attribute
# Edges between products of shared attribute, with the highest similarity in the product title

import torch
from torch import nn
import random

from sklearn.preprocessing import LabelEncoder
import networkx as nx
import pandas as pd
import numpy as np
import Levenshtein


class GraphCreator:
    def __init__(self, df_products, gnn_type, sentence_transformer_model, add_edge=4):
        self._df_products = df_products
        self._gnn_type = gnn_type
        self._sentence_transformer_model = sentence_transformer_model

        self._add_edges = add_edge

    def custom_create_graph(self):
        if 'product_title' in self._df_products.columns:
            product_infos = ['product_title', 'product_description', 'product_bullet_point']
            attributes = ['product_brand', 'product_color']
        else:
            product_infos = ['product_name', 'product_description', 'product_features']
            attributes = ['product_class', 'category hierarchy']

        # Encode both attributes
        self._df_products[attributes[0]] = self._df_products[attributes[0]].fillna("")
        self._df_products[attributes[1]] = self._df_products[attributes[1]].fillna("")

        # Encode product class and hierachy as integers
        enc0 = LabelEncoder()
        enc1 = LabelEncoder()

        self._df_products['enc_att0'] = enc0.fit_transform(self._df_products[attributes[0]])
        self._df_products['enc_att1'] = enc1.fit_transform(self._df_products[attributes[1]])

        # ======
        # Create Nodes
        # =====
        names = []
        G = nx.Graph()
        for _, row in self._df_products.iterrows():
            pid = row['node_id']
            pt = " ".join(str(row[field]) for field in product_infos if pd.notnull(row[field]))
            names.append(pt)
            G.add_node(pid, names=pt, att0=row[attributes[0]], att1=row[attributes[1]])

        # ======
        # Create Edges
        # ======
        for vat0 in self._df_products['enc_att0'].unique():
            products_in_vat0 = self._df_products[self._df_products['enc_att0'] == vat0][
                ["node_id", product_infos[0]]].values.tolist()
            for i in range(len(products_in_vat0)):
                # find most similar products based on title
                product_similarities = [Levenshtein.distance(x[1], products_in_vat0[i][1]) for x in products_in_vat0 if
                                        x[0] != products_in_vat0[i][0]]
                # find indices of the _add_edges most similar products
                products_for_connection = np.argsort(product_similarities)[:self._add_edges]
                for j in products_for_connection:
                    if i == j:
                        continue
                    G.add_edge(products_in_vat0[i][0], products_in_vat0[j][0], type='same_attribute0')

        for vat1 in self._df_products['enc_att1'].unique():
            products_in_vat1 = self._df_products[self._df_products['enc_att1'] == vat1][
                ["node_id", product_infos[1]]].values.tolist()
            for i in range(len(products_in_vat1)):
                # title_i = str(products_in_vat1[i][1] or "")
                # find most similar products based on title
                product_similarities = [Levenshtein.distance(str(x[1] or ""), str(products_in_vat1[i][1] or "")) for x
                                        in products_in_vat1 if x[0] != products_in_vat1[i][0]]
                # find indices of the _add_edges most similar products
                products_for_connection = np.argsort(product_similarities)[:self._add_edges]
                for j in products_for_connection:
                    if i == j:
                        continue
                    G.add_edge(products_in_vat1[i][0], products_in_vat1[j][0], type='same_attribute1')

        embeddings = self._sentence_transformer_model.encode_document(names)
        embeddings = torch.tensor(embeddings, dtype=torch.float32)  # shape: (N, emb_dim)

        adj = nx.adjacency_matrix(G).tocoo()

        if self._gnn_type == 0:  # for no edge types, e.g., graphsage
            edge_index = torch.from_numpy(np.vstack((adj.row, adj.col)).astype(np.int64))
            return G, embeddings, edge_index
        elif self._gnn_type == 1:  # for edge_types, e.g., rgcn
            edges = list(G.edges(data=True))
            edge_index_list = []
            edge_type_list = []

            edge_type_to_id = {'enc_att0': 0, 'enc_att1': 1}

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
        elif self._gnn_type == 2:  # for edge_attributes, e.g., graph transformers
            edge_list = []
            edge_attr_list = []
            for u, v, data in G.edges(data=True):
                edge_list.append((u, v))
                edge_list.append((v, u))  # undirected → add both directions
                if data['type'] == 'same_attribute0':
                    edge_attr_list.append([0.0])
                    edge_attr_list.append([0.0])
                elif data['type'] == 'same_attribute1':
                    edge_attr_list.append([1.0])
                    edge_attr_list.append([1.0])

            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)

            return G, embeddings, edge_index, edge_attr