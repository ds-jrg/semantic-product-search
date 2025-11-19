# For easily testing different models with different data and subsets
# Calculating the final embedding for similarity computation from GNN and Textbased approach
# Prompt:
# python ExperimentHybrid.py gtpyg_gtconv --dataset esci --edges gc_dataset_2 --task_version 1 --test_subset 0  --batch_size 64
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
import importlib
from sentence_transformers import util

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sentence_transformer_model = SentenceTransformer('all-MiniLM-L6-v2')


def main(model, dataset, edges, task_version, test_subset, loss_fct="cosine_mse", batch_size=32):
    # test_gnn(module, dataset, edges, task_version, test_subset, loss_fct, batch_size)
    # dataset: # esci, wands
    # task version: 1= task 1; 2=task 2
    global alpha_tensor
    alpha = 0.9
    alpha_tensor = torch.tensor(alpha, device=device)

    # =========
    # Load in the model and it's charcteristics
    # =========
    global module
    # Import all characteristics of the model -> save in module
    if os.path.exists(f'scripts/{model}.py'):
        module = importlib.import_module(str('scripts.' + model), package=None)
    else:
        module = importlib.import_module(str('diff_scripts.' + model), package=None)
    # Check if module has everything imported
    required = ["custom_gnn"]
    for name in required:
        if not hasattr(module, name):
            raise RuntimeError(f"Model module {module} must define '{name}'")

    # ==========
    # Select subset (for amazon or wands)
    # ==========
    global node_id_to_product_id
    global product_id_to_node_id
    global label_dict
    global df_products
    if dataset == "esci":  # if amazon
        df_examples = pd.read_parquet('data/esci-data/shopping_queries_dataset_examples.parquet')
        df_products = pd.read_parquet('data/esci-data/shopping_queries_dataset_products.parquet')

        df_examples = df_examples[df_examples["product_locale"] == "us"]
        df_products = df_products[df_products["product_locale"] == "us"]

        # Reduce to smaller subset
        df_examples = df_examples[df_examples["small_version"] == 1]

    else:
        df_products = pd.read_csv("data/wands-data/product.csv", sep='\t')
        query_df = pd.read_csv("data/wands-data/query.csv", sep='\t')
        label_df = pd.read_csv("data/wands-data/label.csv", sep='\t')
        df_examples = label_df

    # we want to split data early, reduce unnecessary data cleaning,...
    match test_subset:
        case 0:
            desired_length = 10000
            rkey = 42
        case 1:
            desired_length = 10000
            rkey = 43
        case 2:
            desired_length = 10000
            rkey = 69
        case 3:
            desired_length = 50000
            rkey = 42
        case 4:
            desired_length = 50000
            rkey = 43
        case 5:
            desired_length = 50000
            rkey = 69
        case 6:
            desired_length = 100000
            rkey = 33
        case 7:
            desired_length = 100000
            rkey = 32
        case 8:
            desired_length = 100000
            rkey = 31

    rng = np.random.default_rng(rkey)

    qids = df_examples['query_id'].unique()  # all querys
    num_queries = int(desired_length / (len(df_examples) / len(qids)))  # desired length/ avg depth
    # for calculating ca. desired length many queries
    qids_to_use = rng.choice(qids, size=num_queries, replace=False)

    # include all querys from those qids and only the included products
    df_examples = df_examples[df_examples["query_id"].isin(qids_to_use)]
    df_products = df_products[df_products["product_id"].isin(df_examples["product_id"])]

    df_examples = df_examples.sort_values(by='query_id')

    if dataset == "esci":
        label = 'esci_label'
        df_products = clean_prod_desc(df_products)
        # df_products = clean_color(df_products)
        df_products['product_bullet_point'] = clean_emojis_symbols(df_products['product_bullet_point'])
        df_products['product_title'] = clean_emojis_symbols(df_products['product_title'])
    else:
        label = 'label'
        df_examples = pd.merge(df_examples, query_df, how='left', on='query_id')
        df_examples = df_examples.sort_values(by='query_id')

    # Assign a unique sequential node ID to each product (for easier accessing) from 0 to len(df_products)
    df_products = df_products.reset_index(drop=True).copy()
    df_products['node_id'] = range(len(df_products))
    # Map between original product indices and node_ids
    product_id_to_node_id = dict(zip(df_products['product_id'], df_products['node_id']))
    node_id_to_product_id = dict(zip(df_products['node_id'], df_products['product_id']))

    # Create a lookup table for product relevancy
    label_map = {"E": 1, "S": 0.2, "C": 0.01, "I": 0, "Exact": 1, "Partial": 0.2, "Irrelevant": 0}
    # label_map = {"E": 0, "S": 0.9, "C":0.99, "I": 1, "Exact":0, "Partial":0.8, "Irrelevant":0} this doesn't work as well
    label_dict = create_label_dict(df_examples, label, label_map)

    total_len = len(df_examples)
    train_parts = int(0.7 * total_len)
    eval_parts = int(0.1 * total_len)
    test_parts = total_len - eval_parts - train_parts

    if task_version == 1 and dataset == "esci":  # to predict unseen questions
        train_data = df_examples[df_examples['split'] == 'train']
        train_data, eval_data = train_data[:int(len(train_data) * .8)], train_data[int(len(train_data) * 0.8 + 1):]
        test_data = df_examples[df_examples['split'] == 'test']
    elif task_version == 1:
        train_data = df_examples.iloc[:train_parts, :]
        eval_data = df_examples.iloc[train_parts:(train_parts + eval_parts), :]
        test_data = df_examples.iloc[(train_parts + eval_parts):, :]
    else:
        splits = random_split(
            df_examples,
            [train_parts, eval_parts, test_parts],
            generator=torch.Generator().manual_seed(42)
        )
        train_idx, val_idx, test_idx = [list(s.indices) for s in splits]

        train_data = df_examples.iloc[train_idx]
        eval_data = df_examples.iloc[val_idx]
        test_data = df_examples.iloc[test_idx]

    test_data = test_data[['query', 'query_id']].drop_duplicates()

    # Use the custom data/collate
    train_dataset = CustomData(train_data, sentence_transformer_model)
    eval_dataset = CustomData(eval_data, sentence_transformer_model)
    test_dataset = CustomData(test_data, sentence_transformer_model)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)

    # ========
    # Load in the model and other requirements
    # ========
    global model_gnn
    global optimizer
    global loss_func
    model_gnn = module.custom_gnn(device)
    optimizer = torch.optim.Adam(model_gnn.parameters(), lr=0.001)

    if os.path.exists(f"loss_fncs/{loss_fct}.py"):
        for_loss_func = importlib.import_module(str(f'loss_fncs.{loss_fct}'), package=None)
        loss_func = getattr(for_loss_func, "custom_loss_func")

    # =======
    # Load or Create the custom graph
    # =======
    global kwargs
    gnn_type = model_gnn.get_type()
    graph_path = f"temp_storage/saved_graphs/PG-{edges}-{dataset}-{test_subset}-{gnn_type}"

    # if os.path.exists(os.path.join(graph_path, "graph.pkl")):
    #  print("Loading graph from disk...")
    #   kwargs = load_graph_data(graph_path)
    # else:
    if True:
        print("Creating graph from scratch...")
        if os.path.exists(f'graph_creation/{edges}.py'):
            for_graph_creation = importlib.import_module(f'graph_creation.{edges}', package=None)
            for_graph_creation = for_graph_creation.GraphCreator(df_products, gnn_type, sentence_transformer_model)
            kwargs = for_graph_creation.custom_create_graph()
            # kwargs = G, embeddings, edge_index, (edge_type/edge_attr)
            save_graph_data(kwargs, graph_path)

    G = kwargs[0]
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")

    # ========
    # Train and evaluate the model
    # ========
    result = perfrom_Training_and_Evaluation(train_dataloader, eval_dataloader, test_dataloader, model)

    print(model, result)
    with open(f'outputs/results.txt', 'a') as f:
        f.write(f"{model, dataset, edges, task_version, test_subset, batch_size}={result}")
    return result


# ========
# All helper functions
# ========
class CustomData(Dataset):
    def __init__(self, dataset, sentence_transformer_model, batch_size=64):
        self.pairs = []

        # Extract all queries as a list
        query_ids = dataset['query_id'].tolist()  # query, qid, product_id
        queries = dataset['query'].tolist()
        if 'product_id' in dataset:
            product_ids = dataset['product_id'].tolist()
        else:  # since in eval, we wont use the corresponding pids, we can ignore them
            product_ids = [0] * len(queries)

        all_embeddings = sentence_transformer_model.encode(
            queries, batch_size=batch_size, convert_to_tensor=True
        ).to(torch.float32)

        for emb, qid, pid in zip(all_embeddings, query_ids, product_ids):
            self.pairs.append((emb, qid, pid))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        query, qid, product_id = self.pairs[idx]
        return query, torch.tensor(qid), product_id


def custom_collate(batch):
    anchors = torch.stack([item[0] for item in batch])
    query_ids = torch.stack([item[1] for item in batch])
    pids = [item[2] for item in batch]
    return anchors, query_ids, pids


# Creating a lookup table for query id, product_id to relevance
def create_label_dict(df, label, label_map):
    return {(row['query_id'], row['product_id']): label_map[row[label]] for _, row in df.iterrows()}


# For cleaning product details in amazon
def clean_color(df_products):
    colors = ["WHITE", "YELLOW", "BLUE", "RED", "GREEN", "BLACK", "BROWN", "AZURE", "IVORY", "TEAL", "SILVER", "PURPLE",
              "NAVY BLUE", "PEA GREEN", "GRAY", "ORANGE", "MAROON", "CHARCOAL", "AQUAMARINE", "CORAL", "FUCHSIA",
              "WHEAT", "LIME", "CRIMSON", "KHAKI", "HOT PINK", "MAGENTA", "OLDEN", "PLUM", "OLIVE", "CYAN"]
    pattern = '|'.join(colors)

    def process_colors(text):
        if text is None: return '[]'
        matches = re.findall(pattern, text.upper())
        return str(matches) if matches else '[]'

    df_products['product_color'] = df_products['product_color'].apply(process_colors)
    return df_products


def clean_prod_desc(df_products):
    def process_info(text):
        if text is None: return text
        r = 0
        text_c = ""
        for i in text:
            if r == 0 and i != "<":
                text_c += i
            elif i == ">":
                r = 0
            else:
                r = 1
        return text_c

    df_products['product_description'] = df_products['product_description'].apply(process_info)
    return df_products


emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           "]+", flags=re.UNICODE)


def clean_emojis_symbols(text):
    return text.astype(str).apply(lambda x: emoji_pattern.sub("", x))


# for loading or saving the graph:
def load_graph_data(path="graph_data"):
    with open(os.path.join(path, "graph.pkl"), "rb") as f:
        G = pickle.load(f)

    embeddings = torch.load(os.path.join(path, "embeddings.pt"))
    edge_index = torch.load(os.path.join(path, "edge_index.pt"))
    if os.path.exists(os.path.join(path, "edge_attr.pt")):
        edge_attr = torch.load(os.path.join(path, "edge_attr.pt"))
        return G, embeddings, edge_index, edge_attr
    return G, embeddings, edge_index


def save_graph_data(kwargs, path="graph_data"):
    os.makedirs(path, exist_ok=True)
    if len(kwargs) == 3:
        G, embeddings, edge_index = kwargs
    else:
        G, embeddings, edge_index, edge_attr = kwargs
        torch.save(edge_attr, os.path.join(path, "edge_attr.pt"))

    with open(os.path.join(path, "graph.pkl"), "wb") as f:
        pickle.dump(G, f)
    torch.save(embeddings, os.path.join(path, "embeddings.pt"))
    torch.save(edge_index, os.path.join(path, "edge_index.pt"))


# for training and testing:
def train(train_dataloader):
    model_gnn.train()
    total_loss = 0
    for querys, qids, product_ids in train_dataloader:
        anchor_projs = querys.to(device)
        qids = qids.to(device)
        old_emb = kwargs[1].to(device)
        node_emb = model_gnn.get_node_emb(*kwargs).to(device)

        # Ensure alpha is on correct device
        final_score = old_emb * alpha_tensor + node_emb * (1 - alpha_tensor)

        # Transform product ids to node ids to access, nodes in graph
        node_ids = [product_id_to_node_id.get(pid) for pid in product_ids]
        product_emb = final_score[node_ids]
        labels = torch.Tensor([label_dict[qid.item(), pid] for qid, pid in zip(qids, product_ids)]).to(device)
        loss = loss_func(product_emb, anchor_projs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_dataloader)


@torch.no_grad()
def test(dataloader):
    model_gnn.eval()
    ndcg = []

    with torch.no_grad():
        old_emb = kwargs[1].to(device)
        node_emb = model_gnn.get_node_emb(*kwargs).to(device)

        # Ensure alpha is on correct device
        final_score = old_emb * alpha_tensor + node_emb * (1 - alpha_tensor)

        for anchors, qids, _ in dataloader:
            # We only compare the anchors to all the product nodes (needed for rdf)
            sims = sentence_transformer_model.similarity(anchors, final_score[:len(df_products)])
            for i in range(len(anchors)):
                sim_scores = sims[i].cpu().numpy()
                qid = qids[i].item()
                # Represents the node-indexes with decreasing similarity
                sim_indx = np.argsort(sim_scores)[::-1].tolist()
                # Create relevance vector
                relevance = [0] * len(sim_indx)
                for j in range(len(sim_indx)):
                    pid = node_id_to_product_id[sim_indx[j]]
                    rel = label_dict.get((qid, pid), 0)
                    if rel < 1: continue
                    relevance[j] = 1

                # Total of positive matches
                total_pos = sum(relevance)

                if total_pos == 0:
                    continue

                ranks = np.flatnonzero(np.atleast_1d(relevance) == 1)

                # NDCG: All scores with rel == 0 can be ignored
                def dcg():
                    dcg_im = 0
                    for l in ranks.tolist():
                        dcg_im += 1 / (np.log2(l + 1 + 1))  # bcs we start at 0
                    return dcg_im

                def idcg():
                    idcg_im = 0
                    for n in range(total_pos):
                        idcg_im += 1 / (np.log2(n + 1 + 1))
                    return idcg_im

                ndcg.append(dcg() / idcg())

        return np.mean(ndcg)


@torch.no_grad()
def evaluate(dataloader):
    model_gnn.eval()
    mrr, ndcg, p_at_k, r_precision, sci_ap = [], [], [], [], []

    with torch.no_grad():

        old_emb = kwargs[1].to(device)
        node_emb = model_gnn.get_node_emb(*kwargs).to(device)

        # Ensure alpha is on correct device
        final_score = old_emb * alpha_tensor + node_emb * (1 - alpha_tensor)

        for anchors, qids, _ in dataloader:
            sims = sentence_transformer_model.similarity(anchors, final_score[:len(df_products)])
            for i in range(len(anchors)):
                sim_scores = sims[i].cpu().numpy()
                qid = qids[i].item()
                # Represents the node-indexes with decreasing similarity
                sim_indx = np.argsort(sim_scores)[::-1].tolist()
                # Create relevance vector
                relevance = [0] * len(sim_indx)
                for j in range(len(sim_indx)):
                    pid = node_id_to_product_id[sim_indx[j]]
                    rel = label_dict.get((qid, pid), 0)
                    if rel < 1: continue
                    relevance[j] = 1

                # Total of positive matches
                total_pos = sum(relevance)

                if total_pos == 0:
                    continue

                relevance = np.array(relevance)

                # Precision at k= relevant geschnit retrieved / retrieved
                def p_for_k(x):
                    return sum(relevance[:x + 1]) / (x + 1)

                # Precision at top k (set to 10)
                p_at_k.append(p_for_k(10))

                # R-Precision: relevant, among sum relevant
                r_precision.append(sum(relevance[:total_pos]) / total_pos)

                # MRR
                ranks = np.where(relevance == 1)[0]
                if len(ranks) > 0:
                    mrr.append(1.0 / (ranks[0] + 1))
                else:
                    mrr.append(0)

                # NDCG: All scores with rel == 0 can be ignored
                def dcg():
                    dcg_im = 0
                    for l in ranks.tolist():
                        dcg_im += 1 / (np.log2(l + 1 + 1))  # bcs we start at 0
                    return dcg_im

                def idcg():
                    idcg_im = 0
                    for n in range(total_pos):
                        idcg_im += 1 / (np.log2(n + 1 + 1))
                    return idcg_im

                ndcg.append(dcg() / idcg())

                # Calculate with Scikit Learn
                def maper(x):
                    if x == '1':
                        return 1
                    else:
                        return 0

                y_true = np.array([maper(str(label_dict.get((qid, c), 0))) for c in df_products['product_id']])
                sci_ap.append(average_precision_score(y_true, sim_scores))
        results = {
            "MRR": float(np.mean(mrr)),
            "nDCG": float(np.mean(ndcg)),
            "R-Precision": float(np.mean(r_precision)),
            "Precision@k": float(np.mean(p_at_k)),
            "Scikit-MAP": float(np.mean(sci_ap))
        }
        return json.dumps(results)


def perfrom_Training_and_Evaluation(train_dataloader, eval_dataloader, test_dataloader, model):
    best_ndcg = -float("inf")  # track best validation loss
    patience = 5  # how many epochs to wait
    counter = 0  # how many epochs since last improvement
    MODEL_SAVE_PATH = f"temp_storage/saved_gnns/{model}-parameters"

    for epoch in range(1, 100):
        loss_eval = test(eval_dataloader)

        # check if  eval_loss improved, save params
        if loss_eval > best_ndcg:
            best_ndcg = loss_eval
            counter = 0  # reset patience counter
            torch.save(obj=model_gnn.state_dict(), f=MODEL_SAVE_PATH)
        else:
            counter += 1
        loss = train(train_dataloader)
        print('Evaluation:', loss_eval, 'Training:', loss)
        # stop if no improvement in last 5 epochs
        if counter >= patience:
            break

    # load in params back
    loaded_model_gnn = module.custom_gnn(device)
    loaded_model_gnn.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
    return evaluate(test_dataloader)


if __name__ == "__main__":
    # ========
    # Set arguments
    # ========
    # main(model, dataset, edges, task_version, test_subset, loss_fct="cosine_mse", batch_size=32):
    # task version: 1= task 1; 2=task 2
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, default="graphsage", help='Choose the GNN to use')
    parser.add_argument("--dataset", type=str, default="wands", help='Choose the dataset to use: esci or wands')
    parser.add_argument("--edges", type=str, default="old_2", help="Choose the edge creation for the Product Graph")
    parser.add_argument("--task_version", type=int, default=2,
                        help='Choose the version of testing: 1 = to predict unseen questions; 2 = rank relavants to known questions .')
    parser.add_argument("--test_subset", type=int, default=0, help="Choose the subset, and it's size. {0,...,8}")
    parser.add_argument("--loss_fct", type=str, default="cosine_mse", help="Choose the Loss Function for the approach")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    args = parser.parse_args()

    main(args.model, args.dataset, args.edges, args.task_version, args.test_subset, args.loss_fct, args.batch_size)