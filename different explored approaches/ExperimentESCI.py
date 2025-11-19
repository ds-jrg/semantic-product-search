# Attempt at recreating testing conditions from ESCI dataset
# Where only a small amount of products get ranked

# python
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
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from torch_geometric.utils import from_networkx
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from torch_geometric.nn import GCNConv, FastRGCNConv, RGCNConv, RGATConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import GraphSAGE
from sklearn.metrics import average_precision_score
from sentence_transformers import SentenceTransformer, losses, util
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import random_split
import re
from sklearn.metrics import average_precision_score, ndcg_score
import json
import argparse
import importlib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sentence_transformer_model = SentenceTransformer('all-MiniLM-L6-v2')


def main(model, dataset, edges, task_version, test_subset, loss_fct="cosine_mse", batch_size=32):
    global module, graph_creation, model_gnn, optimizer, loss_func
    global df_products, label_dict, label_dict_by_qid

    # load model module
    if os.path.exists(f'scripts/{model}.py'):
        module = importlib.import_module(str('scripts.' + model), package=None)
    required = ["custom_gnn"]
    for name in required:
        if not hasattr(module, name):
            raise RuntimeError(f"Model module {module} must define '{name}'")

    # load dataset
    if dataset == "esci":
        df_examples = pd.read_parquet('data/esci-data/shopping_queries_dataset_examples.parquet')
        df_products = pd.read_parquet('data/esci-data/shopping_queries_dataset_products.parquet')

        df_examples = df_examples[df_examples["product_locale"] == "us"]
        df_products = df_products[df_products["product_locale"] == "us"]

        df_examples = df_examples[df_examples["small_version"] == 1]

    # choose subset size
    match test_subset:
        case 0:
            desired_length = 10000;
            rkey = 42
        case 1:
            desired_length = 10000;
            rkey = 43
        case 2:
            desired_length = 10000;
            rkey = 69
        case 3:
            desired_length = 50000;
            rkey = 42
        case 4:
            desired_length = 50000;
            rkey = 43
        case 5:
            desired_length = 50000;
            rkey = 69
        case 6:
            desired_length = 100000;
            rkey = 33
        case 7:
            desired_length = 100000;
            rkey = 32
        case 8:
            desired_length = 100000;
            rkey = 31

    rng = np.random.default_rng(rkey)
    qids = df_examples['query_id'].unique()
    num_queries = int(desired_length / (len(df_examples) / len(qids)))
    qids_to_use = rng.choice(qids, size=num_queries, replace=False)

    df_examples = df_examples[df_examples["query_id"].isin(qids_to_use)]
    df_products = df_products[df_products["product_id"].isin(df_examples["product_id"])]

    df_examples = df_examples.sort_values(by='query_id')

    if dataset == "esci":
        label = 'esci_label'
        df_products = clean_prod_desc(df_products)
        df_products['product_bullet_point'] = clean_emojis_symbols(df_products['product_bullet_point'])
        df_products['product_title'] = clean_emojis_symbols(df_products['product_title'])

    # assign sequential node ids
    df_products = df_products.reset_index(drop=True).copy()
    df_products['node_id'] = range(len(df_products))
    product_id_to_node_id = dict(zip(df_products['product_id'], df_products['node_id']))
    node_id_to_product_id = dict(zip(df_products['node_id'], df_products['product_id']))

    # create label dict and per-qid index
    label_map = {"E": 1, "S": 0.2, "C": 0.01, "I": 0, "Exact": 1, "Partial": 0.2, "Irrelevant": 0}
    label_dict = create_label_dict(df_examples, label, label_map)

    # build qid -> list[(pid, label)] for fast lookup
    label_dict_by_qid = defaultdict(list)
    for (qid, pid), lbl in label_dict.items():
        label_dict_by_qid[qid].append((pid, lbl))

    # train/val/test splits
    total_len = len(df_examples)
    train_parts = int(0.7 * total_len)
    eval_parts = int(0.1 * total_len)
    test_parts = total_len - eval_parts - train_parts

    if task_version == 1 and dataset == "esci":
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

    # datasets and loaders
    train_dataset = CustomData(train_data, sentence_transformer_model)
    eval_dataset = CustomData(eval_data, sentence_transformer_model)
    test_dataset = CustomData(test_data, sentence_transformer_model)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)

    # model, optimizer, loss
    model_gnn = module.custom_gnn(device)
    optimizer = torch.optim.Adam(model_gnn.parameters(), lr=0.001)

    if os.path.exists(f"loss_fncs/{loss_fct}.py"):
        for_loss_func = importlib.import_module(str(f'loss_fncs.{loss_fct}'), package=None)
        loss_func = getattr(for_loss_func, "custom_loss_func")
    else:
        raise RuntimeError("Loss function not found")

    # load graph creator module
    if os.path.exists(f'graph_creation/{edges}.py'):
        for_graph_creation = importlib.import_module(f'graph_creation.{edges}', package=None)
        # many graph creator implementations accept (gnn_type, sentence_transformer_model) in __init__
        # others might accept different signatures — instantiation below assumes this common form.
        graph_creation = for_graph_creation.GraphCreator(gnn_type=model_gnn.get_type(),
                                                         sentence_transformer_model=sentence_transformer_model)
    else:
        raise RuntimeError("Graph creation module not found")

    result = perfrom_Training_and_Evaluation(train_dataloader, eval_dataloader, test_dataloader, model)
    print(model, result)
    with open(f'outputs/results.txt', 'a') as f:
        f.write(f"{model, dataset, edges, task_version, test_subset, batch_size}={result}")
    return result


class CustomData(Dataset):
    def __init__(self, dataset, sentence_transformer_model, batch_size=64):
        self.pairs = []
        query_ids = dataset['query_id'].tolist()
        queries = dataset['query'].tolist()
        if 'product_id' in dataset:
            product_ids = dataset['product_id'].tolist()
        else:
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


def create_label_dict(df, label, label_map):
    return {(row['query_id'], row['product_id']): label_map[row[label]] for _, row in df.iterrows()}


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
                           u"\U0001F600-\U0001F64F"
                           u"\U0001F300-\U0001F5FF"
                           u"\U0001F680-\U0001F6FF"
                           "]+", flags=re.UNICODE)


def clean_emojis_symbols(text):
    return text.astype(str).apply(lambda x: emoji_pattern.sub("", x))


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


def train(train_dataloader):
    model_gnn.train()
    total_loss = 0.0
    steps = 0
    for anchors, qids, _ in train_dataloader:
        anchors = anchors.to(device)
        # iterate per-sample (one small graph per query)
        for i in range(len(qids)):
            qid = int(qids[i].item())
            query_emb = anchors[i].unsqueeze(0).to(device)  # shape (1, dim)
            candidates = label_dict_by_qid.get(qid, [])
            if not candidates:
                continue
            candidate_pids = [pid for pid, _ in candidates]
            # build small product df (copy to avoid SettingWithCopyWarning downstream)
            df_prod_small = df_products[df_products['product_id'].isin(candidate_pids)].reset_index(drop=True).copy()
            if df_prod_small.empty:
                continue
            kwargs = graph_creation.custom_create_graph(df_prod_small)
            node_emb = model_gnn.get_node_emb(*kwargs).to(device)  # shape (N_local, dim)

            # map pids -> local indices in df_prod_small, keep order equal to candidate_pids
            pid_to_idx = {pid: idx for idx, pid in enumerate(df_prod_small['product_id'])}
            valid_pids = [p for p in candidate_pids if p in pid_to_idx]
            if not valid_pids:
                continue
            indices = [pid_to_idx[p] for p in valid_pids]
            node_emb_ordered = node_emb[indices]  # shape (M, dim)
            labels = torch.tensor([label_dict.get((qid, p), 0.0) for p in valid_pids], dtype=torch.float32).to(device)

            query_rep = query_emb.expand(node_emb_ordered.size(0), -1)  # shape (M, dim)
            loss = loss_func(query_rep, node_emb_ordered, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            steps += 1

    return total_loss / max(1, steps)


@torch.no_grad()
def test(dataloader):
    model_gnn.eval()
    ndcg_list = []
    for anchors, qids, _ in dataloader:
        anchors = anchors.to(device)
        for i in range(len(qids)):
            qid = int(qids[i].item())
            query_emb = anchors[i].unsqueeze(0).to(device)
            candidates = label_dict_by_qid.get(qid, [])
            if not candidates:
                continue

            candidate_pids = [pid for pid, _ in candidates]
            df_prod_small = df_products[df_products['product_id'].isin(candidate_pids)].reset_index(drop=True).copy()
            if df_prod_small.empty:
                continue
            kwargs = graph_creation.custom_create_graph(df_prod_small)
            node_emb = model_gnn.get_node_emb(*kwargs).to(device)
            # compute cosine similarities
            sims = util.cos_sim(query_emb, node_emb).squeeze(0).cpu().numpy()  # shape (N_local,)

            # order local indices by decreasing similarity
            sim_indx = np.argsort(sims)[::-1].tolist()
            relevance = [0] * len(sim_indx)
            for j, local_idx in enumerate(sim_indx):
                pid = df_prod_small['product_id'].iloc[local_idx]
                rel = label_dict.get((qid, pid), 0)
                if rel < 1:
                    continue
                relevance[j] = 1

            total_pos = sum(relevance)
            if total_pos == 0:
                continue

            ranks = np.flatnonzero(np.atleast_1d(relevance) == 1)

            def dcg():
                dcg_im = 0
                for l in ranks.tolist():
                    dcg_im += 1 / (np.log2(l + 1 + 1))
                return dcg_im

            def idcg():
                idcg_im = 0
                for n in range(total_pos):
                    idcg_im += 1 / (np.log2(n + 1 + 1))
                return idcg_im

            ndcg_list.append(dcg() / idcg())

    return float(np.mean(ndcg_list)) if len(ndcg_list) > 0 else 0.0


@torch.no_grad()
def evaluate(dataloader):
    model_gnn.eval()
    mrr, ndcg, p_at_k, r_precision, sci_ap = [], [], [], [], []

    for anchors, qids, _ in dataloader:
        anchors = anchors.to(device)
        for i in range(len(qids)):
            qid = int(qids[i].item())
            query_emb = anchors[i].unsqueeze(0).to(device)
            candidates = label_dict_by_qid.get(qid, [])
            if not candidates:
                continue
            candidate_pids = [pid for pid, _ in candidates]
            df_prod_small = df_products[df_products['product_id'].isin(candidate_pids)].reset_index(drop=True).copy()
            if df_prod_small.empty:
                continue
            kwargs = graph_creation.custom_create_graph(df_prod_small)
            node_emb = model_gnn.get_node_emb(*kwargs).to(device)
            sims = util.cos_sim(query_emb, node_emb).squeeze(0).cpu().numpy()

            sim_indx = np.argsort(sims)[::-1].tolist()
            relevance = [0] * len(sim_indx)
            for j, local_idx in enumerate(sim_indx):
                pid = df_prod_small['product_id'].iloc[local_idx]
                rel = label_dict.get((qid, pid), 0)
                if rel < 1: continue
                relevance[j] = 1

            total_pos = sum(relevance)
            if total_pos == 0:
                continue

            relevance = np.array(relevance)

            def p_for_k(x):
                return sum(relevance[:x + 1]) / (x + 1)

            p_at_k.append(p_for_k(10))
            r_precision.append(sum(relevance[:total_pos]) / total_pos)
            ranks = np.where(relevance == 1)[0]
            mrr.append(1.0 / (ranks[0] + 1) if len(ranks) > 0 else 0)

            def dcg():
                dcg_im = 0
                for l in ranks.tolist():
                    dcg_im += 1 / (np.log2(l + 1 + 1))
                return dcg_im

            def idcg():
                idcg_im = 0
                for n in range(total_pos):
                    idcg_im += 1 / (np.log2(n + 1 + 1))
                return idcg_im

            ndcg.append(dcg() / idcg())

            # scikit MAP
            # build y_true in local product order
            y_true = np.array([1 if label_dict.get((qid, pid), 0) >= 1 else 0 for pid in df_prod_small['product_id']])
            if y_true.sum() == 0:
                continue
            sci_ap.append(average_precision_score(y_true, sims))

    results = {
        "MRR": float(np.mean(mrr)) if mrr else 0.0,
        "nDCG": float(np.mean(ndcg)) if ndcg else 0.0,
        "R-Precision": float(np.mean(r_precision)) if r_precision else 0.0,
        "Precision@k": float(np.mean(p_at_k)) if p_at_k else 0.0,
        "Scikit-MAP": float(np.mean(sci_ap)) if sci_ap else 0.0
    }
    return json.dumps(results)


def perfrom_Training_and_Evaluation(train_dataloader, eval_dataloader, test_dataloader, model):
    best_ndcg = -float("inf")  # track best validation loss
    patience = 5  # 30/5                 # how many epochs to wait
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
