# For easily testing different models with different data and subsets
# Prompt: 
# python Experiment.py graphsage --dataset esci --edges gc_dataset_2 --task_version 1 --test_subset 0  --batch_size 32
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


def main(model, dataset, test_subset, batch_size=32):
    #test_gnn(module, dataset, edges, task_version, test_subset, loss_fct, batch_size)
    # dataset: # esci, wands
    
    #=========
    # Load in the model and it's charcteristics
    #=========
    global module
    # Import all characteristics of the model -> save in module
    if os.path.exists(f'baselines/{model}.py'):
        module = importlib.import_module(str('baselines.'+model), package=None)
    # Check if module has everything imported

    # TODO LATER

    required = ["custom_gnn"]
    for name in required:
        if not hasattr(module, name):
            raise RuntimeError(f"Model module {module} must define '{name}'")


    #==========
    # Select subset (for amazon or wands)
    #==========
    global node_id_to_product_id
    global product_id_to_node_id
    global label_dict
    global df_products
    if dataset == "esci": # if amazon
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
        
    qids = df_examples['query_id'].unique() # all querys
    num_queries = int(desired_length/(len(df_examples)/ len(qids))) # desired length/ avg depth
    # for calculating ca. desired length many queries
    qids_to_use = rng.choice(qids, size=num_queries, replace=False)
    
    # include all querys from those qids and only the included products
    df_examples = df_examples[df_examples["query_id"].isin(qids_to_use)]
    df_products = df_products[df_products["product_id"].isin(df_examples["product_id"])]

    # Sorting by query-id, so that train, val, test splits are useful
    df_examples = df_examples.sort_values(by='query_id')

    if dataset == "esci":
        label = 'esci_label'
        df_products = clean_prod_desc(df_products)
        df_products['product_bullet_point'] = clean_emojis_symbols(df_products['product_bullet_point'])
        df_products['product_title'] = clean_emojis_symbols(df_products['product_title'])
        product_infos = ['product_title', 'product_description', 'product_bullet_point']
        product_txt = 'product_title'
    else:
        label = 'label' 
        df_examples = pd.merge(df_examples, query_df, how='left', on='query_id')
        df_examples = df_examples.sort_values(by='query_id')
        product_infos = ['product_name', 'product_description', 'product_features']
        product_txt = 'product_name'

    # Sorting by query-id, so that train, val, test splits are useful
    df_examples = df_examples.sort_values(by='query_id')

    # Assign a unique sequential node ID to each product (for easier accessing) from 0 to len(df_products)
    df_products = df_products.reset_index(drop=True).copy()
    df_products['node_id'] = range(len(df_products))
    # Map between original product indices and node_ids
    product_id_to_node_id = dict(zip(df_products['product_id'], df_products['node_id']))
    node_id_to_product_id = dict(zip(df_products['node_id'], df_products['product_id']))

    # Create a lookup table for product relevancy
    label_map = {"E": 1, "S": 0.2, "C":0.01, "I": 0, "Exact":1, "Partial":0.2, "Irrelevant":0}
    label_dict = create_label_dict(df_examples, label, label_map)

    all_product_info = []
    for _, row in df_products.iterrows():
        all_product_info.append(" ".join(str(row[field]) for field in product_infos if pd.notnull(row[field])))


    total_len = len(df_examples)
    train_parts = int(0.7*total_len)
    val_parts = int(0.1*total_len)
    test_parts = total_len - val_parts - train_parts
    
    splits = random_split(
        df_examples,
        [train_parts, val_parts, test_parts],
        generator=torch.Generator().manual_seed(42)
    )
    train_idx, val_idx, test_idx = [list(s.indices) for s in splits]

    test_data  = df_examples.iloc[test_idx]
    
    test_data = test_data[['query','query_id']].drop_duplicates()


    if module. has "train,...": # TODO
    # Use the custom data/collate
        train_data = df_examples.iloc[train_idx]
        val_data = df_examples.iloc[val_idx]


    dataset = QueryDataset(test_data)
    test_dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    # TODO pass the models the all_product_info
    product_embeddings = module.create_necessary_things(all_product_info)

    custom_eval = module.Eval(test_dataloader, product_embeddings)
    result = custom_eval.evaluate()

    #========
    # Train and evaluate the model
    #========

    
    print(model,result)
    with open(f'outputs/results.txt', 'a') as f:
        f.write(f"BL:{model,dataset,test_subset,batch_size}={result}\n\n")
    return result
    


#========
# All helper functions
#========
class Evaluator:
    def __init__(self, dataloader, product_emb):
        self._dataloader = dataloader
        self._product_embeddings = product_emb  # doesnt need to be emb, can also be just strings

        """ 
    def forwards(self, query):
        # needs to get implemented by every one calling eval
        """

    def evaluate(self, skip=False):
        self.skip = skip
        # called self.forward
        if self.skip:
            mrr, ndcg, p_at_k, r_precision, sci_ap = [], [], [], [], 0
        else:
            mrr, ndcg, p_at_k, r_precision, sci_ap = [], [], [], [], []

        with torch.no_grad():
            for queries, qids in self._dataloader:
                for i in range(len(queries)):
                    # sim_scores = bm25.get_scores(queries[i])
                    sim_scores = self.forwards(queries[i], self._product_embeddings)

                    # Represents the node-indexes with decreasing similarity
                    sim_indx = np.argsort(sim_scores)[::-1].tolist()

                    # Create relevance vector
                    relevance = [0] * len(sim_indx)
                    for j in range(len(sim_indx)):
                        pid = node_id_to_product_id[sim_indx[j]]
                        if type(qids[i]) != int:
                            rel = label_dict.get((qids[i].item(), pid), 0)
                        else:
                            rel = label_dict.get((qids[i], pid), 0)
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

                    # Precision at top k
                    p_at_k.append(p_for_k(10))

                    # R-Precision: relevant, among sum relevant
                    r_precision.append(sum(relevance[:total_pos]) / total_pos)

                    # MRR
                    ranks = np.where(relevance == 1)[0]
                    if len(ranks) > 0:
                        mrr.append(1.0 / (ranks[0] + 1))
                    else:
                        mrr.append(0)

                    # NDCG
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

                    # MAP with Scikit-learn ony if skip == False
                    if not self.skip:
                        def maper(x):
                            if x == '1':
                                return 1
                            else:
                                return 0

                        y_true = np.array(
                            [maper(str(label_dict.get((qids[i], c), 0))) for c in df_products['product_id']])
                        sci_ap.append(average_precision_score(y_true, sim_scores))

            return {
                "MRR": float(np.mean(mrr)),
                "nDCG": float(np.mean(ndcg)),
                "R-Precision": float(np.mean(r_precision)),
                "Precision@k": float(np.mean(p_at_k)),
                "Scikit-MAP": float(np.mean(sci_ap))
            }


class QueryDataset(Dataset):
    def __init__(self, df, batch_size=64):
        self.pairs = []
        query_ids = df['query_id'].tolist()  # query, qid, product_id
        queries = df['query'].tolist()

        all_embeddings = [query.split(" ") for query in queries]

        for emb, qid in zip(all_embeddings, query_ids):
            self.pairs.append((emb, qid))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        emb, qid = self.pairs[idx]
        return emb, qid

def collate_fn(batch):
    emb = [item[0] for item in batch]
    query_ids = [item[1] for item in batch]
    return emb, query_ids

# Creating a lookup table for query id, product_id to relevance
def create_label_dict(df, label, label_map):
    return {(row['query_id'], row['product_id']): label_map[row[label]] for _, row in df.iterrows()}


def clean_prod_desc(df_products):
    def process_info(text):
        if text is None: return text
        r = 0
        text_c = ""
        for i in text:
            if r == 0 and i != "<": text_c += i
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






    
if __name__ == "__main__":
    #========
    # Set arguments
    #========
    #main(model, dataset, edges, task_version, test_subset, loss_fct="cosine_mse", batch_size=32):
    # task version: 1= task 1; 2=task 2
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, default="graphsage", help='Choose the GNN to use')
    parser.add_argument("--dataset", type=str, default="wands", help='Choose the dataset to use: esci or wands')
    parser.add_argument("--edges", type=str, default="old_2", help="Choose the edge creation for the Product Graph")
    parser.add_argument("--task_version", type=int, default=2, help='Choose the version of testing: 1 = to predict unseen questions; 2 = rank relavants to known questions .')
    parser.add_argument("--test_subset", type=int, default=0, help="Choose the subset, and it's size. {0,...,8}")
    parser.add_argument("--loss_fct", type=str, default="cosine_mse", help="Choose the Loss Function for the approach")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    args = parser.parse_args()

    main(args.model, args.dataset, args.edges, args.task_version, args.test_subset, args.loss_fct ,args.batch_size)