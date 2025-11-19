# Semantic-Product-Search-with-Graph-Transformers
This is our implementation for the candidate short paper: 
"Semantic Product Search with Graph Transformers"

## Introduction
In this work, we propose a graph-based semantic product
retrieval framework for known queries that combines sentence-
transformer embeddings with graph neural networks (GNNs) to
refine product representations for ranking.
See Figure: ![img.png](img.png)
This repository provides a easy to use and scalable framework for exploring different Graph Neural Network (GNN) architectures for this problem.


## Datasets
We test our models on two datasets, the [Amazon ESCI dataset](https://github.com/amazon-science/esci-data) and the [WANDS Wayfair dataset](https://github.com/wayfair/WANDS). We reduce the size of the datasets by only using a subset of the data and only using english products and queries. 

## Requirements
For installing all baseline packages, you can create a virtual enviroment with python. 
For using the entire repository, please first download all requirements:
```
pip install -r requirements.txt
```
For using the same datasets, clone them and move the specified files into the specified folders:
```
git clone https://github.com/amazon-science/esci-data.git
git clone https://github.com/wayfair/WANDS.git
```
And place the files as specified below:
```
data/esci-data/shopping_queries_dataset_examples.parquet
data/esci-data/shopping_queries_dataset_products.parquet
data/wands-data/product.csv
data/wands-data/query.csv
data/wands-data/label.csv
```
For using the models ``gtpyg-gtconv.py`` it is also required to clone [gt-pyg](https://github.com/pgniewko/gt-pyg). 

# Usage
For running one model, use the ``Experiment.py`` script:
```
python Experiment.py model dataset size test_subset --edges gc_random --batch_size 32 --add_edges 16 --loss_fct cosine_mse
```
With the following arguments:
* ``model``: specifies the model to use from ``/scripts``
* ``dataset``: decides which dataset to use, either esci or wands
* ``size``: specifies the size of judgments, either in {10000, 50000, 100000}
* ``test_subset``: which data subset to use: between 0 to 9
* ``--edges gc_random``: specifies which edge rule to use
* ``--batch_size 32``: specifies the batch size
* ``--add_edges 16``: amount of minimal edges per node (if enough nodes with shared attribute and value)
* ``--loss_fct cosine_mse``: specify the used loss function for learning

To run multiple experiments automatically and average their results, use ``ExperimentBatchTester.py``

