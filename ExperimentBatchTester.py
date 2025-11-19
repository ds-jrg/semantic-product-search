from ExperimentStarter import start_one_test

""" This file allows to create multiple experiments and start them one after another

start_one_test(modules, dataset, size, edges, path_to_save, batch_size, add_edges, loss_fct)

Parameters:
    modules: the GNN modules to use for testing: ['graphsage','rgcn']
    dataset: specicifies which dataset to use: esci/wands
    size: specifies dataset size to test: 10000, 50000, 100000, 200000
    edges: which edge creation rule to use (add_edges default=4): gc_random
    path_to_save: where to store the results
    (Opt) batch_size: specify batch size (default = 64)
    (Opt) add_edges: how many edges minimum per node (default = 4)
    (Opt) loss_fct: can specify a different loss function (default = "cosine_mse")
"""

modules = ['transformerconv.py','gtpyg_gtconv.py']
dataset = "wands"
size = 10000
edges = "gc_random"
batch_size = 32
add_edges = 2
loss_fct = "cosine_mse"
path_to_save = f"outputs/modules-{dataset}-{size}"


# Example: Test the different models on different edge sizes
start_one_test(modules, dataset, size, edges, path_to_save, batch_size, 2, loss_fct)
start_one_test(modules, dataset, size, edges, path_to_save, batch_size, 4, loss_fct)
start_one_test(modules, dataset, size, edges, path_to_save, batch_size, 8, loss_fct)


