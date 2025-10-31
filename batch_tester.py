from test_starter import start_one_test


modules = ['transformerconv.py']
dataset = 0 # 0=esci, 1=wands
dataset_size = 10000 # dataset sizes in 10000, 50000, 100000
edges = "old_2_list" # specify graph creation
task_version = 2 # 1= task 1; 2=task 2
batch_size = 32
#path_to_save = ""
path_to_save = f"dataset-{dataset}-{dataset_size}_task"

"""
# Test the different models on different dataset sizes
start_one_test(dataset, dataset_size, edges, task_version, batch_size, path_to_save)
start_one_test(dataset, 50000, edges, task_version, batch_size, path_to_save)
start_one_test(dataset, 100000, edges, task_version, batch_size, path_to_save)
start_one_test(1, dataset_size, edges, task_version, batch_size, path_to_save)
start_one_test(1, 50000, edges, task_version, batch_size, path_to_save)
start_one_test(1, 100000, edges, task_version, batch_size, path_to_save)
"""

# Test different EDGE Types