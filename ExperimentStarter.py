
import os
import json
import numpy as np
from Experiment import main as test_gnn
#from general_setup_triplet import main as test_gnn

# To simply get all models in folder scripts contained
#modules = os.listdir('scripts')
modules = ['rgcn.py']
print(modules)


# Test every module on the same task, same dataset. Use 3 different subsets, average them
dataset = "wands" # esci, wands
dataset_size = 10000 # dataset sizes in 10000, 50000, 100000
edges = "old_2_list" # specify graph creation
task_version = 2 # 1= task 1; 2=task 2
batch_size = 32
#path_to_save = ""
path_to_save = f"dataset-{dataset}-{dataset_size}_task{task_version}"



def start_one_test(modules, dataset, dataset_size, edges, task_version, batch_size, path_to_save):
    size_to_idx = {"10000":[0,1,2], "50000":[3,4,5], "100000":[6,7,8]}
    for module in modules:
        module = module[:-3]  # removes .py
        if module in  ['__pycach', '.ipynb_checkpoi']:
            continue
            
        score = []
        print(module)
        for test_subset in size_to_idx[dataset_size]:    # replace with subset
            try:
                result_json = test_gnn(module, dataset, edges, task_version, test_subset, batch_size)
                
                result = json.loads(result_json)
                
                score.append(result)
            except Exception as e:
                print(e)
                 
        try:
            avg_score = {key: float(np.mean([s[key] for s in score])) for key in score[0].keys()}
    
            print("Average score across runs:", json.dumps(avg_score, indent=2))
            with open(path_to_save, 'a') as f:
                f.write(f'{module,edges} :: {str(avg_score)}\n \n')
        except Exception as e:
            with open(path_to_save, 'a') as f:
                f.write(f'{module,edges} :: {e}\n\n')

if __name__ == "__main__":
    start_one_test(dataset, dataset_size, edges, task_version, batch_size, path_to_save)