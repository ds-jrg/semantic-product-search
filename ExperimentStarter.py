import os
import json
import numpy as np
from Experiment import main as test_gnn



def start_one_test(modules, dataset, size, edges, path_to_save, batch_size=64, add_edges=4, loss_fct="cosine_mse"):
    """ Starts three runs for dataset, size and then averages the results, stores them in file path_to_save
    """
    size_to_idx = {"10000":[0,1,2], "50000":[3,4,5], "100000":[6,7,8]}
    for module in modules:
        score = []
        print(module)
        for test_subset in size_to_idx[size]:    # replace with subset
            try:
                result_json = test_gnn(module, dataset, size, test_subset, edges, batch_size, add_edges, loss_fct)
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
