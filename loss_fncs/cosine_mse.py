import pandas as pd
import numpy as np
import torch.nn.functional as F

def custom_loss_func(out, embeddings, labels):
    cos_sim = F.cosine_similarity(out, embeddings, dim=1)
    return F.mse_loss(cos_sim, labels)