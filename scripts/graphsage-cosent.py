import torch
from torch import nn

from scripts.graphsage import custom_create_graph, custom_data, custom_collate, custom_gnn
from sentence_transformers import util   

def custom_loss_func(emb1, emb2, labels):
        scores = util.pairwise_cos_sim(emb1, emb2)
        scores = scores * 20                        #Maybe change this value 
        scores = scores[:, None] - scores[None, :]

        # label matrix indicating which pairs are relevant
        labels = labels[:, None] < labels[None, :]
        labels = labels.float()

        # mask out irrelevant pairs so they are negligible after exp()
        scores = scores - (1 - labels) * 1e12

        # append a zero as e^0 = 1
        scores = torch.cat((torch.zeros(1).to(scores.device), scores.view(-1)), dim=0)
        loss = torch.logsumexp(scores, dim=0)

        return loss