from ..ExperimentBaselines import Evaluator
from rank_bm25 import BM25Okapi

class Eval(Evaluator):
    def forwards(self, query, p_embeddings):
        scores = p_embeddings.get_scores(query)
        return scores

def create_necessary_thing():
    # create p_emb
    # create bm25
    tokenized_corpus = [prod.split(" ") for prod in all_product_info]
    p_emb = BM25Okapi(tokenized_corpus)
    return p_emb

#bm25 is p_emb
x = Eval(test_dataloader, bm25)
# maybe remove bm25 and put it directtly to evaluate
metrics = x.evaluate()