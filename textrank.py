from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import networkx as nx
from embedd import EmbeddingModel

class textrank_selection:
    def __init__(self, embedds, topk = 20):
        self.embedds = embedds
        self.topk = topk
    
    def select_sentences(self):
        sim_matrix = cosine_similarity(self.embedds)
        
        # Loại đường chéo chính
        np.fill_diagonal(sim_matrix, 0)
        
        # Tính pagerank
        # Tính đồ thị
        nx_graph = nx.from_numpy_array(sim_matrix)
        # Tính điểm pagerank
        scores = nx.pagerank(nx_graph)
        # print("Scores:", scores)
        # exit()
        rank = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        rank = [idx for idx, score in rank[:self.topk]]
        return rank

if __name__ == "__main__":
    model = EmbeddingModel()
    import pandas as pd
    from pprint import pprint
    df = pd.read_json('train_clean.jsonl', lines =True)
    text = df.loc[0, 'text']
    sents, embs = model.embed_text(text)
    textrank = textrank_selection(embs, 20)
    rank = textrank.select_sentences()
    print(rank)
    