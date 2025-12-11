import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from scipy.special import expit # Sigmoid
from sklearn.preprocessing import MinMaxScaler

class MMRSelector:
    def __init__(self, top_k=5, diversity=0.7):

        self.top_k = top_k
        self.diversity = diversity

    def select(self, candidate_embs, doc_emb, ce_scores=None):
 
        if len(candidate_embs) == 0: return []
        
        if ce_scores is not None:
            mimax = MinMaxScaler()
            # Chuẩn hóa điểm Cross-Encoder về [0,1]
            scores = expit(ce_scores) 
            scores = mimax.fit_transform(scores.reshape(-1, 1)).reshape(-1)
            
        else:
            scores = cos_sim(candidate_embs, doc_emb.reshape(1, -1)).reshape(-1)

       # Ma trận tương đồng giữa các câu candidate với nhau
        candidate_sim_matrix = cos_sim(candidate_embs)
        np.fill_diagonal(candidate_sim_matrix, 0)
        selected_idx = []
        
        for _ in range(0, self.top_k):
            if len(selected_idx) == 0:
                idx = np.argmax(scores)
                selected_idx.append(idx)
                continue
            mmr_score = []
            for i in range(len(candidate_embs)):
                if i in selected_idx:
                    mmr_score.append(-np.inf)
                    continue
                max_sel = max([candidate_sim_matrix[i][j] for j in selected_idx])
                mmr_value = self.diversity * scores[i] - (1 - self.diversity) * max_sel
                mmr_score.append(mmr_value)
            id = np.argmax(mmr_score)
            selected_idx.append(int(id))
            
        return selected_idx
                            
                

if __name__ == "__main__":

    dummy_embs = np.random.rand(10, 768) 
    dummy_doc = np.random.rand(768)      
    dummy_ce_scores = np.random.rand(10) 
    
    
    selector = MMRSelector(top_k=3, diversity=0.3)
    
    selected_idx = selector.select(dummy_embs, dummy_doc, ce_scores=dummy_ce_scores)
    
    print("Index các câu được chọn:", selected_idx)
    # Kết quả VD: [2, 5, 0] -> Câu 2 hay nhất, Câu 5 bổ sung ý, Câu 0 bổ sung tiếp...