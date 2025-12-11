import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from textrank import textrank_selection
from embedd import EmbeddingModel
import warnings
warnings.filterwarnings("ignore")


class crossencoder:
    def __init__(self, model = 'namdp-ptit/ViRanker'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model
    def rank_sentences(self, doc, candidate):
        ce_tokenizer = AutoTokenizer.from_pretrained(self.model)
        ce_model = AutoModelForSequenceClassification.from_pretrained(self.model).to(self.device)
        ce_model.eval()
        
        batch_size = 16
        
        scores = []
        with torch.no_grad():
            for i in range(0, len(doc), batch_size):
                chunk_docs = doc[i:i+batch_size]
                chunk_cands = candidate[i:i+batch_size]
                toks = ce_tokenizer(chunk_cands, chunk_docs, truncation=True, padding=True, max_length=8194, return_tensors='pt').to(self.device)
                out = ce_model(**toks)
                logits = out.logits
                if logits.shape[-1] == 1:
                    sc = torch.sigmoid(logits).squeeze(-1).cpu().numpy().tolist()
                    # print(sc)
                else:
                    probs = torch.softmax(logits, dim=1)[:,1].cpu().numpy().tolist()
                    sc = probs
                scores.extend(sc)
        return scores
    
if __name__ == "__main__":
    model = crossencoder()
    emb = EmbeddingModel()
    import pandas as pd
    from pprint import pprint
    df = pd.read_json('train_clean.jsonl', lines =True)
    text = df.loc[0, 'text']
     
    print(len(text.split()))
    word = text.split()[:1000]
    text = ' '.join(word)
    sents, embs = emb.embed_text(text)
    textrank = textrank_selection(embs, 20)
    rank = textrank.select_sentences()
    cadidate_texts = [sents[idx] for idx in rank]
    doc_texts = [text]*len(cadidate_texts)
    ce_scores = model.rank_sentences(doc_texts, cadidate_texts)   
    print(ce_scores) 
