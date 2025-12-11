from sentence_transformers import SentenceTransformer
import underthesea
import torch

class EmbeddingModel:
    def __init__(self):
        self.model = SentenceTransformer("keepitreal/vietnamese-sbert").to('cuda' if torch.cuda.is_available() else 'cpu')
        # print("Dim embedding:", self.model.get_sentence_embedding_dimension())
    
    def embed_text(self, doc):
        if not isinstance(doc, str):
            return []
        list_sen = underthesea.sent_tokenize(doc)
        # print(list_sen[0])
        
        embeddings = self.model.encode(list_sen)
        return list_sen, embeddings

if __name__ == "__main__":
    model = EmbeddingModel()
    import pandas as pd
    from pprint import pprint
    df = pd.read_json('train_clean.jsonl', lines =True)
    text = df.loc[0, 'text']
    sents, embs = model.embed_text(text)
    pprint(sents)
    pprint(embs)
    print(embs.shape)