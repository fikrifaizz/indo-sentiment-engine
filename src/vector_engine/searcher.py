import chromadb
import torch
import numpy as np
from src.config import CHROMA_DB_DIR, COLLECTION_NAME

class ReviewSearcher:
    def __init__(self, model, tokenizer, device):
        # Inisialisasi ChromaDB Client
        self.client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        self.collection = self.client.get_collection(name=COLLECTION_NAME)
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def _get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # --- BAGIAN PERBAIKAN (THE FIX) ---
            # Kita cek: Apakah ini model Classifier? Jika ya, panggil 'bert'-nya saja.
            if hasattr(self.model, "bert"):
                outputs = self.model.bert(**inputs)
            else:
                # Fallback: Jika ini memang model raw (AutoModel), panggil langsung.
                outputs = self.model(**inputs)
        
        # Sekarang 'outputs' pasti punya last_hidden_state
        token_embeddings = outputs.last_hidden_state
        
        # Mean Pooling Logic
        attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * attention_mask, 1)
        sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
        embedding = sum_embeddings / sum_mask
        
        return embedding.cpu().numpy()[0].tolist()

    def search(self, query, top_k=5):
        # Generate vector dari query user
        query_vec = self._get_embedding(query)
        
        # Cari di ChromaDB
        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=top_k
        )
        
        # Formatting Output agar bersih
        clean_results = []
        # Handle case jika result kosong
        if not results['documents']:
            return []

        for i in range(len(results['documents'][0])):
            clean_results.append({
                "text": results['documents'][0][i],
                "rating": results['metadatas'][0][i]['rating'],
                "sentiment_label": results['metadatas'][0][i]['label'],
                "distance": results['distances'][0][i]
            })
            
        return clean_results