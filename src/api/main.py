from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import torch
import uvicorn
import os
from src.modeling.bert import EndoBertModel
from src.vector_engine.searcher import ReviewSearcher
from src.config import MAX_LEN

ml_components = {
    "model": None,
    "tokenizer": None,
    "device": None,
    "searcher": None
}

def get_ml_components():
    if ml_components["model"] is None:
        print("Membangunkan Model...")
        
        # Deteksi Device
        device = torch.device("cpu")
        if torch.cuda.is_available(): device = torch.device("cuda")
        
        print(f"   Device: {device}")
        
        # Load Model
        bert_handler = EndoBertModel(device=device)
        tokenizer, model = bert_handler.load_for_inference()
        
        # Load Searcher
        searcher = ReviewSearcher(model, tokenizer, device)
        
        # Simpan ke Global
        ml_components["model"] = model
        ml_components["tokenizer"] = tokenizer
        ml_components["device"] = device
        ml_components["searcher"] = searcher
        print("Model selesai dimuat!")
        
    return ml_components

# --- FASTAPI SETUP (TANPA LIFESPAN) ---
app = FastAPI(title="IndoInsight API")

# --- INPUT SCHEMAS ---
class SentimentRequest(BaseModel):
    text: str

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

@app.get("/")
def home():
    return {"message": "IndoInsight API is Online. Models will load on first request."}

@app.post("/analyze")
def analyze_sentiment(req: SentimentRequest):
    if not req.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    components = get_ml_components()
    
    model = components["model"]
    tokenizer = components["tokenizer"]
    device = components["device"]
    
    inputs = tokenizer(req.text, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        confidence, pred_idx = torch.max(probs, dim=-1)
        
    label_map = {0: "Negatif", 1: "Netral", 2: "Positif"}
    result = label_map[pred_idx.item()]
    
    return {
        "text": req.text,
        "sentiment": result,
        "confidence": round(confidence.item(), 4)
    }

@app.post("/search")
def search_reviews(req: SearchRequest):
    components = get_ml_components()
    searcher = components["searcher"]
    
    try:
        results = searcher.search(req.query, top_k=req.top_k)
        return {"query": req.query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)