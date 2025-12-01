import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from peft import PeftModel
from src.config import LORA_ADAPTER_PATH, BASE_MODEL_NAME

class EndoBertModel:
    def __init__(self, device=None):
        self.device = device if device else torch.device("cpu")
        self.tokenizer = None
        self.model = None
        
    def load_for_inference(self):
        print(f"Loading Model for Inference on {self.device} (OFFLINE)...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(LORA_ADAPTER_PATH, local_files_only=True)
        
        base_model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL_NAME, 
            num_labels=3,
            local_files_only=True
        )
        
        peft_model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH, local_files_only=True)
        self.model = peft_model.merge_and_unload()
        
        self.model.to(self.device)
        self.model.eval()
        return self.tokenizer, self.model

    def load_for_embedding(self):
        print(f"Loading Model for Embedding on {self.device} (OFFLINE)...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(LORA_ADAPTER_PATH, local_files_only=True)
        
        base_model = AutoModel.from_pretrained(BASE_MODEL_NAME, local_files_only=True)
        
        peft_model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH, local_files_only=True)
        self.model = peft_model.merge_and_unload()
        
        self.model.to(self.device)
        self.model.eval()
        return self.tokenizer, self.model