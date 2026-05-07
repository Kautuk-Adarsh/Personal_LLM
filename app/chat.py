# app/chat.py
import torch
from core.tokenizer import Tokenizer
from core.model import LLM


class ChatEngine:
    def __init__(self, model_path: str, tokenizer_path: str):
        print("Loading tokenizer...")
        self.tokenizer = Tokenizer.from_json(tokenizer_path)

        print("Loading model...")
        self.model = LLM(
            tokenizer=self.tokenizer,
            embed_dim=512,
            hidden_dim=1024,
            num_heads=8,
            num_layers=6,
            dropout=0.0,
        )
        self.device = torch.device("cpu")
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print("Model ready.")

    def answer(self, question: str) -> str:
        return self.model.generate(
            question, max_new_tokens=30, temperature=0.8, top_k=50
        )
