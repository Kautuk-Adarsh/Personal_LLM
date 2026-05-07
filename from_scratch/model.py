# model.py
import numpy as np
from from_scratch.core import Tokenizer, Embedding, Attention, Network, Layer, Neuron

class LLM:
    def __init__(self, text: str, embed_dim: int, hidden_dim: int):
        self.tokenizer = Tokenizer(text)
        self.embedding = Embedding(self.tokenizer.vocab_size(), embed_dim)
        self.attention = Attention(embed_dim)

        if hidden_dim < 1:
            raise ValueError("hidden_dim must be at least 1")

        layer1 = Layer([
            Neuron(np.random.randn(embed_dim) * 0.01, 0.0)
            for _ in range(hidden_dim)
        ])
        layer2 = Layer([
           Neuron(np.random.randn(hidden_dim) * 0.01, 0.0)
            for _ in range(embed_dim)
        ])
        self.network = Network([layer1, layer2])

    def forward(self, text: str):
        token_ids = self.tokenizer.encode(text)
        if not token_ids:
            return []

        embedded = self.embedding.forward(token_ids)
        attended = self.attention.forward(embedded)
        pooled = attended.mean(axis=0)
        return self.network.forward(pooled)