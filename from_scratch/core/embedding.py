# core/embedding.py
# Converts token IDs into dense vectors (embeddings).
# Each token gets a unique vector of floats — these are learned during training.

import numpy as np

class Embedding:
    def __init__(self, vocab_size: int, embed_dim: int):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.matrix = np.random.randn(vocab_size, embed_dim) * 0.01

    def forward(self, token_ids: list) -> np.ndarray:
        vectors = []
        for id in token_ids:
            vectors.append(self.matrix[id])
        return np.array(vectors)