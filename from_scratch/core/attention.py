# Single-head self-attention.

import numpy as np

class Attention:
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim
        self.W_q = np.random.randn(embed_dim, embed_dim) * 0.01
        self.W_k = np.random.randn(embed_dim, embed_dim) * 0.01
        self.W_v = np.random.randn(embed_dim, embed_dim) * 0.01

    def softmax(self, x: np.ndarray) -> np.ndarray:
        e = np.exp(x)
        return e / np.sum(e, axis=-1, keepdims=True)

    def forward(self, x: np.ndarray) -> np.ndarray:
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v
        scores = Q @ K.T
        scaled = scores / np.sqrt(self.embed_dim)
        weights = self.softmax(scaled)
        return weights @ V