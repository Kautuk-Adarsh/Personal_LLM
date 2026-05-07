# core/tokenizer.py
# Converts raw text into token IDs and back.
# Builds a vocabulary from training text — each unique word gets an integer ID.

class Tokenizer:
    def __init__(self, text: str):
        words = text.split()
        self.vocab = {}
        x = 0
        for word in words:
            if word not in self.vocab:
                self.vocab[word] = x
                x += 1
        self.reverse_vocab = {id: word for word, id in self.vocab.items()}

    def encode(self, text: str) -> list:
        words = text.split()
        ids = []
        for word in words:
            if word in self.vocab:
                ids.append(self.vocab[word])
            else:
                raise ValueError(f"Unknown word: '{word}' — not in vocabulary")
        return ids

    def decode(self, ids: list) -> str:
        words = []
        for i in ids:
            words.append(self.reverse_vocab[i])
        return " ".join(words)

    def vocab_size(self) -> int:
        return len(self.vocab)