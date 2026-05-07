# core/tokenizer.py
import json


class Tokenizer:
    def __init__(self, text: str = ""):
        words = text.split()
        self.vocab = {}
        x = 0
        for word in words:
            if word not in self.vocab:
                self.vocab[word] = x
                x += 1
        self.reverse_vocab = {id: word for word, id in self.vocab.items()}

    @classmethod
    def from_json(cls, filepath: str):
        with open(filepath, "r") as f:
            data = json.load(f)
        tokenizer = cls()
        tokenizer.vocab = data["vocab"]
        tokenizer.reverse_vocab = {int(k): v for k, v in data["reverse_vocab"].items()}
        return tokenizer

    def encode(self, text: str) -> list:
        words = text.split()
        ids = []
        for word in words:
            if word in self.vocab:
                ids.append(self.vocab[word])
            else:
                ids.append(self.vocab["<UNK>"])
        return ids

    def decode(self, ids: list) -> str:
        words = []
        for i in ids:
            word = self.reverse_vocab.get(i, "<UNK>")
            if word in ["<PAD>", "<BOS>", "<EOS>"]:
                continue
            words.append(word)
        return " ".join(words)

    def vocab_size(self) -> int:
        return len(self.vocab)
