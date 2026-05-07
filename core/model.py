# core/model.py
import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim   = embed_dim,
            num_heads   = num_heads,
            dropout     = dropout,
            batch_first = True
        )
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm1   = nn.LayerNorm(embed_dim)
        self.norm2   = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attended, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attended))
        x = self.norm2(x + self.feedforward(x))
        return x


class LLM(nn.Module):
    def __init__(self, tokenizer, embed_dim=512, hidden_dim=1024,
                 num_heads=8, num_layers=6, dropout=0.1, max_seq=512):
        super().__init__()
        self.tokenizer  = tokenizer
        self.max_seq    = max_seq
        vocab_size      = tokenizer.vocab_size()
        self.embedding  = nn.Embedding(vocab_size, embed_dim,
                                       padding_idx=tokenizer.vocab["<PAD>"])
        self.pos_embed  = nn.Embedding(max_seq, embed_dim)
        self.blocks     = nn.ModuleList([
            TransformerBlock(embed_dim, hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.norm    = nn.LayerNorm(embed_dim)
        self.output  = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids):
        seq_len   = token_ids.shape[1]
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)
        x = self.dropout(self.embedding(token_ids) + self.pos_embed(positions))
        mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=token_ids.device
        )
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)
        return self.output(x)

    def generate(self, question: str, max_new_tokens: int = 30,
                 temperature: float = 0.8, top_k: int = 50) -> str:
        self.eval()
        device    = next(self.parameters()).device
        prompt    = f"Question: {question} Answer:"
        words     = prompt.split()
        input_ids = []
        for word in words:
            if word in self.tokenizer.vocab:
                input_ids.append(self.tokenizer.vocab[word])
            else:
                input_ids.append(self.tokenizer.vocab["<UNK>"])

        input_tensor = torch.tensor(input_ids).unsqueeze(0).to(device)
        generated    = []

        with torch.no_grad():
            for _ in range(max_new_tokens):
                output = self.forward(input_tensor)
                logits = output[0, -1, :].clone()

                logits[self.tokenizer.vocab["<UNK>"]] = -float('inf')
                logits[self.tokenizer.vocab["<PAD>"]] = -float('inf')

                for prev_id in generated:
                    logits[prev_id] /= 2.0

                logits = logits / temperature

                top_k_logits, top_k_ids = torch.topk(logits, top_k)
                probs    = torch.softmax(top_k_logits, dim=-1)
                next_pos = torch.multinomial(probs, 1).item()
                next_id  = top_k_ids[int(next_pos)].item()

                if next_id == self.tokenizer.vocab["<EOS>"]:
                    break

                generated.append(next_id)
                next_tensor  = torch.tensor([[next_id]]).to(device)
                input_tensor = torch.cat([input_tensor, next_tensor], dim=1)

                if input_tensor.shape[1] >= self.max_seq:
                    break

        return self.tokenizer.decode(generated)