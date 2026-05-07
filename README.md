# LLM From Scratch → Production Chat

A complete journey — building a language model from zero knowledge of 
neural networks to a deployed conversational AI in three weeks.


## Two versions in this repo

### `from_scratch/` — Pure NumPy implementation
Every component written by hand to understand the internals:
- **Tokenizer** — text → token IDs → text
- **Embedding** — token IDs → vectors
- **Attention** — Q, K, V matrices + softmax from scratch
- **Neuron / Layer / Network** — forward pass without any framework

### `pytorch/` — Production version
Full transformer trained and deployed:
- **Model** — 6 transformer blocks, 8 attention heads, 512 embed dim
- **Parameters** — ~94 million
- **Vocabulary** — 80,000 words
- **Pretraining** — wikitext-103 (500MB Wikipedia)
- **Fine-tuning** — SQuAD + TriviaQA + OpenBookQA (742k Q&A pairs)
- **Interface** — Flask web app with chat UI

## Architecture

```
text → Tokenizer → Embedding → Transformer Blocks → Output
                                      ↑
                            Attention + Feedforward
                            + LayerNorm + Residual
```

## Training infrastructure

```
Pretraining  → Google Colab TPU v5e
Fine-tuning  → Kaggle 2x T4 GPU
Precision    → float16 mixed precision (3x speedup)
Optimizer    → AdamW + gradient clipping + LR scheduler
```

## Project structure

```
llm-from-scratch/
├── from_scratch/         ← NumPy version
│   ├── core/
│   │   ├── tokenizer.py
│   │   ├── embedding.py
│   │   ├── attention.py
│   │   └── network.py
│   ├── model.py
│   └── main.py
│
├── app/                  ← Flask backend
│   ├── chat.py
│   └── interface.py
├── core/                 ← PyTorch model
│   ├── tokenizer.py
│   └── model.py
├── templates/
│   └── index.html
├── static/
│   ├── style.css
│   └── script.js
├── main.py
├── Dockerfile
└── requirements.txt
```

## How to run locally

```bash
git clone https://github.com/kautukadarsh/llm-from-scratch
cd llm-from-scratch
pip install torch flask numpy
python main.py
# open http://localhost:5000
```

## Live demo

[CLICK HERE](https://huggingface.co/spaces/Kautuk2003/llm-chat )

## Versioning

| Version | Description |
|---------|-------------|
| v1.0 | Working chat, 10 epochs fine-tuning |
| v1.1 | More training epochs, lower loss (planned) |
| v1.2 | 150M+ parameters (planned) |

## What's next

- Reduce loss below 2.0 with more training
- Add conversation history
- Scale to 150M parameters
