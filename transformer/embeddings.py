"""Embeddings de token e codificacao posicional senoidal (secao 3.4 e 3.5 do paper)."""

import math

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Embedding comum, multiplicado por sqrt(d_model) como descrito na secao 3.4."""

    def __init__(self, vocab_size, d_model, padding_idx=None):
        super().__init__()
        self.d_model = d_model
        self.lut = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)

    def forward(self, tokens):
        return self.lut(tokens) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Nao tem parametros treinaveis; fica num buffer e vai junto no state_dict.
    O paper aplica dropout na soma embedding + PE (secao 5.4).
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # exp(-log(10000) * 2i/d_model) == 1 / 10000^(2i/d_model), mais estavel numericamente
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].size(1)])  # fatia cobre d_model impar
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, max_len, d_model)

    def forward(self, x):
        """x: (B, L, d_model)"""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class LearnedPositionalEmbedding(nn.Module):
    """Posicoes aprendidas em vez de senoides (Table 3, linha E do paper).

    O paper reporta resultados "quase identicos" aos da versao senoidal; a
    senoidal foi preferida por extrapolar para sequencias mais longas que as
    vistas no treino. Aqui, passar de `max_len` e um erro explicito.
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.max_len = max_len
        self.embedding = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=d_model ** -0.5)

    def forward(self, x):
        length = x.size(1)
        if length > self.max_len:
            raise ValueError(
                f"sequencia de {length} tokens excede max_len={self.max_len}; "
                "posicoes aprendidas nao extrapolam (use a codificacao senoidal)"
            )
        positions = torch.arange(length, device=x.device)
        return self.dropout(x + self.embedding(positions).unsqueeze(0))
