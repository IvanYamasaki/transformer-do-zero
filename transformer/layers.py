"""Feed-forward posicional e as camadas de encoder/decoder (secao 3.1 e 3.3 do paper)."""

import copy

import torch.nn as nn

from .attention import MultiHeadAttention


def clones(module, n):
    """n copias independentes (deep copy) de um modulo."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class PositionwiseFeedForward(nn.Module):
    """FFN(x) = max(0, x W_1 + b_1) W_2 + b_2   (eq. 2). d_ff = 2048 no modelo base."""

    def __init__(self, d_model=512, d_ff=2048, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))


class ResidualConnection(nn.Module):
    """LayerNorm(x + Dropout(Sublayer(x))) — secao 3.1, com o dropout da secao 5.4."""

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))


class EncoderLayer(nn.Module):
    """Self-attention + feed-forward, cada um com residual e norm."""

    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1, impl="math"):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, impl=impl)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayers = clones(ResidualConnection(d_model, dropout), 2)

    def forward(self, x, mask=None, need_weights=False):
        x = self.sublayers[0](x, lambda y: self.self_attn(y, y, y, mask, need_weights))
        return self.sublayers[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    """Self-attention mascarada + cross-attention sobre o encoder + feed-forward."""

    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1, impl="math"):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, impl=impl)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout, impl=impl)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayers = clones(ResidualConnection(d_model, dropout), 3)

    def forward(self, x, memory, src_mask=None, tgt_mask=None, need_weights=False):
        x = self.sublayers[0](x, lambda y: self.self_attn(y, y, y, tgt_mask, need_weights))
        x = self.sublayers[1](
            x, lambda y: self.cross_attn(y, memory, memory, src_mask, need_weights)
        )
        return self.sublayers[2](x, self.feed_forward)
