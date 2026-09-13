"""Scaled Dot-Product Attention e Multi-Head Attention.

Secao 3.2 de "Attention Is All You Need" (Vaswani et al., 2017).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .flash import IMPLEMENTACOES, flash_attention, sdpa_attention


def scaled_dot_product_attention(query, key, value, mask=None, dropout=None):
    """Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V   (eq. 1 do paper).

    query: (..., L_q, d_k)
    key:   (..., L_k, d_k)
    value: (..., L_k, d_v)
    mask:  broadcastavel para (..., L_q, L_k); True = posicao visivel.

    Retorna (saida, pesos_de_atencao).
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # Valor finito grande em vez de -inf: evita NaN caso uma linha inteira
        # seja mascarada (acontece com sequencias 100% padding num batch).
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

    attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        attn = dropout(attn)
    return torch.matmul(attn, value), attn


class MultiHeadAttention(nn.Module):
    """MultiHead(Q,K,V) = Concat(head_1..head_h) W^O,  head_i = Attention(QW_i^Q, KW_i^K, VW_i^V).

    Paper: h = 8 cabecas, d_k = d_v = d_model / h = 64.
    As h projecoes sao feitas por uma unica matriz d_model x d_model e depois
    reorganizadas em cabecas (matematicamente identico, bem mais rapido).

    As equacoes do paper nao tem termo de bias (sao produtos QW, KW, VW e
    Concat(...)W^O puros), por isso bias=False e o padrao.

    `impl` escolhe como o produto da atencao e feito. Os tres dao o mesmo
    resultado; mudam a memoria e a velocidade (ver transformer/flash.py):

        "math"   materializa a matriz (B, h, L, L). E o padrao, e o unico que
                 devolve os pesos para visualizacao.
        "flash"  FlashAttention em PyTorch puro, por blocos, com backward
                 proprio. Nao materializa a matriz nem no treino, entao
                 economiza memoria; nao acelera.
        "sdpa"   kernel fundido do PyTorch; em GPU vira FlashAttention-2.
    """

    def __init__(self, d_model=512, num_heads=8, dropout=0.1, bias=False, impl="math"):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) precisa ser divisivel por num_heads ({num_heads})")
        if impl not in IMPLEMENTACOES:
            raise ValueError(f"impl deve ser um de {IMPLEMENTACOES}, nao {impl!r}")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.impl = impl
        self.dropout_p = dropout

        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.attn_weights = None  # guardado para visualizacao

    def _split_heads(self, x):
        # (B, L, d_model) -> (B, h, L, d_k)
        batch, length, _ = x.shape
        return x.view(batch, length, self.num_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x):
        # (B, h, L, d_k) -> (B, L, d_model)
        batch, _, length, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch, length, self.d_model)

    def forward(self, query, key, value, mask=None, need_weights=False):
        """mask: (B, 1, L_q, L_k) ou (B, 1, 1, L_k). True = visivel."""
        if mask is not None and mask.dim() == 3:
            mask = mask.unsqueeze(1)  # insere a dimensao das cabecas

        q = self._split_heads(self.w_q(query))
        k = self._split_heads(self.w_k(key))
        v = self._split_heads(self.w_v(value))

        # "flash" e "sdpa" nunca montam a matriz de pesos, entao nao ha o que
        # capturar: quem pede os pesos cai no caminho "math".
        impl = "math" if need_weights else self.impl
        if impl == "flash":
            out, attn = flash_attention(q, k, v, mask=mask,
                                        dropout_p=self.dropout_p if self.training else 0.0)
        elif impl == "sdpa":
            out, attn = sdpa_attention(q, k, v, mask=mask,
                                       dropout_p=self.dropout_p if self.training else 0.0)
        else:
            out, attn = scaled_dot_product_attention(q, k, v, mask=mask, dropout=self.dropout)
        out = self.w_o(self._merge_heads(out))

        self.attn_weights = attn.detach() if (need_weights and attn is not None) else None
        return out
