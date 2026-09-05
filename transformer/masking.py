"""Mascaras de padding e mascara causal (secao 3.2.3 do paper).

Convencao usada em todo o projeto: True = posicao visivel, False = mascarada.
"""

import torch


def padding_mask(tokens, pad_idx=0):
    """(B, L) de ids -> (B, 1, 1, L) booleano. True onde nao e padding."""
    return (tokens != pad_idx).unsqueeze(1).unsqueeze(2)


def subsequent_mask(size, device=None):
    """Mascara triangular inferior (1, 1, size, size): posicao i so ve j <= i.

    E o que impede o decoder de "olhar o futuro" durante o treino paralelo.
    """
    mask = torch.ones(size, size, dtype=torch.bool, device=device).tril()
    return mask.unsqueeze(0).unsqueeze(0)


def make_source_mask(src, pad_idx=0):
    return padding_mask(src, pad_idx)


def make_target_mask(tgt, pad_idx=0):
    """Combina padding do alvo com a mascara causal -> (B, 1, L, L)."""
    return padding_mask(tgt, pad_idx) & subsequent_mask(tgt.size(1), device=tgt.device)
