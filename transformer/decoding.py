"""Inferencia: decodificacao greedy e beam search com penalidade de comprimento.

O paper usa beam de tamanho 4 e alpha = 0.6 (secao 6.1).
"""

import torch

from .masking import make_source_mask, make_target_mask


@torch.no_grad()
def greedy_decode(model, src, max_len=64, bos_idx=1, eos_idx=2, src_mask=None):
    """Gera 1 sequencia por linha do batch pegando sempre o argmax.

    Retorna (B, L) incluindo o <bos> inicial.
    """
    model.eval()
    device = src.device
    pad_idx = model.config.pad_idx
    if src_mask is None:
        src_mask = make_source_mask(src, pad_idx)

    memory = model.encode(src, src_mask)
    batch = src.size(0)
    ys = torch.full((batch, 1), bos_idx, dtype=torch.long, device=device)
    finished = torch.zeros(batch, dtype=torch.bool, device=device)

    for _ in range(max_len - 1):
        out = model.decode(ys, memory, src_mask, make_target_mask(ys, pad_idx))
        next_token = model.generator(out[:, -1]).argmax(dim=-1)
        next_token = torch.where(finished, torch.full_like(next_token, pad_idx), next_token)
        ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
        finished |= next_token == eos_idx
        if bool(finished.all()):
            break
    return ys


@torch.no_grad()
def beam_search(model, src, beam_size=4, max_len=64, bos_idx=1, eos_idx=2,
                length_penalty=0.6, return_all=False):
    """Beam search para UMA sequencia de origem: src precisa ser (1, L) ou (L,).

    length_penalty e o alpha de Wu et al.: lp = ((5+|Y|)/6)^alpha, usado no paper.
    Retorna (L,) com a melhor hipotese, ou a lista [(score, tokens), ...] se
    return_all=True.
    """
    model.eval()
    if src.dim() == 1:
        src = src.unsqueeze(0)
    if src.size(0) != 1:
        raise ValueError("beam_search processa uma sequencia por vez (batch=1)")

    device = src.device
    pad_idx = model.config.pad_idx
    src_mask = make_source_mask(src, pad_idx)
    memory = model.encode(src, src_mask)

    # Replica a memoria para todos os feixes
    memory = memory.expand(beam_size, -1, -1).contiguous()
    beam_src_mask = src_mask.expand(beam_size, -1, -1, -1).contiguous()

    ys = torch.full((beam_size, 1), bos_idx, dtype=torch.long, device=device)
    scores = torch.full((beam_size,), float("-inf"), device=device)
    scores[0] = 0.0  # no primeiro passo todos os feixes sao identicos
    finished = []

    for step in range(max_len - 1):
        out = model.decode(ys, memory, beam_src_mask, make_target_mask(ys, pad_idx))
        log_probs = torch.log_softmax(model.generator(out[:, -1]), dim=-1)  # (beam, V)
        candidates = scores.unsqueeze(1) + log_probs
        flat = candidates.view(-1)

        top_scores, top_idx = flat.topk(beam_size)
        beam_idx = torch.div(top_idx, log_probs.size(-1), rounding_mode="floor")
        token_idx = top_idx % log_probs.size(-1)

        ys = torch.cat([ys[beam_idx], token_idx.unsqueeze(1)], dim=1)
        scores = top_scores

        for i in range(beam_size):
            if token_idx[i].item() == eos_idx:
                length = ys.size(1)
                lp = ((5.0 + length) / 6.0) ** length_penalty
                finished.append((scores[i].item() / lp, ys[i].clone()))
                scores[i] = float("-inf")  # tira o feixe encerrado da disputa

        if len(finished) >= beam_size or bool(torch.isinf(scores).all()):
            break

    if not finished:  # nenhum <eos> gerado dentro de max_len
        lp = ((5.0 + ys.size(1)) / 6.0) ** length_penalty
        finished = [(scores[i].item() / lp, ys[i].clone()) for i in range(beam_size)]

    finished.sort(key=lambda pair: pair[0], reverse=True)
    return finished if return_all else finished[0][1]
