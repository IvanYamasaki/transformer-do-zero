"""Utilitarios de checkpoint.

Media de checkpoints e o truque da secao 6.1 do paper: as ultimas 5 gravacoes
(20 para o modelo "big") sao promediadas antes da avaliacao, o que costuma valer
alguns decimos de BLEU de graca.
"""

from pathlib import Path

import torch


def average_checkpoints(paths, key="model"):
    """Media aritmetica dos pesos de varios checkpoints (secao 6.1).

    paths: lista de arquivos salvos com torch.save({"model": state_dict, ...}).
    Retorna o state_dict promediado, pronto para load_state_dict.
    """
    paths = [Path(p) for p in paths]
    if not paths:
        raise ValueError("nenhum checkpoint informado")

    averaged = None
    for path in paths:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt[key] if key in ckpt else ckpt

        if averaged is None:
            averaged = {k: v.clone().float() for k, v in state.items()}
            continue
        if state.keys() != averaged.keys():
            raise ValueError(f"{path} tem parametros diferentes dos demais checkpoints")
        for k, v in state.items():
            averaged[k] += v.float()

    for k in averaged:
        averaged[k] /= len(paths)
    return averaged


def count_parameters(model, trainable_only=True):
    """Conta parametros unicos (pesos amarrados contam uma vez so)."""
    seen, total = set(), 0
    for p in model.parameters():
        if trainable_only and not p.requires_grad:
            continue
        if id(p) not in seen:
            seen.add(id(p))
            total += p.numel()
    return total
