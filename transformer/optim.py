"""Schedule de learning rate (Noam) e label smoothing — secao 5.3 e 5.4 do paper."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR


def noam_lambda(step, d_model=512, warmup=4000):
    """lrate = d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)   (eq. 3).

    Sobe linearmente ate `warmup` passos, depois cai com 1/sqrt(step).
    """
    step = max(step, 1)  # evita divisao por zero no passo 0
    return (d_model ** -0.5) * min(step ** -0.5, step * warmup ** -1.5)


class NoamLR(LambdaLR):
    """Aplica o schedule do paper por cima de um otimizador.

    Use com Adam(lr=1.0, betas=(0.9, 0.98), eps=1e-9) — o lr base precisa ser 1.0
    porque o schedule devolve a taxa absoluta, nao um fator.
    """

    def __init__(self, optimizer, d_model=512, warmup=4000, last_epoch=-1):
        self.d_model = d_model
        self.warmup = warmup
        super().__init__(optimizer, lambda step: noam_lambda(step, d_model, warmup), last_epoch)


def make_optimizer(model, d_model=512, warmup=4000, betas=(0.9, 0.98), eps=1e-9):
    """Adam + NoamLR exatamente com os hiperparametros da secao 5.3."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=betas, eps=eps)
    return optimizer, NoamLR(optimizer, d_model=d_model, warmup=warmup)


class LabelSmoothingLoss(nn.Module):
    """Label smoothing com eps_ls = 0.1 (secao 5.4).

    A massa 1-eps fica no token correto e eps e espalhada pelo restante do
    vocabulario (excluindo o proprio token e o padding). Piora a perplexidade
    mas melhora BLEU, como o paper reporta.
    """

    def __init__(self, vocab_size, pad_idx=0, smoothing=0.1, reduction="mean"):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.reduction = reduction

    def forward(self, logits, target):
        """logits: (B, L, V) ou (N, V);  target: (B, L) ou (N,)."""
        logits = logits.reshape(-1, self.vocab_size)
        target = target.reshape(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Classes que recebem a massa suavizada: todas menos a correta e o pad
        n_spread = self.vocab_size - 2 if self.pad_idx is not None else self.vocab_size - 1
        true_dist = torch.full_like(log_probs, self.smoothing / max(n_spread, 1))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        if self.pad_idx is not None:
            true_dist[:, self.pad_idx] = 0.0

        non_pad = target != self.pad_idx if self.pad_idx is not None else torch.ones_like(target, dtype=torch.bool)
        true_dist = true_dist * non_pad.unsqueeze(1)

        loss = -(true_dist * log_probs).sum(dim=-1)
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.sum() / non_pad.sum().clamp(min=1)  # media por token real
