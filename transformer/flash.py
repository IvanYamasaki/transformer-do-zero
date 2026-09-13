"""FlashAttention: atencao exata sem materializar a matriz L x L.

Referencia: Dao, Fu, Ermon, Rudra, Re, "FlashAttention: Fast and Memory-Efficient
Exact Attention with IO-Awareness", NeurIPS 2022, e Dao, "FlashAttention-2", 2023.

A atencao do paper de 2017 calcula softmax(QK^T/sqrt(d_k))V materializando a
matriz de scores inteira, de tamanho (B, h, L, L). Essa matriz e o gargalo: ela
cresce com L^2 e precisa ir e voltar da memoria da GPU.

O detalhe que costuma passar batido e que o gargalo do TREINO e pior que o da
inferencia. Para calcular o gradiente, o autograd precisa da matriz de
probabilidades P do forward, entao ele a GUARDA ate o backward acontecer. Uma
pilha de 6 camadas com 8 cabecas segura 6 dessas matrizes ao mesmo tempo.

FlashAttention resolve os dois de uma vez:

  no forward   percorre Q em blocos de linhas e, para cada um, varre K e V em
               blocos de colunas, mantendo o softmax incremental. Guarda so a
               saida O e, por linha, o logsumexp L. Ambos sao O(L), nao O(L^2).

  no backward  NAO le a matriz P de lugar nenhum: recalcula cada bloco a partir
               de Q, K e do L guardado. Recomputar custa FLOPs, mas atencao e
               limitada por memoria, nao por conta, entao sai barato.

O resultado e EXATO nos dois sentidos: mesma saida e mesmo gradiente que a
implementacao ingenua, a menos de erro de ponto flutuante.

Duas implementacoes aqui:

    flash_attention  o algoritmo completo, forward e backward, em PyTorch puro.
                     Economiza memoria de verdade, inclusive no treino, mas NAO
                     acelera: o laco de blocos roda no interpretador.

    sdpa_attention   chama torch.nn.functional.scaled_dot_product_attention, que
                     em GPU despacha para o kernel FlashAttention-2 fundido em
                     CUDA. E este o caminho rapido.


O softmax incremental (forward)
-------------------------------

softmax([x, y]) pode ser montado a partir das partes, reescalando cada uma por
exp(m_parcial - m_novo), onde m e o maximo corrente da linha. Guardando por linha
o maximo m e a soma dos expoentes l, cada bloco novo corrige o que ja foi
acumulado antes de somar a sua contribuicao. No fim guarda-se

    L = m + log(l)

que e o logsumexp da linha. Com ele, P = exp(S - L) ja sai normalizado, o que
torna o backward direto.


A identidade que faz o backward caber em blocos
-----------------------------------------------

Derivando o = sum_j p_j v_j com p = softmax(s):

    dL/ds_i = p_i * ( dP_i - sum_j p_j dP_j ),      dP_j = dO . v_j

O termo sum_j p_j dP_j parece exigir a linha inteira de P de uma vez. Mas

    sum_j p_j dP_j = dO . ( sum_j p_j v_j ) = dO . o

ou seja, ele e igual a soma por linha de dO * O, que sao dois tensores O(L*d)
que ja existem. Chamando isso de D, cada bloco precisa so de D_i:

    dS_ij = P_ij * ( dP_ij - D_i )

E por isso que o backward pode ser feito bloco a bloco sem nunca montar P.
"""

import math

import torch
import torch.nn.functional as F

__all__ = ["flash_attention", "sdpa_attention", "IMPLEMENTACOES"]

IMPLEMENTACOES = ("math", "flash", "sdpa")


def _fatia(mask, i0, i1, j0, j1):
    """Recorta a mascara no bloco (linhas i0:i1, colunas j0:j1).

    A mascara pode vir com dimensoes de tamanho 1 para broadcast, por exemplo
    (B, 1, 1, L_k) no caso do padding. Uma dimensao de tamanho 1 nao e cortada.
    """
    if mask is None:
        return None
    if mask.size(-2) > 1:
        mask = mask[..., i0:i1, :]
    if mask.size(-1) > 1:
        mask = mask[..., j0:j1]
    return mask


def _mascara_dropout(forma, p, gerador, dtype, device):
    """Mascara de dropout reproduzivel: o backward refaz a mesma do forward."""
    manter = (torch.rand(forma, generator=gerador, device=device) >= p)
    return manter.to(dtype) / (1.0 - p)


class _FlashAttention(torch.autograd.Function):
    """Forward e backward em blocos. Nenhum dos dois monta a matriz L x L."""

    @staticmethod
    def forward(ctx, query, key, value, mask, dropout_p, block_q, block_k, semente):
        d_k = query.size(-1)
        escala = 1.0 / math.sqrt(d_k)
        l_q, l_k = query.size(-2), key.size(-2)
        # Acumula em float32 quando a entrada e half/bfloat16/float32; em
        # float64 acumula em float64, senao o teste de gradiente perde precisao.
        acc = torch.float32 if query.dtype.itemsize <= 4 else query.dtype
        piso = torch.finfo(acc).min

        saida = query.new_zeros(query.shape[:-1] + (value.size(-1),))
        # Uma estatistica por linha: o logsumexp. E O(L), nao O(L^2).
        lse = torch.full(query.shape[:-1] + (1,), piso,
                         dtype=acc, device=query.device)

        gerador = None
        if dropout_p > 0:
            gerador = torch.Generator(device=query.device)
            gerador.manual_seed(semente)

        with torch.no_grad():
            for i0 in range(0, l_q, block_q):
                i1 = min(i0 + block_q, l_q)
                q_bloco = query[..., i0:i1, :]
                o_bloco = torch.zeros(q_bloco.shape[:-1] + (value.size(-1),),
                                      dtype=acc, device=query.device)
                m_bloco = lse[..., i0:i1, :].clone()          # maximo corrente
                l_bloco = torch.zeros_like(m_bloco)           # soma dos expoentes

                for j0 in range(0, l_k, block_k):
                    j1 = min(j0 + block_k, l_k)
                    scores = torch.matmul(
                        q_bloco, key[..., j0:j1, :].transpose(-2, -1)).to(acc) * escala

                    recorte = _fatia(mask, i0, i1, j0, j1)
                    if recorte is not None:
                        # Piso finito, como na implementacao "math": uma linha
                        # 100% mascarada vira uniforme em vez de NaN.
                        scores = scores.masked_fill(~recorte, piso)

                    m_novo = torch.maximum(m_bloco, scores.amax(dim=-1, keepdim=True))
                    correcao = torch.exp(m_bloco - m_novo)
                    p = torch.exp(scores - m_novo)

                    # l acumula a soma SEM dropout: e o denominador do softmax.
                    l_bloco = correcao * l_bloco + p.sum(dim=-1, keepdim=True)

                    if dropout_p > 0:
                        p = p * _mascara_dropout(p.shape, dropout_p, gerador,
                                                 p.dtype, p.device)

                    o_bloco = correcao * o_bloco + torch.matmul(
                        p.to(value.dtype), value[..., j0:j1, :]).to(acc)
                    m_bloco = m_novo

                seguro = l_bloco.clamp(min=torch.finfo(acc).tiny)
                saida[..., i0:i1, :] = (o_bloco / seguro).to(query.dtype)
                lse[..., i0:i1, :] = m_bloco + seguro.log()

        ctx.save_for_backward(query, key, value, saida, lse)
        ctx.mask = mask
        ctx.dropout_p = dropout_p
        ctx.block_q = block_q
        ctx.block_k = block_k
        ctx.semente = semente
        ctx.acc = acc
        return saida

    @staticmethod
    def backward(ctx, d_saida):
        query, key, value, saida, lse = ctx.saved_tensors
        mask = ctx.mask
        dropout_p = ctx.dropout_p
        block_q, block_k = ctx.block_q, ctx.block_k

        d_k = query.size(-1)
        escala = 1.0 / math.sqrt(d_k)
        l_q, l_k = query.size(-2), key.size(-2)
        acc = ctx.acc
        piso = torch.finfo(acc).min

        d_saida = d_saida.contiguous()
        d_query = torch.zeros_like(query, dtype=acc)
        d_key = torch.zeros_like(key, dtype=acc)
        d_value = torch.zeros_like(value, dtype=acc)

        # D = soma por linha de (dO * O). Substitui o sum_j p_j dP_j que exigiria
        # a linha inteira de P. Dois tensores O(L*d) que ja estao na memoria.
        D = (d_saida.to(acc) * saida.to(acc)).sum(dim=-1, keepdim=True)

        gerador = None
        if dropout_p > 0:
            gerador = torch.Generator(device=query.device)
            gerador.manual_seed(ctx.semente)

        # Mesma ordem de varredura do forward, para o dropout sortear igual.
        for i0 in range(0, l_q, block_q):
            i1 = min(i0 + block_q, l_q)
            q_bloco = query[..., i0:i1, :]
            do_bloco = d_saida[..., i0:i1, :]
            lse_bloco = lse[..., i0:i1, :]
            d_bloco = D[..., i0:i1, :]
            dq_bloco = torch.zeros_like(q_bloco, dtype=acc)

            for j0 in range(0, l_k, block_k):
                j1 = min(j0 + block_k, l_k)
                k_bloco = key[..., j0:j1, :]
                v_bloco = value[..., j0:j1, :]

                # P recalculado do zero. Ele nao foi guardado em lugar nenhum.
                scores = torch.matmul(
                    q_bloco, k_bloco.transpose(-2, -1)).to(acc) * escala
                recorte = _fatia(mask, i0, i1, j0, j1)
                if recorte is not None:
                    scores = scores.masked_fill(~recorte, piso)
                p = torch.exp(scores - lse_bloco)      # ja normalizado pelo lse

                mascara_drop = None
                if dropout_p > 0:
                    mascara_drop = _mascara_dropout(p.shape, dropout_p, gerador,
                                                    p.dtype, p.device)
                    p_usado = p * mascara_drop
                else:
                    p_usado = p

                d_value[..., j0:j1, :] += torch.matmul(
                    p_usado.transpose(-2, -1), do_bloco.to(acc))

                dp = torch.matmul(do_bloco.to(acc), v_bloco.transpose(-2, -1).to(acc))
                if mascara_drop is not None:
                    dp = dp * mascara_drop

                # Jacobiano do softmax, em forma de bloco
                ds = p * (dp - d_bloco) * escala

                dq_bloco += torch.matmul(ds, k_bloco.to(acc))
                d_key[..., j0:j1, :] += torch.matmul(ds.transpose(-2, -1), q_bloco.to(acc))

            d_query[..., i0:i1, :] = dq_bloco

        return (d_query.to(query.dtype), d_key.to(key.dtype), d_value.to(value.dtype),
                None, None, None, None, None)


def flash_attention(query, key, value, mask=None, dropout_p=0.0,
                    block_q=128, block_k=128, semente=None):
    """Attention(Q, K, V) por blocos, com softmax online. Resultado exato.

    query: (..., L_q, d_k)
    key:   (..., L_k, d_k)
    value: (..., L_k, d_v)
    mask:  broadcastavel para (..., L_q, L_k); True = posicao visivel.

    O gradiente tambem e calculado por blocos, recomputando os scores em vez de
    guarda-los, entao o pico de memoria nao cresce com L^2 nem no treino.

    Retorna (saida, None). O `None` no lugar dos pesos de atencao e proposital:
    a matriz de pesos nunca chega a existir. Para visualizar a atencao, use a
    implementacao "math".
    """
    if semente is None:
        semente = int(torch.randint(0, 2 ** 31 - 1, (1,)).item())
    saida = _FlashAttention.apply(query, key, value, mask, dropout_p,
                                  block_q, block_k, semente)
    return saida, None


def sdpa_attention(query, key, value, mask=None, dropout_p=0.0):
    """Caminho rapido: o kernel fundido do PyTorch.

    Em GPU, `scaled_dot_product_attention` escolhe sozinho entre o kernel
    FlashAttention-2, o kernel de memoria eficiente e o caminho ingenuo, segundo
    dtype, alinhamento e formato da mascara. Em CPU ha uma versao fundida
    propria, mais modesta.

    Retorna (saida, None) pelo mesmo motivo de `flash_attention`.
    """
    saida = F.scaled_dot_product_attention(query, key, value, attn_mask=mask,
                                           dropout_p=dropout_p)
    return saida, None
