"""Testes do Transformer. Roda com pytest ou direto: python tests/test_transformer.py

Os testes checam propriedades da arquitetura (causalidade, invariancia a padding,
formulas do paper), nao so formatos de tensor.
"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformer import (  # noqa: E402
    LabelSmoothingLoss,
    LearnedPositionalEmbedding,
    MultiHeadAttention,
    PositionalEncoding,
    average_checkpoints,
    Transformer,
    TransformerConfig,
    beam_search,
    greedy_decode,
    make_target_mask,
    noam_lambda,
    scaled_dot_product_attention,
    subsequent_mask,
)
from transformer.flash import flash_attention, sdpa_attention  # noqa: E402

torch.manual_seed(0)
PAD, BOS, EOS = 0, 1, 2


def tiny_model(**kwargs):
    cfg = dict(src_vocab_size=50, tgt_vocab_size=50, num_layers=2, d_model=32,
               num_heads=4, d_ff=64, dropout=0.0)
    cfg.update(kwargs)
    return Transformer(TransformerConfig(**cfg)).eval()


# --------------------------------------------------------------------------- #
# Atencao
# --------------------------------------------------------------------------- #

def test_attention_weights_are_a_distribution():
    q, k, v = (torch.randn(2, 4, 7, 8) for _ in range(3))
    out, attn = scaled_dot_product_attention(q, k, v)
    assert out.shape == (2, 4, 7, 8)
    assert torch.allclose(attn.sum(-1), torch.ones(2, 4, 7), atol=1e-5)


def test_attention_respects_mask():
    q, k, v = (torch.randn(1, 1, 3, 4) for _ in range(3))
    mask = torch.tensor([[[[True, True, False]]]]).expand(1, 1, 3, 3)
    _, attn = scaled_dot_product_attention(q, k, v, mask=mask)
    assert torch.allclose(attn[..., 2], torch.zeros(1, 1, 3), atol=1e-7), "posicao mascarada recebeu peso"


def test_attention_scaling_matches_formula():
    """Confere a divisao por sqrt(d_k) contra um calculo manual."""
    torch.manual_seed(1)
    q, k, v = (torch.randn(1, 1, 5, 16) for _ in range(3))
    out, _ = scaled_dot_product_attention(q, k, v)
    manual = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(16), dim=-1) @ v
    assert torch.allclose(out, manual, atol=1e-6)


def test_multihead_equals_concat_of_heads():
    """h cabecas em paralelo == h atencoes independentes concatenadas."""
    mha = MultiHeadAttention(d_model=16, num_heads=4, dropout=0.0).eval()
    x = torch.randn(2, 6, 16)
    ref = mha(x, x, x, need_weights=True)

    q = mha.w_q(x).view(2, 6, 4, 4).transpose(1, 2)
    k = mha.w_k(x).view(2, 6, 4, 4).transpose(1, 2)
    v = mha.w_v(x).view(2, 6, 4, 4).transpose(1, 2)
    heads = [scaled_dot_product_attention(q[:, i], k[:, i], v[:, i])[0] for i in range(4)]
    manual = mha.w_o(torch.cat(heads, dim=-1))
    assert torch.allclose(ref, manual, atol=1e-5)


def test_multihead_matches_pytorch_reference():
    """Compara com nn.MultiheadAttention do PyTorch usando os mesmos pesos.

    Se as duas implementacoes divergirem, o erro esta na nossa.
    """
    torch.manual_seed(3)
    d_model, heads = 32, 4
    ours = MultiHeadAttention(d_model, heads, dropout=0.0).eval()
    reference = torch.nn.MultiheadAttention(d_model, heads, dropout=0.0, bias=False,
                                            batch_first=True).eval()

    with torch.no_grad():
        reference.in_proj_weight.copy_(
            torch.cat([ours.w_q.weight, ours.w_k.weight, ours.w_v.weight], dim=0)
        )
        reference.out_proj.weight.copy_(ours.w_o.weight)

    x = torch.randn(2, 7, d_model)
    causal = subsequent_mask(7)  # (1, 1, 7, 7), True = visivel

    got = ours(x, x, x, mask=causal)
    expected, _ = reference(x, x, x, attn_mask=~causal[0, 0])  # PyTorch: True = bloqueado
    assert torch.allclose(got, expected, atol=1e-5), (got - expected).abs().max().item()


def test_multihead_rejects_bad_head_count():
    try:
        MultiHeadAttention(d_model=10, num_heads=4)
    except ValueError:
        return
    raise AssertionError("d_model nao divisivel por num_heads deveria falhar")


# --------------------------------------------------------------------------- #
# Codificacao posicional
# --------------------------------------------------------------------------- #

def test_positional_encoding_matches_paper_formula():
    d_model, pos, i = 64, 7, 5
    pe = PositionalEncoding(d_model, dropout=0.0).pe[0]
    expected_sin = math.sin(pos / (10000 ** (2 * i / d_model)))
    expected_cos = math.cos(pos / (10000 ** (2 * i / d_model)))
    assert abs(pe[pos, 2 * i].item() - expected_sin) < 1e-6
    assert abs(pe[pos, 2 * i + 1].item() - expected_cos) < 1e-6


def test_positional_encoding_is_bounded_and_unique():
    pe = PositionalEncoding(32, dropout=0.0, max_len=100).pe[0]
    assert pe.abs().max() <= 1.0 + 1e-6
    assert not torch.allclose(pe[10], pe[11]), "posicoes diferentes com mesma codificacao"


def test_positional_encoding_handles_odd_d_model():
    pe = PositionalEncoding(31, dropout=0.0, max_len=10).pe
    assert pe.shape == (1, 10, 31)


def test_sinusoidal_extrapolates_but_learned_does_not():
    """Motivo da escolha do paper: senoides valem para qualquer comprimento."""
    sinusoidal = PositionalEncoding(16, dropout=0.0, max_len=8)
    learned = LearnedPositionalEmbedding(16, dropout=0.0, max_len=8)
    curta = torch.zeros(1, 8, 16)
    assert sinusoidal(curta).shape == learned(curta).shape == (1, 8, 16)

    longa = torch.zeros(1, 12, 16)
    try:
        learned(longa)
    except ValueError:
        pass
    else:
        raise AssertionError("posicoes aprendidas deveriam recusar L > max_len")


def test_learned_positional_model_trains():
    model = tiny_model(positional="learned").train()
    src = torch.randint(4, 50, (2, 5))
    tgt = torch.randint(4, 50, (2, 5))
    model(src, tgt).sum().backward()
    assert model.pos_encoding.embedding.weight.grad is not None


# --------------------------------------------------------------------------- #
# Mascaras
# --------------------------------------------------------------------------- #

def test_subsequent_mask_is_lower_triangular():
    m = subsequent_mask(4)[0, 0]
    assert m.equal(torch.ones(4, 4, dtype=torch.bool).tril())


def test_target_mask_combines_padding_and_causality():
    tgt = torch.tensor([[BOS, 5, 6, PAD]])
    m = make_target_mask(tgt, PAD)[0, 0]
    assert not m[:, 3].any(), "coluna de padding deveria estar toda mascarada"
    assert not m[0, 1], "posicao 0 nao pode ver a posicao 1"


# --------------------------------------------------------------------------- #
# Propriedades do modelo completo
# --------------------------------------------------------------------------- #

def test_forward_shapes():
    model = tiny_model()
    src = torch.randint(4, 50, (3, 9))
    tgt = torch.randint(4, 50, (3, 7))
    assert model(src, tgt).shape == (3, 7, 50)
    assert model.encode(src).shape == (3, 9, 32)


def test_decoder_cannot_see_the_future():
    """Mudar um token futuro nao pode alterar as saidas anteriores a ele."""
    model = tiny_model()
    src = torch.randint(4, 50, (1, 6))
    tgt = torch.tensor([[BOS, 10, 11, 12, 13]])
    out_a = model(src, tgt)

    tgt_b = tgt.clone()
    tgt_b[0, 3] = 40  # altera a posicao 3
    out_b = model(src, tgt_b)

    assert torch.allclose(out_a[:, :3], out_b[:, :3], atol=1e-6), "vazamento de informacao futura"
    assert not torch.allclose(out_a[:, 3], out_b[:, 3], atol=1e-6)


def test_padding_does_not_change_predictions():
    """Acrescentar padding na origem nao pode mexer no resultado."""
    model = tiny_model()
    src = torch.tensor([[5, 6, 7, 8]])
    src_padded = torch.tensor([[5, 6, 7, 8, PAD, PAD]])
    tgt = torch.tensor([[BOS, 9, 10]])
    assert torch.allclose(model(src, tgt), model(src_padded, tgt), atol=1e-5)


def test_all_padding_row_does_not_produce_nan():
    """Uma sequencia 100% padding zera a linha inteira do softmax — nao pode dar NaN."""
    model = tiny_model()
    src = torch.tensor([[5, 6, 7], [PAD, PAD, PAD]])
    tgt = torch.tensor([[BOS, 9, 10], [BOS, 9, 10]])
    out = model(src, tgt)
    assert torch.isfinite(out).all(), "NaN/inf com sequencia toda de padding"


def test_base_model_parameter_count_matches_paper():
    """Modelo base com vocabulario BPE de 37k: ~65M parametros (Table 3)."""
    model = Transformer(TransformerConfig(37000, 37000))
    millions = model.num_parameters() / 1e6
    assert 60 < millions < 70, f"esperado ~65M, obtido {millions:.1f}M"


def test_weight_tying():
    model = tiny_model(share_embeddings=True, tie_generator=True)
    assert model.src_embed.lut.weight is model.tgt_embed.lut.weight
    assert model.generator.weight is model.tgt_embed.lut.weight


def test_gradients_reach_every_parameter():
    model = tiny_model().train()
    src = torch.randint(4, 50, (2, 5))
    tgt = torch.randint(4, 50, (2, 5))
    model(src, tgt).sum().backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"sem gradiente: {missing}"


def test_eval_mode_is_deterministic():
    model = tiny_model(dropout=0.5).eval()
    src = torch.randint(4, 50, (2, 5))
    tgt = torch.randint(4, 50, (2, 5))
    assert torch.allclose(model(src, tgt), model(src, tgt))


def test_encoder_keeps_positions_distinct_at_init():
    """Post-LN com N=6 e Xavier puro reduz a distancia entre posicoes a ~16% ja na
    inicializacao (e o treino em corpus pequeno colapsa num saco de palavras). A
    escala 1/sqrt(2N) nas saidas dos sublayers mantem ~80%."""
    torch.manual_seed(0)
    model = Transformer(TransformerConfig(100, 100, num_layers=6, dropout=0.0)).eval()
    src = torch.randint(4, 100, (1, 12))
    entrada = model.pos_encoding(model.src_embed(src))[0]
    saida = model.encode(src)[0]
    ratio = (torch.cdist(saida, saida).mean() / torch.cdist(entrada, entrada).mean()).item()
    assert ratio > 0.5, f"encoder mistura as posicoes ja na inicializacao (razao {ratio:.2f})"


def test_attention_projections_have_no_bias():
    """As equacoes do paper sao QW^Q, KW^K, VW^V e Concat(...)W^O — sem bias."""
    mha = MultiHeadAttention(32, 4)
    assert all(getattr(mha, n).bias is None for n in ("w_q", "w_k", "w_v", "w_o"))


def test_attention_maps_are_captured():
    model = tiny_model()
    src = torch.randint(4, 50, (1, 6))
    tgt = torch.randint(4, 50, (1, 5))
    model(src, tgt, need_weights=True)
    maps = model.attention_maps()
    assert "encoder.0.self_attn" in maps and "decoder.0.cross_attn" in maps
    assert maps["decoder.0.cross_attn"].shape == (1, 4, 5, 6)


# --------------------------------------------------------------------------- #
# FlashAttention (Dao et al., 2022): mesma conta, sem a matriz L x L
# --------------------------------------------------------------------------- #

def _tres_tensores(b=2, h=4, l_q=37, l_k=37, d_k=16, d_v=16, dtype=torch.float64):
    """Q, K, V em float64 — precisao suficiente para comparar gradiente."""
    torch.manual_seed(0)
    q = torch.randn(b, h, l_q, d_k, dtype=dtype, requires_grad=True)
    k = torch.randn(b, h, l_k, d_k, dtype=dtype, requires_grad=True)
    v = torch.randn(b, h, l_k, d_v, dtype=dtype, requires_grad=True)
    return q, k, v


def _mascaras(l=37):
    causal = subsequent_mask(l)
    padding = torch.ones(2, 1, 1, l, dtype=torch.bool)
    padding[1, ..., 20:] = False
    return {"sem": None, "causal": causal, "padding": padding,
            "causal+padding": causal & padding}


def test_flash_attention_matches_math_exactly():
    """FlashAttention e atencao EXATA, nao aproximacao: o softmax online
    reescala os blocos ja acumulados e chega no mesmo numero."""
    q, k, v = _tres_tensores()
    esperado, _ = scaled_dot_product_attention(q, k, v)
    # blocos que nao dividem 37 de proposito, para pegar erro de borda
    obtido, pesos = flash_attention(q, k, v, block_q=8, block_k=5)
    assert torch.allclose(esperado, obtido, atol=1e-12)
    assert pesos is None, "o algoritmo nao materializa a matriz de pesos"


def test_flash_backward_matches_autograd_under_every_mask():
    """O backward recomputa P em vez de le-lo, e ainda assim devolve o MESMO
    gradiente que o autograd calcula sobre a matriz inteira."""
    for nome, mask in _mascaras().items():
        q, k, v = _tres_tensores()
        alvo = torch.randn(2, 4, 37, 16, dtype=torch.float64)

        esperado, _ = scaled_dot_product_attention(q, k, v, mask=mask)
        g_esperado = torch.autograd.grad(esperado, (q, k, v), alvo)

        obtido, _ = flash_attention(q, k, v, mask=mask, block_q=8, block_k=5)
        g_obtido = torch.autograd.grad(obtido, (q, k, v), alvo)

        for nome_grad, a, b in zip("QKV", g_esperado, g_obtido):
            assert torch.allclose(a, b, atol=1e-10), f"d{nome_grad} com mascara {nome}"


def test_flash_backward_passes_gradcheck_with_dropout():
    """Com semente fixa a funcao e deterministica, entao gradcheck se aplica.
    Passar aqui prova que o backward refaz a MESMA mascara de dropout do
    forward — se errasse o sorteio, o gradiente nao bateria."""
    q, k, v = _tres_tensores(b=1, h=2, l_q=9, l_k=9, d_k=8, d_v=8)
    fn = lambda a, b, c: flash_attention(a, b, c, dropout_p=0.3, block_q=4,
                                         block_k=3, semente=1234)[0]
    assert torch.autograd.gradcheck(fn, (q, k, v), eps=1e-6, atol=1e-8)


def test_flash_dropout_is_reproducible_from_the_seed():
    q, k, v = _tres_tensores(b=1, h=2, l_q=9, l_k=9, d_k=8, d_v=8)
    a, _ = flash_attention(q, k, v, dropout_p=0.3, semente=7)
    b, _ = flash_attention(q, k, v, dropout_p=0.3, semente=7)
    c, _ = flash_attention(q, k, v, dropout_p=0.3, semente=8)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_flash_attention_handles_d_v_different_from_d_k():
    """O paper usa d_v = d_k, mas a equacao nao exige: a saida herda d_v de V."""
    q, k, v = _tres_tensores(l_q=19, l_k=23, d_k=16, d_v=32)
    esperado, _ = scaled_dot_product_attention(q, k, v)
    obtido, _ = flash_attention(q, k, v, block_q=5, block_k=7)
    assert obtido.shape == (2, 4, 19, 32)
    assert torch.allclose(esperado, obtido, atol=1e-12)


def test_flash_attention_respects_masks():
    """Mascara causal e mascara de padding, inclusive com broadcast em L_q."""
    q, k, v = _tres_tensores(l_q=13, l_k=13)
    for mask in (subsequent_mask(13), None):
        esperado, _ = scaled_dot_product_attention(q, k, v, mask=mask)
        obtido, _ = flash_attention(q, k, v, mask=mask, block_q=4, block_k=4)
        assert torch.allclose(esperado, obtido, atol=1e-12)


def test_flash_attention_survives_a_fully_masked_row():
    """Linha 100% mascarada nao pode virar NaN, nem no forward nem no backward."""
    q, k, v = _tres_tensores(b=1, h=2, l_q=6, l_k=6)
    mask = torch.ones(1, 1, 6, 6, dtype=torch.bool)
    mask[..., 3, :] = False
    obtido, _ = flash_attention(q, k, v, mask=mask, block_q=2, block_k=2)
    assert torch.isfinite(obtido).all()
    grads = torch.autograd.grad(obtido.sum(), (q, k, v))
    assert all(torch.isfinite(g).all() for g in grads)


def test_sdpa_attention_matches_math():
    """O kernel fundido do PyTorch faz a mesma conta, saida e gradiente."""
    q, k, v = _tres_tensores(l_q=21, l_k=21)
    mask = subsequent_mask(21)
    alvo = torch.randn(2, 4, 21, 16, dtype=torch.float64)
    esperado, _ = scaled_dot_product_attention(q, k, v, mask=mask)
    g_esperado = torch.autograd.grad(esperado, (q, k, v), alvo)
    obtido, _ = sdpa_attention(q, k, v, mask=mask)
    g_obtido = torch.autograd.grad(obtido, (q, k, v), alvo)
    assert torch.allclose(esperado, obtido, atol=1e-12)
    for a, b in zip(g_esperado, g_obtido):
        assert torch.allclose(a, b, atol=1e-10)


def test_model_gives_same_logits_and_grads_with_every_attention_impl():
    """Trocar a implementacao nao muda nem a saida nem o gradiente do modelo."""
    torch.manual_seed(0)
    src = torch.randint(4, 50, (2, 7))
    tgt = torch.randint(4, 50, (2, 6))
    referencia, grad_ref = None, None
    for impl in ("math", "flash", "sdpa"):
        torch.manual_seed(0)
        cfg = TransformerConfig(50, 50, num_layers=2, d_model=32, num_heads=4,
                                d_ff=64, dropout=0.0, attention=impl)
        model = Transformer(cfg).train()
        logits = model(src, tgt)
        logits.sum().backward()
        g = model.encoder.layers[0].self_attn.w_q.weight.grad.clone()
        if referencia is None:
            referencia, grad_ref = logits.detach(), g
        else:
            assert torch.allclose(referencia, logits.detach(), atol=1e-5), impl
            assert torch.allclose(grad_ref, g, atol=1e-5), f"gradiente com {impl}"


def test_need_weights_falls_back_to_math():
    """Pedir os pesos com impl='flash' cai no caminho que os materializa."""
    torch.manual_seed(0)
    cfg = TransformerConfig(50, 50, num_layers=1, d_model=32, num_heads=4,
                            d_ff=64, dropout=0.0, attention="flash")
    model = Transformer(cfg).eval()
    model(torch.randint(4, 50, (1, 5)), torch.randint(4, 50, (1, 4)), need_weights=True)
    maps = model.attention_maps()
    assert maps["decoder.0.cross_attn"].shape == (1, 4, 4, 5)


def test_flash_saves_only_o_and_logsumexp():
    """O forward guarda para o backward apenas tensores O(L), nunca O(L^2).

    Se a matriz de probabilidades ficasse guardada, o total salvo passaria de
    L*L por cabeca. Aqui o maior tensor salvo tem L*d_k elementos.
    """
    q, k, v = _tres_tensores(b=1, h=2, l_q=64, l_k=64, d_k=8, d_v=8)
    saida, _ = flash_attention(q, k, v, block_q=16, block_k=16)
    salvos = saida.grad_fn.saved_tensors
    maior = max(s.numel() for s in salvos)
    assert maior <= 1 * 2 * 64 * 8, f"algo O(L^2) ficou guardado: {maior} elementos"
    assert all(s.numel() != 1 * 2 * 64 * 64 for s in salvos)


# --------------------------------------------------------------------------- #
# Dados: BPE (secao 5.1), vocabulario compartilhado e batches por tokens
# --------------------------------------------------------------------------- #

CORPUS = [
    "the cat sat on the mat", "the cat ate the fish", "a dog sat on the cat",
    "the fish sat", "cats and dogs", "the mat", "a fish and a cat",
]


def test_bpe_learns_frequent_pairs_first():
    from transformer import learn_bpe
    merges = learn_bpe(CORPUS, num_merges=3, min_pair_freq=1)
    assert len(merges) == 3
    # "the" e "cat" sao as palavras mais frequentes: os primeiros merges vem delas
    first_symbols = "".join(a + b for a, b in merges)
    assert "th" in first_symbols or "at" in first_symbols


def test_bpe_roundtrip_and_continuation_marker():
    from transformer import BPE, learn_bpe
    bpe = BPE(learn_bpe(CORPUS, num_merges=20, min_pair_freq=1))
    pieces = bpe.encode("the cats sat")
    assert BPE.decode(pieces) == "the cats sat"
    # todo pedaco que nao termina palavra carrega o marcador
    joined = " ".join(pieces).replace("@@ ", "")
    assert joined == "the cats sat"


def test_bpe_unseen_word_is_split_into_known_symbols():
    from transformer import BPE, learn_bpe
    bpe = BPE(learn_bpe(CORPUS, num_merges=20, min_pair_freq=1))
    pieces = bpe.encode("catfish")           # nunca vista, mas feita de partes vistas
    assert len(pieces) >= 2 and BPE.decode(pieces) == "catfish"


def test_shared_vocab_ties_all_three_weight_matrices():
    """Secao 3.4: um vocabulario unico permite amarrar src_embed, tgt_embed e generator."""
    from transformer.data import Vocab
    vocab = Vocab.build(CORPUS, bpe_merges=10)
    model = Transformer(TransformerConfig(len(vocab), len(vocab), num_layers=1, d_model=16,
                                          num_heads=2, d_ff=32))
    assert model.src_embed is model.tgt_embed
    assert model.generator.weight is model.src_embed.lut.weight


def test_vocab_encode_decode_roundtrip():
    from transformer.data import BOS_IDX, EOS_IDX, Vocab
    vocab = Vocab.build(CORPUS, bpe_merges=15)
    ids = vocab.encode("the cat sat on the mat")
    assert ids[0] == BOS_IDX and ids[-1] == EOS_IDX
    assert vocab.decode(ids) == "the cat sat on the mat"
    restored = Vocab.from_dict(vocab.to_dict())
    assert restored.encode("a dog") == vocab.encode("a dog")


def test_token_batch_sampler_respects_budget():
    from transformer.data import TokenBatchSampler
    lengths = [3, 10, 4, 9, 5, 8, 6, 7, 20, 2]
    sampler = TokenBatchSampler(lengths, max_tokens=24, shuffle=False)
    seen = []
    for batch in sampler:
        assert max(lengths[i] for i in batch) * len(batch) <= 24
        seen.extend(batch)
    assert sorted(seen) == list(range(len(lengths))), "toda frase entra exatamente uma vez"


# --------------------------------------------------------------------------- #
# Otimizacao
# --------------------------------------------------------------------------- #

def test_noam_schedule_peaks_at_warmup():
    warmup, d_model = 4000, 512
    lrs = [noam_lambda(s, d_model, warmup) for s in range(1, 12000)]
    peak_step = max(range(len(lrs)), key=lrs.__getitem__) + 1
    assert abs(peak_step - warmup) <= 1, f"pico em {peak_step}, esperado {warmup}"
    assert lrs[0] < lrs[warmup - 1] > lrs[-1]
    expected_peak = d_model ** -0.5 * warmup ** -0.5
    assert abs(lrs[warmup - 1] - expected_peak) < 1e-9


def test_label_smoothing_ignores_padding():
    loss_fn = LabelSmoothingLoss(vocab_size=10, pad_idx=PAD, smoothing=0.1)
    logits = torch.randn(1, 3, 10)
    target_all_pad = torch.full((1, 3), PAD)
    assert loss_fn(logits, target_all_pad).item() == 0.0


def test_label_smoothing_beats_confident_wrong_answer():
    loss_fn = LabelSmoothingLoss(vocab_size=6, pad_idx=PAD, smoothing=0.1)
    target = torch.tensor([[4]])
    right = torch.tensor([[[0.0, 0.0, 0.0, 0.0, 10.0, 0.0]]])
    wrong = torch.tensor([[[0.0, 0.0, 0.0, 10.0, 0.0, 0.0]]])
    assert loss_fn(right, target) < loss_fn(wrong, target)


# --------------------------------------------------------------------------- #
# Decodificacao
# --------------------------------------------------------------------------- #

def test_greedy_decode_shape_and_bos():
    model = tiny_model()
    src = torch.randint(4, 50, (2, 6))
    out = greedy_decode(model, src, max_len=12, bos_idx=BOS, eos_idx=EOS)
    assert out.size(0) == 2 and out.size(1) <= 12
    assert (out[:, 0] == BOS).all()


def test_beam_search_returns_ranked_hypotheses():
    model = tiny_model()
    src = torch.randint(4, 50, (1, 6))
    hyps = beam_search(model, src, beam_size=3, max_len=10, bos_idx=BOS, eos_idx=EOS, return_all=True)
    scores = [s for s, _ in hyps]
    assert scores == sorted(scores, reverse=True)


# --------------------------------------------------------------------------- #
# Aprendizado de verdade
# --------------------------------------------------------------------------- #

def test_model_overfits_a_tiny_batch():
    """Prova de que o grafo treina: decorar 4 sequencias tem que zerar a loss."""
    torch.manual_seed(0)
    model = tiny_model(dropout=0.0).train()
    src = torch.randint(4, 50, (4, 7))
    tgt = torch.cat([torch.full((4, 1), BOS), src], dim=1)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    first = last = None
    for step in range(220):
        logits = model(src, tgt[:, :-1])
        loss = loss_fn(logits.reshape(-1, 50), tgt[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()

    assert last < 0.05, f"loss ficou em {last:.3f} (inicial {first:.3f}) — nao decorou"


def test_config_roundtrip_tolerates_missing_fields():
    """Checkpoint de uma versao antiga (sem campos novos) tem que carregar."""
    cfg = TransformerConfig(50, 50, num_layers=2, d_model=32, num_heads=4, d_ff=64)
    data = cfg.to_dict()
    assert TransformerConfig.from_dict(data) == cfg

    data.pop("positional")          # campo adicionado depois
    data["campo_inexistente"] = 42  # campo removido depois
    restored = TransformerConfig.from_dict(data)
    assert restored.positional == "sinusoidal" and restored.d_model == 32


def test_checkpoint_averaging():
    """Media de 2 checkpoints tem que ser o ponto medio dos pesos (secao 6.1)."""
    import tempfile
    from pathlib import Path as _Path

    a, b = tiny_model(), tiny_model()
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i, m in enumerate((a, b)):
            p = _Path(tmp) / f"ckpt{i}.pt"
            torch.save({"model": m.state_dict()}, p)
            paths.append(p)
        averaged = average_checkpoints(paths)

    key = "encoder.layers.0.self_attn.w_q.weight"
    expected = (a.state_dict()[key] + b.state_dict()[key]) / 2
    assert torch.allclose(averaged[key], expected, atol=1e-6)

    a.load_state_dict(averaged)  # precisa carregar sem erro


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} testes passaram")
    sys.exit(1 if failures else 0)
