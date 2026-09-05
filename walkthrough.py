"""Executa, passo a passo, o percurso pelo codigo descrito no README.

Cada passo aqui corresponde a um "Passo N" do tutorial, e refaz À MÃO o que uma
linha do pacote `transformer/` faz. No fim de cada um, compara com a saída do
módulo: quando imprime [IGUAL], o que você acabou de ler é exatamente o que o
código executa.

    python walkthrough.py                # tudo, na ordem do tutorial
    python walkthrough.py --list         # lista os passos
    python walkthrough.py --stage 8      # só o passo 8 (self-attention por dentro)
    python walkthrough.py --stage 8 --pause    # abre o pdb antes do passo
    python walkthrough.py --frase "Zwei Hunde spielen."

Usa o modelo base do paper treinado em checkpoints/multi30k.pt. Sem ele, cai
para um modelo aleatório: o fluxo é o mesmo, só os valores não significam nada.
"""

import argparse
import contextlib
import io
import math
import sys
from pathlib import Path

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from transformer import Transformer, TransformerConfig
from transformer.bpe import CONTINUE, END, pretokenize
from transformer.data import BOS_IDX, EOS_IDX, PAD_IDX, Vocab
from transformer.masking import make_source_mask, make_target_mask

FRASE = "Ein Mann geht."
PALAVRA_RARA = "Schneemobilen"


# --------------------------------------------------------------------------- #
# Impressão
# --------------------------------------------------------------------------- #

def titulo(n, texto, onde):
    print(f"\n{'=' * 78}")
    print(f"PASSO {n} — {texto}")
    print(f"código: {onde}")
    print("=" * 78)


def mostra(nome, t, n=5):
    vals = ", ".join(f"{v:+.4f}" for v in t.reshape(-1)[:n].tolist())
    print(f"  {nome:<30} {str(tuple(t.shape)):<16} [{vals}, ...]")


def matriz(nome, m, rot_lin, rot_col, fmt="{:6.2f}"):
    print(f"\n  {nome}")
    larg = max(len(r) for r in rot_lin) + 1
    print("  " + " " * (larg + 2) + " ".join(f"{c[:6]:>6}" for c in rot_col))
    for rotulo, linha in zip(rot_lin, m.tolist()):
        print(f"  {rotulo:>{larg}}  " + " ".join(fmt.format(v) for v in linha))


def confere(nome, a, b, atol=1e-5):
    if not torch.allclose(a, b, atol=atol):
        raise AssertionError(f"{nome} divergiu do passo a passo")
    print(f"\n  [IGUAL] o passo a passo acima reproduz {nome}")


# --------------------------------------------------------------------------- #

class Contexto:
    def __init__(self, checkpoint, frase):
        self.frase = frase
        caminho = Path(checkpoint)
        if caminho.exists():
            ckpt = torch.load(caminho, map_location="cpu", weights_only=False)
            if "vocab" not in ckpt:
                raise SystemExit(f"{caminho} é de uma versão antiga; rode train_multi30k.py")
            self.model = Transformer(TransformerConfig.from_dict(ckpt["config"]))
            self.model.load_state_dict(ckpt["model"])
            self.vocab = Vocab.from_dict(ckpt["vocab"])
            self.treinado = True
        else:
            print(f"(sem {caminho} — modelo aleatório; rode train_multi30k.py para "
                  "ver os números do tutorial)\n")
            frases = [frase, "Ein Hund läuft.", "Eine Frau sitzt.", "Ein Mann steht.",
                      "A man walks.", "A dog runs.", "A woman sits.", "A man stands."]
            self.vocab = Vocab.build(frases, bpe_merges=40)
            self.model = Transformer(TransformerConfig(
                len(self.vocab), len(self.vocab), num_layers=2, d_model=64,
                num_heads=8, d_ff=128, dropout=0.0))
            self.treinado = False
        self.model.eval()          # sem isso o dropout torna tudo irreproduzível
        self.cfg = self.model.config

    def rot_src(self):
        return [self.vocab.itos[i] for i in self.src[0].tolist()]


# --------------------------------------------------------------------------- #
# PARTE 1 — do texto para números
# --------------------------------------------------------------------------- #

def passo_1(c):
    titulo(1, "QUEBRAR A FRASE EM PEDAÇOS (BPE)", "transformer/bpe.py:27 e :122")
    print(f'\n  entrada: "{c.frase}"')
    palavras = pretokenize(c.frase)
    print(f"\n  --- bpe.py:27, pretokenize: separa palavras e pontuação, PRESERVANDO maiúsculas ---")
    print(f"  {palavras}")

    print(f"\n  --- bpe.py:122, BPE.segment_word em cada palavra ---")
    for w in palavras:
        print(f"  {w:<14} -> {c.vocab.bpe.segment_word(w)}")
    c.pedacos = c.vocab.tokenize(c.frase)
    print(f"\n  pedaços = {c.pedacos}")
    print("\n  Palavras frequentes ficam inteiras. Uma palavra rara mostra o BPE trabalhando:")
    _mostra_merges(c.vocab.bpe, PALAVRA_RARA)
    print(f"\n  '{CONTINUE}' no fim de um pedaço quer dizer: continua na próxima peça.")
    print(f"  O paper (5.1) usa BPE com ~37k tokens no WMT; aqui são {len(c.vocab.bpe)} merges.")


def _mostra_merges(bpe, word):
    """Repete o loop de segment_word imprimindo cada fusão."""
    symbols = list(word[:-1]) + [word[-1] + END]
    print(f"\n  {word}: começa como caracteres {[s.replace(END, '</w>') for s in symbols]}")
    passo = 0
    while len(symbols) > 1:
        best_rank, best_i = None, None
        for i, pair in enumerate(zip(symbols, symbols[1:])):
            rank = bpe.ranks.get(pair)
            if rank is not None and (best_rank is None or rank < best_rank):
                best_rank, best_i = rank, i
        if best_i is None:
            break
        a, b = symbols[best_i], symbols[best_i + 1]
        symbols[best_i:best_i + 2] = [a + b]
        passo += 1
        if passo <= 6:
            print(f"    merge #{best_rank:<5} {a!r} + {b!r:<8} -> {[s.replace(END, '</w>') for s in symbols]}")
        elif passo == 7:
            print("    ...")
    print(f"  resultado: {bpe.segment_word(word)}   ({passo} fusões; nenhum par restante está nos merges)")


def passo_2(c):
    titulo(2, "PEDAÇOS VIRAM IDS", "transformer/data.py:53-55  (Vocab.encode)")
    c.ids = c.vocab.encode(c.frase)
    print(f"\n  {'pedaço':<10} id")
    for tok, i in zip(["<bos>"] + c.pedacos + ["<eos>"], c.ids):
        obs = "   <- fora do vocabulário" if i == 3 else ""
        print(f"  {tok:<10} {i}{obs}")
    print(f"\n  ids = {c.ids}")
    print(f"\n  Ids especiais fixos (data.py:21): <pad>={PAD_IDX} <bos>={BOS_IDX} "
          f"<eos>={EOS_IDX} <unk>=3")
    print(f"  O vocabulário é UM SÓ para alemão e inglês ({len(c.vocab)} tokens): é o que")
    print("  permite amarrar embedding de origem, de destino e generator (model.py:122).")
    print("  Ordenado por frequência: ids baixos = pedaços comuns.")


def passo_3(c):
    titulo(3, "VIRAR TENSOR", "src = torch.tensor(ids).unsqueeze(0)")
    c.src = torch.tensor(c.ids).unsqueeze(0)
    print(f"\n  src = {c.src.tolist()}      forma {tuple(c.src.shape)}")
    print("\n  O unsqueeze(0) cria a dimensão de batch. Daqui pra frente toda forma")
    print("  tem batch na frente.")


def passo_4(c):
    titulo(4, "A MÁSCARA DE PADDING", "transformer/masking.py:11  (padding_mask)")
    etapa1 = c.src != c.cfg.pad_idx
    etapa2 = etapa1.unsqueeze(1)
    c.src_mask = etapa2.unsqueeze(2)
    print(f"\n  src != pad_idx      {str(tuple(etapa1.shape)):<12} {etapa1[0].tolist()}")
    print(f"  .unsqueeze(1)       {tuple(etapa2.shape)}   <- abre a dimensão das cabeças")
    print(f"  .unsqueeze(2)       {tuple(c.src_mask.shape)}   <- abre a dimensão das queries")
    print(f"\n  Os scores terão forma (batch, heads, queries, keys) = "
          f"(1, {c.cfg.num_heads}, {c.src.size(1)}, {c.src.size(1)}).")
    print("  Com a máscara em (1,1,1,L), o broadcast a estica para todas as cabeças")
    print("  e todas as queries de uma vez.")
    print("\n  Convenção do projeto: True = posição VISÍVEL.")
    confere("make_source_mask", make_source_mask(c.src, c.cfg.pad_idx).int(), c.src_mask.int())


# --------------------------------------------------------------------------- #
# PARTE 2 — o encoder
# --------------------------------------------------------------------------- #

def passo_5(c):
    titulo(5, "EMBEDDING (× √d_model)", "transformer/embeddings.py:18")
    emb_mod = c.model.src_embed
    d = emb_mod.d_model
    lut = emb_mod.lut(c.src)
    c.x_emb = lut * math.sqrt(d)

    print(f"\n  lut é um nn.Embedding({c.cfg.src_vocab_size}, {d}): a linha i é o vetor")
    print("  do pedaço i. Consultar é indexação pura, não multiplicação de matriz.\n")
    mostra("lut[<bos>]", lut[0, 0])
    mostra(f"lut[{c.pedacos[0]}]", lut[0, 1])
    print(f"\n  fator de escala: √{d} = {math.sqrt(d)}\n")
    mostra("emb[<bos>] = lut × √d", c.x_emb[0, 0])
    mostra(f"emb[{c.pedacos[0]}]", c.x_emb[0, 1])

    print(f"\n  {'pedaço':<10} {'‖lut‖':>8} {'‖lut × √d‖':>12}")
    for tok, a, b in zip(c.rot_src(), lut.norm(dim=-1)[0].tolist(), c.x_emb.norm(dim=-1)[0].tolist()):
        print(f"  {tok:<10} {a:>8.2f} {b:>12.2f}")
    print("\n  A codificação posicional que vem a seguir tem valores em [-1, 1].")
    print("  Sem a escala o conteúdo seria abafado pela posição.")

    confere("TokenEmbedding.forward", emb_mod(c.src), c.x_emb)


def passo_6(c):
    titulo(6, "CODIFICAÇÃO POSICIONAL", "transformer/embeddings.py:33-41 e :45")
    pe_mod = c.model.pos_encoding
    d = c.cfg.d_model
    L = c.x_emb.size(1)

    print("\n  --- metade 1: o cálculo, uma vez só no __init__ (linhas 33-41) ---")
    div_term = torch.exp(torch.arange(0, d, 2, dtype=torch.float) * (-math.log(10000.0) / d))
    print("  A linha 36 é a fórmula do paper reescrita:")
    print("      exp(-log(10000) · 2i/d)  ==  1 / 10000^(2i/d)")
    print("  Idêntico na matemática, mais estável nos números.\n")
    mostra("div_term", div_term)
    print(f"\n  As linhas 39-40 fatiam com passo 2: 0::2 recebe seno, 1::2 recebe cosseno.")
    print(f"  Resultado: buffer de forma {tuple(pe_mod.pe.shape)}\n")
    for pos in range(3):
        mostra(f"pe[posição {pos}]", pe_mod.pe[0, pos], 6)
    print("\n  Conferindo a fórmula na mão, pos=1, i=0:")
    print(f"    PE(1,0) = sin(1/10000^0) = {math.sin(1):+.6f}   buffer: {pe_mod.pe[0,1,0]:+.6f}")
    print(f"    PE(1,1) = cos(1/10000^0) = {math.cos(1):+.6f}   buffer: {pe_mod.pe[0,1,1]:+.6f}")

    print("\n  --- metade 2: o uso, a cada forward (linha 45) ---")
    c.x_pos = c.x_emb + pe_mod.pe[:, :L]
    print()
    mostra("emb[<bos>]", c.x_emb[0, 0])
    mostra("pe[0]", pe_mod.pe[0, 0])
    mostra("x_pos = soma dos dois", c.x_pos[0, 0])
    print("\n  É SOMA, não concatenação: a posição entra no mesmo espaço vetorial.")
    print(f"  Confira a 2ª dimensão: {c.x_emb[0,0,1]:+.4f} + {pe_mod.pe[0,0,1]:+.4f} "
          f"= {c.x_pos[0,0,1]:+.4f}")

    confere("PositionalEncoding.forward", pe_mod(c.x_emb), c.x_pos)


def passo_7(c):
    titulo(7, "A PILHA DE CAMADAS", "transformer/model.py:67  e  layers.py:49")
    print(f"\n  Encoder.forward é um for sobre as {c.cfg.num_layers} camadas (model.py:67-68).")
    print("  clones() usa deepcopy: as camadas são INDEPENDENTES, cada uma com seus")
    print("  próprios pesos. Não é a mesma camada aplicada N vezes.")
    print("\n  EncoderLayer.forward tem duas linhas, uma por sublayer:")
    print("    linha 50:  sublayers[0](x, lambda y: self_attn(y, y, y, mask))")
    print("    linha 51:  sublayers[1](x, self.feed_forward)")
    print("\n  O lambda existe porque ResidualConnection só sabe chamar sublayer(x)")
    print("  com UM argumento, mas a atenção precisa de (y, y, y, mask).")
    print("  E ser (y, y, y) — o mesmo tensor três vezes — é o que faz ser SELF-attention.")


def passo_8(c):
    titulo(8, "DENTRO DA SELF-ATTENTION", "transformer/attention.py:80-85 e :24-34")
    mha = c.model.encoder.layers[0].self_attn
    x = c.x_pos
    B, L, _ = x.shape
    h, d_k = mha.num_heads, mha.d_k
    rot = c.rot_src()

    print("\n  --- linhas 80-82: as projeções ---")
    print(f"  w_q, w_k, w_v são nn.Linear({mha.d_model}, {mha.d_model}, bias={mha.w_q.bias is not None}):")
    print("  as equações do paper são QW^Q, KW^K, VW^V — produtos puros, sem bias.\n")
    q, k, v = mha.w_q(x), mha.w_k(x), mha.w_v(x)
    mostra("Q[<bos>]", q[0, 0])
    mostra("K[<bos>]", k[0, 0])
    mostra("V[<bos>]", v[0, 0])

    print(f"\n  --- linha 68 (_split_heads): onde estão as {h} cabeças? ---")
    qh = q.view(B, L, h, d_k).transpose(1, 2)
    kh = k.view(B, L, h, d_k).transpose(1, 2)
    vh = v.view(B, L, h, d_k).transpose(1, 2)
    print(f"  {tuple(x.shape)}  --view-->  {(B, L, h, d_k)}  --transpose(1,2)-->  {tuple(qh.shape)}")
    print("   B  L  d_model         B  L  h  d_k                        B  h  L  d_k")
    print(f"\n  As {h} cabeças viraram dimensão de BATCH. Não existe loop sobre cabeças:")
    print(f"  o matmul seguinte opera nas {h} em paralelo.")

    print(f"\n  --- linha 24: scores = Q Kᵀ / √d_k ---")
    bruto = torch.matmul(qh, kh.transpose(-2, -1))
    scores = bruto / math.sqrt(d_k)
    print(f"  sem escala:      desvio padrão {bruto.std():.3f}")
    print(f"  ÷ √{d_k} = {math.sqrt(d_k):.3f}:  desvio padrão {scores.std():.3f}")
    print("  Com desvio grande o softmax quase vira one-hot e o gradiente morre.")
    matriz("scores da cabeça 0 (linha = quem pergunta, coluna = quem é olhado)",
           scores[0, 0], rot, rot)

    print(f"\n  --- linha 29: a máscara ---")
    mascarado = scores.masked_fill(~c.src_mask, torch.finfo(scores.dtype).min)
    print(f"  Posições False viram {torch.finfo(scores.dtype).min:.2e} — finfo.min, NÃO -inf.")
    print("  Com -inf, uma linha 100% mascarada faria o softmax dar 0/0 = NaN.")
    print("  Aqui nada muda: esta frase não tem padding.")

    print("\n  --- linha 31: softmax na última dimensão ---")
    attn = torch.softmax(mascarado, dim=-1)
    matriz("cabeça 0 — cada linha soma 1", attn[0, 0], rot, rot)
    if h > 1:
        matriz("cabeça 1 — mesmo texto, padrão diferente", attn[0, 1], rot, rot)
        print("\n  Cabeças diferentes aprendem views diferentes da mesma frase.")

    print("\n  --- linha 34: saída = pesos @ V ---")
    ctx = torch.matmul(attn, vh)
    print(f"  {tuple(attn.shape)} @ {tuple(vh.shape)} -> {tuple(ctx.shape)}")
    print("  Cada posição vira uma média dos V, ponderada pela atenção.")
    print("  É AQUI que a informação se move entre posições.")

    print("\n  --- linha 85: juntar as cabeças e projetar por W_O ---")
    concat = ctx.transpose(1, 2).contiguous().view(B, L, mha.d_model)
    c.attn_out = mha.w_o(concat)
    mostra("Concat(head_1..head_h)", concat)
    mostra("× W_O", c.attn_out)
    print("\n  Saímos com a mesma forma com que entramos.")

    confere("MultiHeadAttention.forward", mha(x, x, x, c.src_mask), c.attn_out)


def passo_9(c):
    titulo(9, "ADD & NORM", "transformer/layers.py:37  (ResidualConnection)")
    res = c.model.encoder.layers[0].sublayers[0]
    soma = c.x_pos + c.attn_out
    c.pos_norm = res.norm(soma)

    print("\n  linha 37:  return self.norm(x + self.dropout(sublayer(x)))")
    print("  É o post-LN do paper: LayerNorm(x + Sublayer(x)).\n")
    mostra("x             (entrada)", c.x_pos[0, 0])
    mostra("Sublayer(x)   (atenção)", c.attn_out[0, 0])
    mostra("x + Sublayer(x)", soma[0, 0])
    mostra("LayerNorm(...)", c.pos_norm[0, 0])

    medias = ", ".join(f"{v:+.5f}" for v in c.pos_norm.mean(-1)[0].tolist())
    desvios = ", ".join(f"{v:.4f}" for v in c.pos_norm.std(-1)[0].tolist())
    print(f"\n  média por posição:  [{medias}]")
    print(f"  desvio por posição: [{desvios}]")
    print("\n  Média ~0 e desvio ~1 em cada posição: é isso que mantém os valores")
    print("  estáveis mesmo empilhando muitas camadas.")
    print("  E o `x +` dá ao gradiente um caminho direto até as camadas iniciais.")

    confere("ResidualConnection.forward",
            res(c.x_pos, lambda y: c.model.encoder.layers[0].self_attn(y, y, y, c.src_mask)),
            c.pos_norm)


def passo_10(c):
    titulo(10, "FEED FORWARD", "transformer/layers.py:25  (PositionwiseFeedForward)")
    ffn = c.model.encoder.layers[0].feed_forward
    h1 = ffn.w_1(c.pos_norm)
    a1 = h1.relu()
    h2 = ffn.w_2(a1)

    print("\n  linha 25:  return self.w_2(self.dropout(self.w_1(x).relu()))")
    print("  Lendo de dentro pra fora: w_1 expande, ReLU corta, w_2 volta.\n")
    mostra("entrada", c.pos_norm[0, 0])
    mostra("w_1 (expande)", h1[0, 0])
    mostra("ReLU (zera negativos)", a1[0, 0])
    mostra("w_2 (volta)", h2[0, 0])
    print(f"\n  {c.cfg.d_model} -> {c.cfg.d_ff} -> {c.cfg.d_model}")
    print(f"  {100 * (a1 == 0).float().mean():.1f}% das ativações foram zeradas pelo ReLU.")
    print("\n  'Positionwise': o MESMO MLP roda em cada posição, sem misturar posições.")
    print("  Quem mistura posições é só a atenção. Essa divisão de papéis é o")
    print("  coração da arquitetura.")

    confere("PositionwiseFeedForward.forward", ffn(c.pos_norm), h2)


def passo_11(c):
    titulo(11, "REPETIR E OBTER A MEMÓRIA", "transformer/model.py:66-69")
    x = c.x_pos
    print(f"\n  Acompanhando só o vetor do token <bos> (o tensor todo é "
          f"{tuple(x.shape)}):\n")
    mostra("entrada (x_pos)", x[0, 0])
    for i, camada in enumerate(c.model.encoder.layers):
        x = camada(x, c.src_mask)
        mostra(f"depois da camada {i}", x[0, 0])

    print(f"\n  A forma NUNCA muda: {tuple(x.shape)} do começo ao fim.")
    print("  Empilhar camadas enriquece a representação, não a redimensiona.")
    c.memoria = c.model.encode(c.src, c.src_mask)
    confere("Transformer.encode", c.memoria, x)
    print("\n  Este tensor é a MEMÓRIA. O encoder rodou uma vez; o decoder vai")
    print("  consultá-la em todos os passos de geração.")


# --------------------------------------------------------------------------- #
# PARTE 3 — o decoder
# --------------------------------------------------------------------------- #

def passo_12(c):
    titulo(12, "O DECODER GERA PEDAÇO POR PEDAÇO", "transformer/decoding.py:23-36")
    rot_src = c.rot_src()
    ultima = c.model.decoder.layers[-1]

    print("\n  linha 23:  memory = model.encode(...)   <- FORA do loop")
    print("  A memória não depende do que já foi gerado. Recalcular seria desperdício.")
    print("  linha 25:  ys = [[<bos>]]                <- começa com um token só\n")

    ys = torch.full((1, 1), BOS_IDX, dtype=torch.long)

    for volta in range(c.src.size(1) + 50):
        rot_tgt = [c.vocab.itos[i] for i in ys[0].tolist()]
        print(f"\n  {'-' * 70}")
        print(f"  VOLTA {volta}:  ys = {ys.tolist()} = {rot_tgt}")

        tgt_mask = make_target_mask(ys, c.cfg.pad_idx)
        matriz("tgt_mask (masking.py:29 = padding & causal)",
               tgt_mask[0, 0].int(), rot_tgt, rot_tgt, fmt="{:6d}")

        oculto = c.model.decode(ys, c.memoria, c.src_mask, tgt_mask, need_weights=True)
        cross = ultima.cross_attn.attn_weights[0].mean(0)   # média das cabeças
        print(f"\n  cross-attention da última camada, posição atual (média das cabeças):")
        print("       " + " ".join(f"{r[:6]:>6}" for r in rot_src))
        print("       " + " ".join(f"{x:6.2f}" for x in cross[-1].tolist()))

        logits = c.model.generator(oculto[:, -1])[0]
        probs = torch.softmax(logits, dim=-1)
        topo = probs.topk(5)
        print(f"\n  generator (model.py:120): Linear({c.cfg.d_model} -> "
              f"{c.cfg.tgt_vocab_size}), depois softmax\n")
        print(f"    {'pedaço':<14} {'logit':>8} {'prob':>8}")
        for p, idx in zip(topo.values.tolist(), topo.indices.tolist()):
            print(f"    {c.vocab.itos[idx]:<14} {logits[idx]:>8.3f} {p:>8.2%}")

        proximo = int(logits.argmax())
        print(f"\n  argmax -> '{c.vocab.itos[proximo]}'   (decoding.py:30)")
        ys = torch.cat([ys, torch.tensor([[proximo]])], dim=1)
        if proximo == EOS_IDX:
            print("  <eos>: linha 33 marca finished, linha 35 dá break.")
            break

    print(f"\n  {'-' * 70}")
    print("\n  ALINHAMENTO COMPLETO (cross-attention de todos os passos):")
    entrada = ys[:, :-1]
    c.model.decode(entrada, c.memoria, c.src_mask,
                   make_target_mask(entrada, c.cfg.pad_idx), need_weights=True)
    cross = ultima.cross_attn.attn_weights[0].mean(0)
    rot_tgt = [c.vocab.itos[i] for i in entrada[0].tolist()]
    matriz("linha = o que foi lido, coluna = origem", cross, rot_tgt, rot_src, fmt="{:6.2f}")

    print(f"\n  Vocab.decode (data.py:57) pula <pad>/<bos>, PARA no <eos> e desfaz o '@@ ':")
    print(f"\n    {ys[0].tolist()}")
    print(f'    -->  "{c.vocab.decode(ys[0].tolist())}"')
    print(f'\n  DE: "{c.frase}"')
    print(f'  EN: "{c.vocab.decode(ys[0].tolist())}"')


PASSOS = [
    (1, "Quebrar a frase em pedaços (BPE)", passo_1),
    (2, "Pedaços viram ids", passo_2),
    (3, "Virar tensor", passo_3),
    (4, "A máscara de padding", passo_4),
    (5, "Embedding × √d_model", passo_5),
    (6, "Codificação posicional", passo_6),
    (7, "A pilha de camadas", passo_7),
    (8, "Dentro da self-attention", passo_8),
    (9, "Add & Norm", passo_9),
    (10, "Feed Forward", passo_10),
    (11, "Repetir e obter a memória", passo_11),
    (12, "O decoder gera pedaço por pedaço", passo_12),
]


def main():
    ap = argparse.ArgumentParser(description="Executa o percurso pelo codigo descrito no README")
    ap.add_argument("--stage", type=int, help="roda só este passo")
    ap.add_argument("--pause", action="store_true", help="abre o pdb antes do passo")
    ap.add_argument("--list", action="store_true", help="lista os passos")
    ap.add_argument("--frase", default=FRASE)
    ap.add_argument("--checkpoint", default="checkpoints/multi30k.pt")
    args = ap.parse_args()

    if args.list:
        print("\nPassos (correspondem aos 'Passo N' do README):\n")
        for n, nome, _ in PASSOS:
            print(f"  {n:>2}  {nome}")
        print("\n  python walkthrough.py --stage N\n")
        return

    torch.manual_seed(0)
    c = Contexto(args.checkpoint, args.frase)

    if args.stage:
        alvo = dict((n, fn) for n, _, fn in PASSOS).get(args.stage)
        if alvo is None:
            print(f"passo {args.stage} não existe; use --list")
            return
        for n, _, fn in PASSOS:            # dependências, em silêncio
            if n < args.stage:
                with contextlib.redirect_stdout(io.StringIO()):
                    fn(c)
        if args.pause:
            print("\npdb: `n` avança, `s` entra, `p nome` imprime, `c` continua, `q` sai.\n")
            breakpoint()
        alvo(c)
    else:
        for _, _, fn in PASSOS:
            fn(c)

    print(f"\n{'=' * 78}")
    print("Fim. O percurso comentado, com estes mesmos valores, está no README.")
    print("=" * 78)


if __name__ == "__main__":
    main()
