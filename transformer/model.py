"""O Transformer completo: encoder-decoder de "Attention Is All You Need".

Hiperparametros do modelo base (Table 3 do paper):
    N = 6 camadas, d_model = 512, d_ff = 2048, h = 8, P_drop = 0.1
"""

from dataclasses import asdict, dataclass, fields

import torch
import torch.nn as nn

from .embeddings import LearnedPositionalEmbedding, PositionalEncoding, TokenEmbedding
from .layers import DecoderLayer, EncoderLayer, clones
from .masking import make_source_mask, make_target_mask


@dataclass
class TransformerConfig:
    src_vocab_size: int
    tgt_vocab_size: int
    num_layers: int = 6          # N
    d_model: int = 512
    num_heads: int = 8           # h
    d_ff: int = 2048
    dropout: float = 0.1         # P_drop
    max_len: int = 5000
    pad_idx: int = 0
    share_embeddings: bool = True   # amarra embeddings src/tgt (exige vocabulario unico)
    tie_generator: bool = True      # amarra a projecao pre-softmax ao embedding (secao 3.4)
    positional: str = "sinusoidal"  # "sinusoidal" (secao 3.5) ou "learned" (Table 3, linha E)
    attention: str = "math"         # "math" (paper), "flash" ou "sdpa" (ver transformer/flash.py)

    @classmethod
    def base(cls, src_vocab_size, tgt_vocab_size, **kwargs):
        return cls(src_vocab_size, tgt_vocab_size, **kwargs)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        """Reconstroi a config a partir de um dict OU de uma config antiga.

        Chaves desconhecidas sao ignoradas e campos ausentes ficam no default,
        entao checkpoints salvos por versoes anteriores continuam carregando.
        """
        if not isinstance(data, dict):
            data = {f.name: getattr(data, f.name) for f in fields(cls) if hasattr(data, f.name)}
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def big(cls, src_vocab_size, tgt_vocab_size, **kwargs):
        """Variante "big" do paper: 213M parametros, BLEU 28.4 em EN-DE."""
        defaults = dict(num_layers=6, d_model=1024, num_heads=16, d_ff=4096, dropout=0.3)
        defaults.update(kwargs)
        return cls(src_vocab_size, tgt_vocab_size, **defaults)


class Encoder(nn.Module):
    """Pilha de N camadas identicas de encoder."""

    def __init__(self, layer, n):
        super().__init__()
        self.layers = clones(layer, n)

    def forward(self, x, mask=None, need_weights=False):
        for layer in self.layers:
            x = layer(x, mask, need_weights)
        return x


class Decoder(nn.Module):
    """Pilha de N camadas identicas de decoder."""

    def __init__(self, layer, n):
        super().__init__()
        self.layers = clones(layer, n)

    def forward(self, x, memory, src_mask=None, tgt_mask=None, need_weights=False):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask, need_weights)
        return x


class Transformer(nn.Module):
    """Arquitetura encoder-decoder da Figura 1 do paper.

    forward() devolve logits (B, L_tgt, V). Aplique softmax/log_softmax por fora
    ou use `log_probs()` — a loss de treino ja espera logits.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        cfg = config

        self.src_embed = TokenEmbedding(cfg.src_vocab_size, cfg.d_model, cfg.pad_idx)
        if cfg.share_embeddings:
            if cfg.src_vocab_size != cfg.tgt_vocab_size:
                raise ValueError(
                    "share_embeddings=True exige vocabularios de mesmo tamanho "
                    f"(src={cfg.src_vocab_size}, tgt={cfg.tgt_vocab_size})"
                )
            self.tgt_embed = self.src_embed
        else:
            self.tgt_embed = TokenEmbedding(cfg.tgt_vocab_size, cfg.d_model, cfg.pad_idx)

        if cfg.positional == "sinusoidal":
            self.pos_encoding = PositionalEncoding(cfg.d_model, cfg.dropout, cfg.max_len)
        elif cfg.positional == "learned":
            self.pos_encoding = LearnedPositionalEmbedding(cfg.d_model, cfg.dropout, cfg.max_len)
        else:
            raise ValueError(f"positional deve ser 'sinusoidal' ou 'learned', nao {cfg.positional!r}")

        enc_layer = EncoderLayer(cfg.d_model, cfg.num_heads, cfg.d_ff, cfg.dropout,
                                 impl=cfg.attention)
        dec_layer = DecoderLayer(cfg.d_model, cfg.num_heads, cfg.d_ff, cfg.dropout,
                                 impl=cfg.attention)
        self.encoder = Encoder(enc_layer, cfg.num_layers)
        self.decoder = Decoder(dec_layer, cfg.num_layers)

        self.generator = nn.Linear(cfg.d_model, cfg.tgt_vocab_size, bias=False)
        if cfg.tie_generator:
            self.generator.weight = self.tgt_embed.lut.weight

        self._reset_parameters()

    def _reset_parameters(self):
        """Xavier uniforme nas projecoes; saidas dos sublayers escaladas por 1/sqrt(2N).

        O paper nao especifica a inicializacao. Xavier puro tem um problema com
        post-LN profundo: cada camada mistura as posicoes pela atencao e o
        LayerNorm normaliza a soma, e com N=6 a distancia media entre posicoes
        na saida do encoder ja e so ~16% da entrada ANTES de treinar. Num corpus
        pequeno o treino cai num encoder "saco de palavras" com cross-attention
        uniforme. Escalar W^O (atencao) e W_2 (FFN) por 1/sqrt(2N) deixa a razao
        em ~80% e o modelo aprende alinhamentos (ver test_encoder_keeps_positions
        _distinct_at_init).
        """
        for name, p in self.named_parameters():
            # Embeddings (tokens e posicoes aprendidas) tem inicializacao propria abaixo
            if p.dim() > 1 and "lut" not in name and "pos_encoding" not in name:
                nn.init.xavier_uniform_(p)

        scale = (2 * self.config.num_layers) ** -0.5
        with torch.no_grad():
            for layer in list(self.encoder.layers) + list(self.decoder.layers):
                layer.self_attn.w_o.weight.mul_(scale)
                if hasattr(layer, "cross_attn"):
                    layer.cross_attn.w_o.weight.mul_(scale)
                layer.feed_forward.w_2.weight.mul_(scale)
        # Embeddings: N(0, d_model^-0.5) mantem a escala apos o *sqrt(d_model)
        std = self.config.d_model ** -0.5
        nn.init.normal_(self.src_embed.lut.weight, mean=0.0, std=std)
        if self.tgt_embed is not self.src_embed:
            nn.init.normal_(self.tgt_embed.lut.weight, mean=0.0, std=std)
        if self.config.pad_idx is not None:
            with torch.no_grad():
                self.src_embed.lut.weight[self.config.pad_idx].zero_()
                self.tgt_embed.lut.weight[self.config.pad_idx].zero_()

    # ------------------------------------------------------------------ #

    def encode(self, src, src_mask=None, need_weights=False):
        """src: (B, L_src) de ids -> memoria (B, L_src, d_model)."""
        if src_mask is None:
            src_mask = make_source_mask(src, self.config.pad_idx)
        x = self.pos_encoding(self.src_embed(src))
        return self.encoder(x, src_mask, need_weights)

    def decode(self, tgt, memory, src_mask=None, tgt_mask=None, need_weights=False):
        """tgt: (B, L_tgt) de ids -> estados (B, L_tgt, d_model)."""
        if tgt_mask is None:
            tgt_mask = make_target_mask(tgt, self.config.pad_idx)
        x = self.pos_encoding(self.tgt_embed(tgt))
        return self.decoder(x, memory, src_mask, tgt_mask, need_weights)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, need_weights=False):
        """Teacher forcing: `tgt` ja deve vir deslocado (comeca com <bos>).

        Retorna logits (B, L_tgt, tgt_vocab_size).
        """
        if src_mask is None:
            src_mask = make_source_mask(src, self.config.pad_idx)
        if tgt_mask is None:
            tgt_mask = make_target_mask(tgt, self.config.pad_idx)

        memory = self.encode(src, src_mask, need_weights)
        hidden = self.decode(tgt, memory, src_mask, tgt_mask, need_weights)
        return self.generator(hidden)

    def log_probs(self, src, tgt, **kwargs):
        return torch.log_softmax(self.forward(src, tgt, **kwargs), dim=-1)

    # ------------------------------------------------------------------ #

    def num_parameters(self, trainable_only=True):
        params = self.parameters()
        if trainable_only:
            params = (p for p in params if p.requires_grad)
        # Pesos amarrados aparecem uma vez so
        seen, total = set(), 0
        for p in params:
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
        return total

    def attention_maps(self):
        """Ultimos pesos de atencao capturados (precisa de need_weights=True no forward).

        Retorna um dict {nome_da_camada: (B, h, L_q, L_k)}.
        """
        maps = {}
        for i, layer in enumerate(self.encoder.layers):
            if layer.self_attn.attn_weights is not None:
                maps[f"encoder.{i}.self_attn"] = layer.self_attn.attn_weights
        for i, layer in enumerate(self.decoder.layers):
            if layer.self_attn.attn_weights is not None:
                maps[f"decoder.{i}.self_attn"] = layer.self_attn.attn_weights
            if layer.cross_attn.attn_weights is not None:
                maps[f"decoder.{i}.cross_attn"] = layer.cross_attn.attn_weights
        return maps


def make_model(src_vocab_size, tgt_vocab_size, n=6, d_model=512, d_ff=2048,
               h=8, dropout=0.1, **kwargs):
    """Atalho no estilo da implementacao de referencia (The Annotated Transformer)."""
    cfg = TransformerConfig(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        num_layers=n,
        d_model=d_model,
        d_ff=d_ff,
        num_heads=h,
        dropout=dropout,
        **kwargs,
    )
    return Transformer(cfg)
