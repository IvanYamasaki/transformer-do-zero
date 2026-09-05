"""Implementacao em PyTorch do Transformer de "Attention Is All You Need".

Vaswani et al., NeurIPS 2017 — https://arxiv.org/abs/1706.03762

Uso rapido:
    from transformer import Transformer, TransformerConfig
    model = Transformer(TransformerConfig(src_vocab_size=32000, tgt_vocab_size=32000))
    logits = model(src, tgt_in)          # (B, L, V)
"""

from .attention import MultiHeadAttention, scaled_dot_product_attention
from .bpe import BPE, learn_bpe
from .decoding import beam_search, greedy_decode
from .embeddings import LearnedPositionalEmbedding, PositionalEncoding, TokenEmbedding
from .layers import DecoderLayer, EncoderLayer, PositionwiseFeedForward, ResidualConnection
from .masking import make_source_mask, make_target_mask, padding_mask, subsequent_mask
from .model import Decoder, Encoder, Transformer, TransformerConfig, make_model
from .optim import LabelSmoothingLoss, NoamLR, make_optimizer, noam_lambda
from .utils import average_checkpoints, count_parameters

__version__ = "2.0.0"

__all__ = [
    "BPE",
    "learn_bpe",
    "MultiHeadAttention",
    "scaled_dot_product_attention",
    "TokenEmbedding",
    "PositionalEncoding",
    "LearnedPositionalEmbedding",
    "PositionwiseFeedForward",
    "ResidualConnection",
    "EncoderLayer",
    "DecoderLayer",
    "Encoder",
    "Decoder",
    "Transformer",
    "TransformerConfig",
    "make_model",
    "padding_mask",
    "subsequent_mask",
    "make_source_mask",
    "make_target_mask",
    "LabelSmoothingLoss",
    "NoamLR",
    "noam_lambda",
    "make_optimizer",
    "greedy_decode",
    "beam_search",
    "average_checkpoints",
    "count_parameters",
]
