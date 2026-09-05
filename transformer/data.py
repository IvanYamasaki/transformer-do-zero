"""Dados: vocabulario BPE compartilhado, download do Multi30k e batches por tokens.

Segue a secao 5.1 do paper:
  - BPE com vocabulario compartilhado entre origem e destino
    (o paper usa ~37k tokens no WMT EN-DE; o Multi30k e pequeno e comporta ~10k)
  - batches montados por NUMERO DE TOKENS, agrupando frases de tamanho parecido
    (o paper usa ~25k tokens de origem + 25k de destino por batch)
"""

import gzip
import random
import urllib.request
from collections import Counter
from pathlib import Path

import torch

from .bpe import BPE, learn_bpe, pretokenize

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
PAD_IDX, BOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3

MULTI30K_URL = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw"
SPLITS = {"train": "train", "valid": "val", "test": "test_2016_flickr"}


def tokenize(text):
    """Separa palavras e pontuacao, preservando maiusculas (como no paper)."""
    return pretokenize(text)


class Vocab:
    """Token <-> id, com BPE opcional e os 4 tokens especiais em posicoes fixas."""

    def __init__(self, itos=None, merges=None):
        self.itos = list(itos) if itos is not None else [PAD, BOS, EOS, UNK]
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        self.bpe = BPE(merges) if merges else None

    def __len__(self):
        return len(self.itos)

    # -- texto <-> pedacos -------------------------------------------------- #

    def tokenize(self, text):
        return self.bpe.encode(text) if self.bpe else tokenize(text)

    def detokenize(self, pieces):
        return BPE.decode(pieces) if self.bpe else " ".join(pieces)

    # -- texto <-> ids ------------------------------------------------------- #

    def encode(self, text, add_special=True):
        ids = [self.stoi.get(tok, UNK_IDX) for tok in self.tokenize(text)]
        return [BOS_IDX] + ids + [EOS_IDX] if add_special else ids

    def decode(self, ids, strip_special=True):
        pieces = []
        for i in ids:
            i = int(i)
            if strip_special and i in (PAD_IDX, BOS_IDX):
                continue
            if strip_special and i == EOS_IDX:
                break
            pieces.append(self.itos[i] if i < len(self.itos) else UNK)
        return self.detokenize(pieces)

    # -- construcao e serializacao ----------------------------------------- #

    @classmethod
    def build(cls, sentences, bpe_merges=10000, min_freq=1):
        """Aprende o BPE nas frases (origem E destino juntas) e monta o vocabulario.

        bpe_merges=0 desliga o BPE e volta para tokens por palavra.
        """
        merges = learn_bpe(sentences, bpe_merges) if bpe_merges else None
        vocab = cls(merges=merges)
        counter = Counter()
        for sentence in sentences:
            counter.update(vocab.tokenize(sentence))
        for token, freq in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
            if freq < min_freq:
                break
            vocab.itos.append(token)
        vocab.stoi = {tok: i for i, tok in enumerate(vocab.itos)}
        return vocab

    def to_dict(self):
        return {"itos": self.itos, "merges": self.bpe.merges if self.bpe else None}

    @classmethod
    def from_dict(cls, data):
        return cls(itos=data["itos"], merges=data.get("merges"))


def download_multi30k(data_dir="data", splits=("train", "valid", "test"), langs=("de", "en")):
    """Baixa e descompacta os arquivos do Multi30k. Retorna {(split, lang): [frases]}."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    corpus = {}

    for split in splits:
        remote = SPLITS[split]
        for lang in langs:
            local = data_dir / f"{split}.{lang}"
            if not local.exists():
                url = f"{MULTI30K_URL}/{remote}.{lang}.gz"
                print(f"  baixando {url}")
                gz_path = data_dir / f"{remote}.{lang}.gz"
                urllib.request.urlretrieve(url, gz_path)
                with gzip.open(gz_path, "rt", encoding="utf-8") as fh:
                    local.write_text(fh.read(), encoding="utf-8")
                gz_path.unlink()
            lines = [ln.strip() for ln in local.read_text(encoding="utf-8").splitlines() if ln.strip()]
            corpus[(split, lang)] = lines
    return corpus


class TranslationDataset(torch.utils.data.Dataset):
    """Pares (src_ids, tgt_ids) ja com <bos>/<eos>, filtrados por comprimento."""

    def __init__(self, src_sentences, tgt_sentences, vocab, max_len=100):
        self.pairs = []
        for src, tgt in zip(src_sentences, tgt_sentences):
            s = vocab.encode(src)
            t = vocab.encode(tgt)
            if 2 < len(s) <= max_len and 2 < len(t) <= max_len:
                self.pairs.append((torch.tensor(s), torch.tensor(t)))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]

    def lengths(self):
        return [max(len(s), len(t)) for s, t in self.pairs]


class TokenBatchSampler(torch.utils.data.Sampler):
    """Batches com ate `max_tokens` tokens (contando o padding), como no paper.

    Ordena as frases por comprimento para que cada batch tenha frases parecidas
    (menos padding desperdicado) e embaralha a ORDEM dos batches a cada epoca.
    """

    def __init__(self, lengths, max_tokens, shuffle=True, seed=0):
        self.lengths = list(lengths)
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.rng = random.Random(seed)
        self.batches = self._make_batches()

    def _make_batches(self):
        order = list(range(len(self.lengths)))
        if self.shuffle:  # ruido no desempate: frases do mesmo tamanho mudam de batch
            self.rng.shuffle(order)
        order.sort(key=lambda i: self.lengths[i])

        batches, current, longest = [], [], 0
        for i in order:
            longest_if_added = max(longest, self.lengths[i])
            if current and longest_if_added * (len(current) + 1) > self.max_tokens:
                batches.append(current)
                current, longest = [], 0
                longest_if_added = self.lengths[i]
            current.append(i)
            longest = longest_if_added
        if current:
            batches.append(current)
        return batches

    def __iter__(self):
        if self.shuffle:
            self.batches = self._make_batches()
            self.rng.shuffle(self.batches)
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def collate_batch(batch, pad_idx=PAD_IDX):
    """Padding a direita -> (src, tgt_in, tgt_out).

    tgt_in  = tgt[:-1]  (entrada do decoder, comeca com <bos>)
    tgt_out = tgt[1:]   (alvo deslocado, termina com <eos>)
    """
    srcs, tgts = zip(*batch)
    src = torch.nn.utils.rnn.pad_sequence(srcs, batch_first=True, padding_value=pad_idx)
    tgt = torch.nn.utils.rnn.pad_sequence(tgts, batch_first=True, padding_value=pad_idx)
    return src, tgt[:, :-1], tgt[:, 1:]
