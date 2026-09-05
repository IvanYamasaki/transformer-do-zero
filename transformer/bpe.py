"""Byte-Pair Encoding (Sennrich et al., 2016), como usado na secao 5.1 do paper.

O paper codifica as frases com BPE e usa um vocabulario COMPARTILHADO entre
origem e destino (~37 000 tokens para EN-DE). Vocabulario unico e o que permite
amarrar os tres pesos da secao 3.4: embedding de origem, de destino e a
projecao pre-softmax.

Convencao de saida (a mesma do subword-nmt): um pedaco que NAO termina a
palavra recebe o sufixo "@@".

    "Schneemobilen"  ->  ["Schnee@@", "mobil@@", "en"]

Implementacao pura em Python, sem dependencias. Aprender 10k merges no
Multi30k leva alguns segundos.
"""

import heapq
import re
from collections import Counter, defaultdict

END = "</w>"          # marcador interno de fim de palavra
CONTINUE = "@@"       # sufixo de pedaco que continua na proxima peca

_PRETOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def pretokenize(text):
    """Separa palavras e pontuacao. Preserva maiusculas, como no paper."""
    return _PRETOKEN_RE.findall(text)


# --------------------------------------------------------------------------- #
# Aprendizado dos merges
# --------------------------------------------------------------------------- #

def learn_bpe(sentences, num_merges, min_pair_freq=2):
    """Aprende `num_merges` regras de fusao por frequencia de pares adjacentes.

    Algoritmo do paper de Sennrich: comeca com cada palavra como sequencia de
    caracteres (mais </w> no ultimo) e, repetidamente, funde o par de simbolos
    adjacentes mais frequente do corpus.

    Retorna a lista de merges em ordem de prioridade: [(a, b), ...].
    """
    word_freq = Counter()
    for sentence in sentences:
        word_freq.update(pretokenize(sentence))

    words = []   # cada palavra como tupla de simbolos
    freqs = []
    for word, freq in word_freq.items():
        words.append(tuple(word[:-1]) + (word[-1] + END,))
        freqs.append(freq)

    # Estatisticas de pares + indice de quais palavras contem cada par.
    stats = Counter()
    where = defaultdict(Counter)
    for i, (word, freq) in enumerate(zip(words, freqs)):
        for pair in zip(word, word[1:]):
            stats[pair] += freq
            where[pair][i] += 1

    # Heap de maximo com invalidacao preguicosa: entradas velhas sao ignoradas
    # quando a frequencia guardada nao bate com a atual.
    heap = [(-freq, pair) for pair, freq in stats.items()]
    heapq.heapify(heap)

    merges = []
    while len(merges) < num_merges and heap:
        neg_freq, best = heapq.heappop(heap)
        if stats.get(best, 0) != -neg_freq:
            continue                      # entrada obsoleta
        if -neg_freq < min_pair_freq:
            break
        merges.append(best)
        new_symbol = best[0] + best[1]

        for i in list(where[best]):
            word, freq = words[i], freqs[i]
            for pair in zip(word, word[1:]):        # retira as contagens antigas
                stats[pair] -= freq
                where[pair][i] -= 1
                if where[pair][i] == 0:
                    del where[pair][i]
            word = _merge_word(word, best, new_symbol)
            words[i] = word
            for pair in zip(word, word[1:]):        # poe as novas
                stats[pair] += freq
                where[pair][i] += 1
                heapq.heappush(heap, (-stats[pair], pair))

        del stats[best]
        del where[best]

    return merges


def _merge_word(word, pair, new_symbol):
    out, i = [], 0
    while i < len(word):
        if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
            out.append(new_symbol)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


# --------------------------------------------------------------------------- #
# Aplicacao
# --------------------------------------------------------------------------- #

class BPE:
    """Segmenta texto com um conjunto de merges aprendido por `learn_bpe`."""

    def __init__(self, merges):
        self.merges = [tuple(m) for m in merges]
        self.ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self._cache = {}

    def segment_word(self, word):
        """'Schneemobilen' -> ['Schnee@@', 'mobil@@', 'en']"""
        if word in self._cache:
            return self._cache[word]

        symbols = list(word[:-1]) + [word[-1] + END]
        while len(symbols) > 1:
            # Funde sempre o par de MENOR rank (aprendido mais cedo = mais frequente)
            best_rank, best_i = None, None
            for i, pair in enumerate(zip(symbols, symbols[1:])):
                rank = self.ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_i = rank, i
            if best_i is None:
                break
            symbols[best_i:best_i + 2] = [symbols[best_i] + symbols[best_i + 1]]

        pieces = [s[:-len(END)] if s.endswith(END) else s + CONTINUE for s in symbols]
        self._cache[word] = pieces
        return pieces

    def encode(self, text):
        """Texto -> lista de pedacos."""
        pieces = []
        for word in pretokenize(text):
            pieces.extend(self.segment_word(word))
        return pieces

    @staticmethod
    def decode(pieces):
        """Lista de pedacos -> texto (desfaz o '@@ ')."""
        return " ".join(pieces).replace(CONTINUE + " ", "").removesuffix(CONTINUE)

    def __len__(self):
        return len(self.merges)
