# Transformer do zero

Repositório de estudo sobre **"Attention Is All You Need"** (Vaswani et al.,
NeurIPS 2017 — [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)).

A arquitetura está reimplementada em PyTorch do zero, sem usar `nn.Transformer`
nem `nn.MultiheadAttention`: cada peça descrita no artigo está escrita
explicitamente, para poder ser lida, testada e modificada. O objetivo é ser
**fiel ao artigo**, não apenas parecido — modelo base da Table 3 (N=6,
d_model=512, h=8, d_ff=2048), post-LN, projeções de atenção sem bias como nas
equações, BPE com vocabulário compartilhado e os três pesos amarrados (§3.4),
batches por número de tokens, Adam(0.9, 0.98, 10⁻⁹) com warmup de 4000 passos,
label smoothing 0.1, beam 4 com α=0.6 e média dos últimos checkpoints (§6.1).

Não é só arquitetura no papel: o modelo treina de verdade em dados reais
(Multi30k alemão→inglês, baixado automaticamente) e chega a BLEU 41,15. Uma
bateria de 38 testes verifica as propriedades da arquitetura — causalidade,
invariância a padding, as fórmulas do artigo — e não apenas formatos de tensor,
incluindo equivalência numérica com o `nn.MultiheadAttention` do próprio PyTorch.

**Por onde começar**

| Se você quer | Vá para |
|---|---|
| Entender a arquitetura em alto nível, com diagramas | [`relatorio/transformer.pdf`](relatorio/transformer.pdf) |
| Ver como cada etapa é feita no código, linha por linha | [O código, passo a passo](#o-código-passo-a-passo), mais abaixo |
| Rodar e mexer | [Começando](#começando) |

O percurso pelo código segue uma única frase, `"Ein Mann geht."`, do texto cru
até a tradução `"A man is walking ."`, mostrando os valores reais produzidos pelo
modelo treinado em cada etapa.

---

## Começando

```bash
pip install torch                    # única dependência obrigatória

python demo.py                       # inspeciona a arquitetura (segundos, CPU)
python walkthrough.py                # percorre o código passo a passo com valores reais
python tests/test_transformer.py     # 38 testes
python train_copy.py                 # tarefa de cópia, ~20s, prova que treina
python train_multi30k.py             # modelo base do paper, DE→EN, ~30 min numa GPU de 6 GB
python translate.py                  # traduz usando o checkpoint treinado
```

## O que tem aqui

```text
transformer/          o pacote — cada arquivo é uma parte do artigo
  attention.py        Scaled Dot-Product Attention + Multi-Head Attention (§3.2)
  embeddings.py       Embedding × √d_model + posição senoidal ou aprendida (§3.4, §3.5)
  layers.py           FFN posicional, residual + LayerNorm, camadas enc/dec (§3.1, §3.3)
  masking.py          Máscara de padding e máscara causal (§3.2.3)
  model.py            Encoder, Decoder, Transformer completo, TransformerConfig
  bpe.py              Byte-Pair Encoding do zero, vocabulário compartilhado (§5.1)
  data.py             Vocabulário BPE, batches por tokens (§5.1), download do Multi30k
  optim.py            Noam LR schedule (eq. 3) e label smoothing (§5.4)
  decoding.py         Decodificação greedy e beam search com penalidade (§6.1)
  utils.py            Média de checkpoints (§6.1) e contagem de parâmetros

walkthrough.py        Executa o percurso deste README imprimindo os tensores reais
demo.py               Inspeção: parâmetros, máscaras, causalidade, atenção, schedule
train_copy.py         Tarefa de cópia (sanity check completo do pipeline, ~20 s)
train_multi30k.py     Treino do modelo base com a receita do §5 + avaliação BLEU
translate.py          Tradução por linha de comando ou interativa
tests/                Bateria de 38 testes
relatorio/            Relatório em PDF explicando a arquitetura em alto nível
```

## Correspondência com o paper

| Elemento do paper | Onde está |
|---|---|
| eq. 1 — `softmax(QKᵀ/√d_k)V` | [attention.py:13](transformer/attention.py#L13) |
| §3.2.2 — Multi-Head Attention, h=8, sem bias nas projeções | [attention.py:37](transformer/attention.py#L37) |
| eq. 2 — `FFN(x) = max(0, xW₁+b₁)W₂+b₂` | [layers.py:15](transformer/layers.py#L15) |
| §3.1 — `LayerNorm(x + Sublayer(x))` | [layers.py:28](transformer/layers.py#L28) |
| §3.2.3 — máscara causal no decoder | [masking.py:14](transformer/masking.py#L14) |
| §3.4 — embeddings × √d_model | [embeddings.py:9](transformer/embeddings.py#L9) |
| §3.4 — os três pesos amarrados | [model.py:122](transformer/model.py#L122) |
| §3.5 — codificação posicional senoidal | [embeddings.py:21](transformer/embeddings.py#L21) |
| §5.1 — BPE com vocabulário compartilhado | [bpe.py:36](transformer/bpe.py#L36), [data.py:71](transformer/data.py#L71) |
| §5.1 — batches por número de tokens | [data.py:140](transformer/data.py#L140) |
| §5.3 — Adam β₁=0.9, β₂=0.98, ε=10⁻⁹ | [optim.py:31](transformer/optim.py#L31) |
| eq. 3 — Noam warmup | [optim.py:9](transformer/optim.py#L9) |
| §5.4 — label smoothing ε=0.1 | [optim.py:37](transformer/optim.py#L37) |
| §6.1 — beam 4, α=0.6 | [decoding.py:40](transformer/decoding.py#L40) |
| §6.1 — comprimento máximo = entrada + 50 | [translate.py:35](translate.py#L35) |
| §6.1 — média dos últimos checkpoints | [utils.py:13](transformer/utils.py#L13) |
| Table 3 linha E — posições aprendidas | [embeddings.py:49](transformer/embeddings.py#L49) |

## Hiperparâmetros

| | base | big |
|---|---|---|
| N (camadas) | 6 | 6 |
| d_model | 512 | 1024 |
| d_ff | 2048 | 4096 |
| h (cabeças) | 8 | 16 |
| d_k = d_v | 64 | 64 |
| dropout | 0.1 | 0.3 |
| parâmetros (vocab 37k) | **63.05M** (paper: 65M) | **214.2M** (paper: 213M) |

Otimizador: Adam com β₁=0.9, β₂=0.98, ε=10⁻⁹ e o schedule da eq. 3 (warmup 4000).
A diferença de ~2M para os 65M do paper está no vocabulário exato e em detalhes
de contagem que o paper não especifica.

## Usando o modelo

```python
import torch
from transformer import Transformer, TransformerConfig, greedy_decode

model = Transformer(TransformerConfig(
    src_vocab_size=32000,
    tgt_vocab_size=32000,
    num_layers=6, d_model=512, num_heads=8, d_ff=2048, dropout=0.1,
))

src    = torch.randint(4, 32000, (8, 20))   # (batch, comprimento)
tgt_in = torch.randint(4, 32000, (8, 15))   # entrada do decoder (teacher forcing)

logits = model(src, tgt_in)                 # (8, 15, 32000)

# inferência
saida = greedy_decode(model, src, max_len=50, bos_idx=1, eos_idx=2)
```

As máscaras são construídas automaticamente a partir de `pad_idx` — não é preciso
passá-las à mão. Convenção: `True` = posição visível.

### Treino

```python
from transformer import LabelSmoothingLoss, make_optimizer

loss_fn = LabelSmoothingLoss(vocab_size=32000, pad_idx=0, smoothing=0.1)
optimizer, scheduler = make_optimizer(model, d_model=512, warmup=4000)

logits = model(src, tgt[:, :-1])       # entrada do decoder
loss = loss_fn(logits, tgt[:, 1:])     # alvo deslocado em uma posição
loss.backward()
optimizer.step()
scheduler.step()                       # o schedule avança por passo, não por época
```

### Texto ↔ ids com BPE

```python
from transformer.data import Vocab

vocab = Vocab.build(frases_de + frases_en, bpe_merges=10000)   # um vocabulário só
ids   = vocab.encode("Ein Mann geht.")     # [1, ..., 2]  com <bos>/<eos>
texto = vocab.decode(ids)                  # desfaz o "@@ " dos pedaços
```

### Visualizando a atenção

```python
model(src, tgt, need_weights=True)
maps = model.attention_maps()          # {'encoder.0.self_attn': (B, h, L_q, L_k), ...}
```

## Resultados verificados

**Tarefa de cópia** (`python train_copy.py`) — 600 passos, ~20s: 5/5 sequências
de 10 tokens reproduzidas exatamente com decodificação greedy.

**Multi30k DE→EN** (`python train_multi30k.py`) — 29 000 pares de treino, modelo
base do paper (N=6, d_model=512, h=8, d_ff=2048), 49.1M parâmetros, BPE de
10 000 merges com vocabulário compartilhado de 9 740 tokens, 40 épocas em ~27 min
numa RTX 3050 de 6 GB (bfloat16):

```text
epoca  1/40  treino 7.855  valid 6.050  ppl 424.2
epoca  6/40  treino 3.379  valid 3.282  ppl  26.6
epoca 13/40  treino 2.188  valid 2.858  ppl  17.4   <- melhor
epoca 40/40  treino 1.475  valid 3.012  ppl  20.3   (overfitting)

media dos checkpoints 9-13 (secao 6.1):  valid 2.858 -> 2.803
BLEU-4 em 300 frases de teste: 41.15     (41.32 em minusculas; zero <unk>)
```

A média de checkpoints baixa a loss de validação em 0.055 — a ordem de grandeza
dos "alguns décimos de BLEU" do paper. O log completo fica em
`checkpoints/train_log.txt` (gerado pelo treino).

Traduções do conjunto de teste (nenhuma vista no treino):

| Alemão | Referência | Gerado |
|---|---|---|
| Leute Reparieren das Dach eines Hauses. | People are fixing the roof of a house. | People are fixing the roof of a house . |
| Fünf Leute in Winterjacken und mit Helmen stehen im Schnee mit Schneemobilen im Hintergrund. | Five people wearing winter jackets and helmets stand in the snow, with snowmobiles in the background. | Five people in winter jackets and helmets are standing in the snow with snowboards in the background . |
| Ein Boston Terrier läuft über saftig-grünes Gras vor einem weißen Zaun. | A Boston Terrier is running on lush green grass in front of a white fence. | A terrier runs across the grass in front of a green and white fence . |

"Schneemobilen" vira `Schneem@@ ob@@ il@@ en` no BPE e sai como "snowboards":
errado, mas é uma palavra de verdade e não um `<unk>` — é exatamente o que o
BPE compra. Com 29k pares o modelo base começa a decorar na época 13 (49M
parâmetros para 466k tokens de treino); a receita do §5 foi desenhada para 4.5M
pares.

> O BLEU acima é calculado pela função em `train_multi30k.py`, com suavização
> add-1 e tokenização própria. Serve para acompanhar o treino, mas **não é
> comparável** a números de sacreBLEU publicados na literatura. E o Multi30k
> (29k pares) é ~150× menor que o WMT14 do paper (4.5M pares): a arquitetura é a
> mesma, o resultado não é.

## Variantes disponíveis

Tudo é chaveado por `TransformerConfig`:

```python
TransformerConfig(
    src_vocab_size=32000, tgt_vocab_size=32000,
    positional="learned",     # "sinusoidal" (padrão, §3.5) ou "learned" (Table 3, linha E)
    share_embeddings=False,   # vocabulários de origem e destino separados
    tie_generator=False,      # projeção pré-softmax com pesos próprios
)

TransformerConfig.big(37000, 37000)   # d_model 1024, h 16, d_ff 4096, dropout 0.3
```

Média dos últimos checkpoints, como no §6.1 (vale alguns décimos de BLEU):

```python
from transformer import average_checkpoints
model.load_state_dict(average_checkpoints(["ep21.pt", "ep22.pt", "ep23.pt", "ep24.pt", "ep25.pt"]))
```

## Decisões de implementação

**Post-LN, e só post-LN.** `LayerNorm(x + Sublayer(x))`, exatamente como o paper.
A variante pre-norm (que treina sem warmup) é posterior ao paper e não está aqui
de propósito — é o motivo de o warmup do Noam ser obrigatório.

**Sem bias nas projeções de atenção.** As equações do §3.2.2 são `QWᵢ^Q`, `KWᵢ^K`,
`VWᵢ^V` e `Concat(…)W^O` — produtos puros. A FFN (eq. 2) tem `b₁` e `b₂`, e os tem
aqui também.

**BPE do zero, vocabulário único.** `bpe.py` implementa o algoritmo de Sennrich
et al. (2016), que o paper usa: aprende fusões por frequência de pares e segmenta
com o sufixo `@@` nos pedaços que continuam. Origem e destino compartilham o
vocabulário, o que é o pré-requisito para amarrar os três pesos do §3.4.
Maiúsculas são preservadas, como no paper.

**Batches por tokens.** `TokenBatchSampler` agrupa frases de tamanho parecido e
fecha o batch por orçamento de tokens (contando padding), como o paper faz com
~25k tokens. Numa GPU de 6 GB o padrão é 2 500.

**Máscara com `finfo.min` em vez de `-inf`.** Uma linha inteiramente mascarada
(acontece com sequências 100% padding num batch) gera `NaN` no softmax se o
preenchimento for `-inf`. O valor finito mínimo do dtype dá o mesmo resultado
prático sem o `NaN`.

**Projeções das cabeças fundidas.** As h projeções por cabeça são feitas por uma
única matriz `d_model × d_model` e depois reorganizadas — matematicamente
idêntico ao paper e bem mais rápido. Há um teste que compara essa versão com h
atenções calculadas separadamente.

**Pesos amarrados.** Embedding da origem, do destino e a projeção pré-softmax
são o mesmo tensor (§3.4). `train_multi30k.py` usa isso; desligue com
`share_embeddings=False` / `tie_generator=False` só se os vocabulários forem
diferentes.

**Média de checkpoints ancorada na melhor época.** O paper promedia os últimos 5
checkpoints de um treino de duração fixa. Com 29k pares o modelo base começa a
decorar, então o "fim" do treino é a melhor época de validação e a janela de 5
que termina nela é a promediada.

**Inicialização.** Xavier uniforme nas matrizes de projeção; embeddings com
`N(0, d_model^-0.5)`, que mantém a escala depois da multiplicação por `√d_model`;
e as projeções de **saída** de cada sublayer (`W^O` da atenção, `W_2` da FFN)
escaladas por `1/√(2N)`. O paper não especifica a inicialização, e essa escala
foi necessária: com Xavier puro, 6 camadas post-LN reduzem a distância média
entre as posições da saída do encoder a **16% da entrada antes de qualquer
treino** (3 camadas: 47%; 12 camadas: 2%), porque cada camada mistura posições
pela atenção e o LayerNorm renormaliza a soma. No Multi30k o treino então
converge para um encoder "saco de palavras", com cross-attention uniforme em
todas as camadas — o modelo ainda traduz (BLEU 29), mas sem alinhamento. Com a
escala a razão na inicialização sobe para 80%, o encoder não colapsa e a
validação na época 6 cai de 4.09 para 3.28. Há um teste que protege isso.

**Decodificação sem cache de KV.** `greedy_decode` e `beam_search` reprocessam o
prefixo inteiro a cada passo — O(L²) de trabalho em vez de O(L). Mantém o código
legível; para inferência em produção, um cache de chaves/valores por camada seria
o próximo passo.

**Codificação senoidal por padrão.** O paper testou posições aprendidas e obteve
resultados quase idênticos (Table 3, linha E), mas ficou com as senoides porque
elas extrapolam para sequências mais longas que as do treino. As duas estão
implementadas, e há um teste que demonstra justamente essa diferença.

## O que os testes verificam

Além dos formatos, as propriedades que realmente importam:

- **Equivalência com o PyTorch** — com os mesmos pesos, nossa `MultiHeadAttention`
  produz exatamente a mesma saída que `torch.nn.MultiheadAttention` (atol 1e-5).
- **Causalidade** — alterar um token futuro não muda nenhuma saída anterior a ele.
- **Invariância a padding** — acrescentar `<pad>` na origem não altera o resultado.
- **Equivalência multi-cabeça** — a versão fundida bate com h atenções separadas.
- **Robustez numérica** — sequência inteiramente de padding não gera `NaN`.
- **Fórmulas do paper** — codificação posicional e escala `1/√d_k` conferidas
  contra o cálculo analítico; pico do Noam exatamente no passo de warmup; sem
  bias nas projeções de atenção.
- **BPE** — ida e volta exata, marcador `@@` correto, palavra nunca vista
  segmentada em pedaços conhecidos, vocabulário único amarrando os três pesos.
- **Batches por tokens** — nenhum batch estoura o orçamento e toda frase entra
  exatamente uma vez.
- **Inicialização sem colapso** — com N=6, a saída do encoder na inicialização
  mantém mais de 50% da distância entre posições (Xavier puro dá 16%).
- **Contagem de parâmetros** — modelo base entre 60M e 70M, como na Table 3.
- **Aprendizado real** — o modelo decora um batch minúsculo (loss < 0.05),
  provando que o grafo de gradientes está inteiro.

---

# O código, passo a passo

Daqui em diante o README vira um tutorial de execução: pegamos a frase
`"Ein Mann geht."` e a seguimos pelo código, linha por linha, com os valores
reais impressos em cada etapa. Para acompanhar ao vivo, rode
`python walkthrough.py` num terminal ao lado — cada "Passo N" aqui
corresponde a `python walkthrough.py --stage N`.

A versão com diagramas e gráficos está em
[`relatorio/transformer.pdf`](relatorio/transformer.pdf).

---

## Parte 1 — Do texto para números

### Passo 1: quebrar a frase em pedaços (BPE)

O paper (§5.1) não trabalha com palavras, e sim com **Byte-Pair Encoding**:
palavras frequentes ficam inteiras, palavras raras são quebradas em pedaços
conhecidos. Tudo começa em `transformer/bpe.py`.

Primeiro, separar palavras e pontuação:

**📄 [transformer/bpe.py:24-29](transformer/bpe.py#L24-L29)**

```python
_PRETOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)     # linha 24

def pretokenize(text):                                      # linha 27
    return _PRETOKEN_RE.findall(text)                       # linha 29
```

A regex tem duas alternativas: `\w+` casa sequências de letras/dígitos, e
`[^\w\s]` casa um caractere que não é letra nem espaço — ou seja, pontuação.
Por isso o ponto final vira um token separado. Repare que **não há `.lower()`**:
o paper preserva maiúsculas, e "Ein" continua "Ein".

```text
"Ein Mann geht."   -->   ['Ein', 'Mann', 'geht', '.']
```

Depois, cada palavra passa por [`BPE.segment_word`](transformer/bpe.py#L122):

**📄 [transformer/bpe.py:122-141](transformer/bpe.py#L122-L141)**

```python
def segment_word(self, word):                                                # linha 122
    symbols = list(word[:-1]) + [word[-1] + END]                             # linha 127
    while len(symbols) > 1:                                                  # linha 128
        best_rank, best_i = None, None
        for i, pair in enumerate(zip(symbols, symbols[1:])):                # linha 131
            rank = self.ranks.get(pair)
            if rank is not None and (best_rank is None or rank < best_rank):
                best_rank, best_i = rank, i
        if best_i is None:                                                   # linha 135
            break
        symbols[best_i:best_i + 2] = [symbols[best_i] + symbols[best_i + 1]] # linha 137

    pieces = [s[:-len(END)] if s.endswith(END) else s + CONTINUE for s in symbols]  # 139
    return pieces                                                            # linha 141
```

A palavra começa como uma lista de caracteres, com um marcador `</w>` colado no
último. O `while` procura, entre todos os pares adjacentes, o que tem **menor
rank** — isto é, a fusão que foi aprendida mais cedo no treino, porque era a mais
frequente — e funde só ela. Repete até não sobrar nenhum par conhecido.

Para as nossas quatro palavras nada acontece de interessante: todas são
frequentes o bastante para terem virado um símbolo só durante o aprendizado, então
o `while` para na primeira volta:

```text
Ein   -> ['Ein']
Mann  -> ['Mann']
geht  -> ['geht']
.     -> ['.']
```

Uma palavra rara mostra o algoritmo trabalhando. "Schneemobilen" (motos de neve)
aparece uma vez no corpus:

```text
Schneemobilen: começa como ['S', 'c', 'h', 'n', 'e', 'e', 'm', 'o', 'b', 'i', 'l', 'e', 'n</w>']
  merge #1     'e' + 'n</w>'  -> ['S', 'c', 'h', 'n', 'e', 'e', 'm', 'o', 'b', 'i', 'l', 'en</w>']
  merge #6     'c' + 'h'      -> ['S', 'ch', 'n', 'e', 'e', 'm', 'o', 'b', 'i', 'l', 'en</w>']
  merge #82    'i' + 'l'      -> ['S', 'ch', 'n', 'e', 'e', 'm', 'o', 'b', 'il', 'en</w>']
  merge #109   'n' + 'e'      -> ['S', 'ch', 'ne', 'e', 'm', 'o', 'b', 'il', 'en</w>']
  merge #120   'S' + 'ch'     -> ['Sch', 'ne', 'e', 'm', 'o', 'b', 'il', 'en</w>']
  merge #129   'e' + 'm'      -> ['Sch', 'ne', 'em', 'o', 'b', 'il', 'en</w>']
  ...
resultado: ['Schneem@@', 'ob@@', 'il@@', 'en']     (9 fusões)
```

O merge #1 é `e + n</w>`: o final "-en" é o par mais frequente de todo o corpus
alemão+inglês. O `@@` da linha 139 marca um pedaço que **continua** na próxima
peça; o último pedaço não tem, porque tinha o `</w>`.

Os 10 000 merges foram aprendidos por [`learn_bpe`](transformer/bpe.py#L36), uma
vez só, antes do treino, sobre o alemão **e** o inglês juntos. É isso que dá um
vocabulário único para as duas línguas.

### Passo 2: pedaços viram ids

Agora [`Vocab.encode`](transformer/data.py#L53):

**📄 [transformer/data.py:53-55](transformer/data.py#L53-L55)**

```python
def encode(self, text, add_special=True):                              # linha 53
    ids = [self.stoi.get(tok, UNK_IDX) for tok in self.tokenize(text)]  # linha 54
    return [BOS_IDX] + ids + [EOS_IDX] if add_special else ids         # linha 55
```

`self.tokenize` é o BPE do passo 1. `stoi` é um dicionário `pedaço -> int`. O
`.get(tok, UNK_IDX)` é o que salva quando o pedaço não existe: devolve o id 3
(`<unk>`). Com BPE isso quase nunca acontece — nas 300 frases de teste, zero
`<unk>`.

A linha 55 envelopa a frase com dois tokens especiais:

| pedaço | id |
|---|---|
| `<bos>` | 1 |
| `Ein` | 8 |
| `Mann` | 16 |
| `geht` | 111 |
| `.` | 4 |
| `<eos>` | 2 |

```text
ids = [1, 8, 16, 111, 4, 2]
```

Os quatro ids especiais são **fixos**:

**📄 [transformer/data.py:21](transformer/data.py#L21)**

```python
PAD_IDX, BOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3
```

O `pad_idx = 0` aparece como default em praticamente toda função do projeto.
Guarde isso: metade da lógica de máscara depende dele.

O vocabulário é ordenado por frequência, então `Ein` = 8 e `Mann` = 16 são
pedaços muito comuns; `geht` = 111 é menos frequente. E o vocabulário é **um só
para alemão e inglês** (9 740 tokens): a mesma tabela que codifica a entrada vai
decodificar a saída. É isso que permite amarrar as três matrizes do §3.4 —
embedding de origem, de destino e a projeção pré-softmax são o mesmo tensor
([model.py:122](transformer/model.py#L122)).

### Passo 3: virar tensor

**📄 [walkthrough.py:162](walkthrough.py#L162)** — este trecho é do tutorial, não do pacote

```python
src = torch.tensor(ids).unsqueeze(0)     # (1, 6)
```

O `unsqueeze(0)` cria a dimensão de batch. Daqui para frente, **toda** a forma
tem batch na frente: `(1, 6)` = uma frase de seis pedaços.

```text
src = [[1, 8, 16, 111, 4, 2]]      forma (1, 6)
```

### Passo 4: a máscara de padding

[`make_source_mask`](transformer/masking.py#L23) chama
[`padding_mask`](transformer/masking.py#L9):

**📄 [transformer/masking.py:9-11](transformer/masking.py#L9-L11)**

```python
def padding_mask(tokens, pad_idx=0):                       # linha 9
    return (tokens != pad_idx).unsqueeze(1).unsqueeze(2)   # linha 11
```

Três coisas acontecem nessa linha:

1. `tokens != pad_idx` → tensor booleano `(1, 6)`, `True` onde **não** é padding.
2. `.unsqueeze(1)` → `(1, 1, 6)`, abre espaço para a dimensão das cabeças.
3. `.unsqueeze(2)` → `(1, 1, 1, 6)`, abre espaço para a dimensão das queries.

```text
src_mask = [[[[True, True, True, True, True, True]]]]     forma (1, 1, 1, 6)
```

Tudo `True` porque nossa frase não tem padding. Mas **por que** as duas
dimensões vazias? Porque os scores de atenção terão forma
`(batch, heads, queries, keys)` = `(1, 8, 6, 6)`. Com a máscara em
`(1, 1, 1, 6)`, o broadcast do PyTorch a estica para todas as cabeças e todas as
queries automaticamente. Uma coluna mascarada some para todo mundo de uma vez.

> **Convenção do projeto inteiro: `True` = posição visível.** O
> `nn.MultiheadAttention` do PyTorch usa o inverso. Se você comparar os dois um
> dia, é aqui que a confusão nasce.

---

## Parte 2 — O encoder

Estamos entrando em [`Transformer.encode`](transformer/model.py#L162):

**📄 [transformer/model.py:162-167](transformer/model.py#L162-L167)**

```python
def encode(self, src, src_mask=None, need_weights=False):     # linha 162
    if src_mask is None:                                       # linha 164
        src_mask = make_source_mask(src, self.config.pad_idx)  # linha 165
    x = self.pos_encoding(self.src_embed(src))                 # linha 166
    return self.encoder(x, src_mask, need_weights)             # linha 167
```

As linhas 164-165 explicam por que `model(src, tgt)` funciona sem você passar
máscara nenhuma: ele constrói sozinho.

A linha 166 tem duas chamadas aninhadas. Vamos abrir as duas.

### Passo 5: embedding (a parte de dentro da linha 166)

[`TokenEmbedding.forward`](transformer/embeddings.py#L17) é uma linha:

**📄 [transformer/embeddings.py:17-18](transformer/embeddings.py#L17-L18)**

```python
def forward(self, tokens):                                # linha 17
    return self.lut(tokens) * math.sqrt(self.d_model)     # linha 18
```

`lut` ("look-up table") é um `nn.Embedding`: uma matriz `(9740, 512)` onde a
linha `i` é o vetor do pedaço `i`. Consultá-la é indexação pura, não
multiplicação de matriz.

Os primeiros 5 valores do vetor de `<bos>` (id 1) e de `Ein` (id 8):

```text
lut[<bos>] = [-0.0597, +0.0219, -0.0405, +0.0074, +0.1287, ...]   # 512 números
lut[Ein]   = [-0.0108, +0.0040, +0.0268, +0.0680, -0.0716, ...]
```

Agora a multiplicação por `√512 = 22.627`:

```text
emb[<bos>] = [-1.3500, +0.4963, -0.9160, +0.1685, +2.9118, ...]
emb[Ein]   = [-0.2453, +0.0908, +0.6070, +1.5389, -1.6212, ...]
```

Confira na mão: `-0.0597 × 22.627 = -1.351` ≈ `-1.3500` (a diferença é
arredondamento na impressão).

**Por que multiplicar?** Olhe as normas dos vetores, antes e depois:

| pedaço | ‖lut‖ | ‖lut × √512‖ |
|---|---|---|
| `<bos>` | 0.91 | 20.58 |
| `Ein` | 0.92 | 20.89 |
| `Mann` | 0.95 | 21.40 |
| `geht` | 1.00 | 22.68 |
| `.` | 0.97 | 21.97 |
| `<eos>` | 0.91 | 20.51 |

No próximo passo vamos **somar** a codificação posicional, cujos valores ficam
em [-1, 1] — um vetor de posição tem norma `√256 = 16`. Com norma ~0.9, o
embedding seria abafado pela posição. Com norma ~21, o conteúdo domina e a
posição entra como um ajuste. É exatamente isso que a seção 3.4 do paper quer
dizer com "multiplicamos os pesos pela raiz de d_model".

### Passo 6: codificação posicional (a parte de fora da linha 166)

Esta etapa tem duas metades, em lugares diferentes do arquivo.

**Metade 1 — o cálculo, que roda uma vez só no `__init__`:**

**📄 [transformer/embeddings.py:33-41](transformer/embeddings.py#L33-L41)**

```python
pe = torch.zeros(max_len, d_model)                                    # linha 33
position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)   # linha 34
div_term = torch.exp(
    torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
)                                                                      # linhas 36-38
pe[:, 0::2] = torch.sin(position * div_term)                          # linha 39
pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].size(1)])   # linha 40
self.register_buffer("pe", pe.unsqueeze(0), persistent=False)         # linha 41
```

A linha 36 parece complicada, mas é só a fórmula do paper reescrita:

```text
exp(-log(10000) · 2i/d)  ==  1 / 10000^(2i/d)
```

Matematicamente idêntico, numericamente mais estável (não calcula uma potência
gigante para depois inverter). Os primeiros valores, com d = 512:

```text
div_term = [+1.0000, +0.9647, +0.9306, +0.8977, +0.8660, ...]   # 256 números
```

As linhas 39-40 usam fatiamento com passo 2: `0::2` são as dimensões pares
(recebem seno) e `1::2` as ímpares (recebem cosseno). O resultado é uma matriz
`(5000, 512)` — 5000 posições pré-calculadas.

Valores reais das três primeiras posições (primeiras 6 dimensões):

```text
pe[posição 0] = [+0.0000, +1.0000, +0.0000, +1.0000, +0.0000, +1.0000, ...]
pe[posição 1] = [+0.8415, +0.5403, +0.8219, +0.5697, +0.8020, +0.5974, ...]
pe[posição 2] = [+0.9093, -0.4161, +0.9364, -0.3509, +0.9581, -0.2863, ...]
```

Confira a fórmula na mão, para `pos=1, i=0`:

```text
PE(1, 0) = sin(1 / 10000^0) = sin(1) = +0.841471   <- bate com pe[1][0]
PE(1, 1) = cos(1 / 10000^0) = cos(1) = +0.540302   <- bate com pe[1][1]
```

A posição 0 é `[0, 1, 0, 1, ...]` porque `sin(0)=0` e `cos(0)=1`.

`register_buffer(..., persistent=False)` na linha 41 significa: acompanha o
modelo quando ele vai para a GPU, mas **não** entra no `state_dict`. Faz
sentido — é recalculado no `__init__`, não há o que salvar.

**Metade 2 — o uso, a cada forward:**

**📄 [transformer/embeddings.py:43-46](transformer/embeddings.py#L43-L46)**

```python
def forward(self, x):                     # linha 43
    x = x + self.pe[:, : x.size(1)]       # linha 45
    return self.dropout(x)                # linha 46
```

Fatia as 6 primeiras posições das 5000 e **soma**. Não concatena: a posição
entra no mesmo espaço vetorial do conteúdo.

```text
emb[<bos>]  = [-1.3500, +0.4963, -0.9160, +0.1685, +2.9118, ...]
pe[0]       = [+0.0000, +1.0000, +0.0000, +1.0000, +0.0000, ...]
                 ↓ soma
x_pos[<bos>] = [-1.3500, +1.4963, -0.9160, +1.1685, +2.9118, ...]
```

Veja a segunda dimensão: `+0.4963 + 1.0000 = +1.4963`. É literalmente uma soma.

O dropout da linha 46 é identidade porque chamamos `model.eval()`. Em treino ele
zeraria 10% dos valores.

Fim da linha 166. Temos `x` com forma `(1, 6, 512)`.

### Passo 7: a pilha de camadas

Linha 167, `self.encoder(x, src_mask)`, cai em
[`Encoder.forward`](transformer/model.py#L66):

**📄 [transformer/model.py:66-69](transformer/model.py#L66-L69)**

```python
def forward(self, x, mask=None, need_weights=False):   # linha 66
    for layer in self.layers:                          # linha 67
        x = layer(x, mask, need_weights)               # linha 68
    return x                                           # linha 69
```

Um `for` de três linhas. `self.layers` foi construído por
[`clones`](transformer/layers.py#L10), que faz `deepcopy` — as 6 camadas são
**independentes**, cada uma com seus próprios pesos. Não é a mesma camada
aplicada 6 vezes.

Vamos entrar na **camada 0**. [`EncoderLayer.forward`](transformer/layers.py#L49):

**📄 [transformer/layers.py:49-51](transformer/layers.py#L49-L51)**

```python
def forward(self, x, mask=None, need_weights=False):                          # linha 49
    x = self.sublayers[0](x, lambda y: self.self_attn(y, y, y, mask, need_weights))  # linha 50
    return self.sublayers[1](x, self.feed_forward)                            # linha 51
```

Duas linhas, dois sublayers. Repare no `lambda` da linha 50: `ResidualConnection`
só sabe chamar `sublayer(x)` com **um** argumento, mas a atenção precisa de
`(y, y, y, mask)`. O lambda faz a ponte. E o fato de ser `(y, y, y)` — o mesmo
tensor três vezes — é o que faz disso *self*-attention.

### Passo 8: dentro da self-attention

O lambda chama [`MultiHeadAttention.forward`](transformer/attention.py#L75):

**📄 [transformer/attention.py:75-85](transformer/attention.py#L75-L85)**

```python
def forward(self, query, key, value, mask=None, need_weights=False):  # linha 75
    if mask is not None and mask.dim() == 3:                          # linha 77
        mask = mask.unsqueeze(1)                                       # linha 78

    q = self._split_heads(self.w_q(query))                             # linha 80
    k = self._split_heads(self.w_k(key))                               # linha 81
    v = self._split_heads(self.w_v(value))                             # linha 82

    out, attn = scaled_dot_product_attention(q, k, v, mask=mask, dropout=self.dropout)  # linha 84
    out = self.w_o(self._merge_heads(out))                             # linha 85
```

**Linhas 80-82 — as projeções.** `w_q`, `w_k` e `w_v` são três
`nn.Linear(512, 512, bias=False)`. Sem bias de propósito: as equações do paper
são `QWᵢ^Q`, `KWᵢ^K`, `VWᵢ^V` — produtos puros
([attention.py:57](transformer/attention.py#L57)). Nossa entrada `x_pos`
`(1, 6, 512)` vira três tensores de mesma forma:

```text
Q[<bos>] = [-0.1140, -1.2431, +0.5463, -1.9969, +0.3929, ...]
K[<bos>] = [+0.3269, +0.9409, +0.4669, -0.5634, +0.6921, ...]
V[<bos>] = [+1.3245, +0.5720, +0.3095, +1.1356, -0.6611, ...]
```

Aqui está a primeira coisa que confunde: **onde estão as 8 cabeças?** O paper
descreve `h` projeções separadas de tamanho `d_k = 64`. O código faz **uma**
projeção de tamanho 512 e depois reorganiza. Veja
[`_split_heads`](transformer/attention.py#L65):

**📄 [transformer/attention.py:65-68](transformer/attention.py#L65-L68)**

```python
def _split_heads(self, x):                                                   # linha 65
    batch, length, _ = x.shape                                                # linha 67
    return x.view(batch, length, self.num_heads, self.d_k).transpose(1, 2)   # linha 68
```

```text
(1, 6, 512)  --view-->  (1, 6, 8, 64)  --transpose(1,2)-->  (1, 8, 6, 64)
 B  L  d_model           B  L  h  d_k                        B  h  L  d_k
```

As 8 cabeças viraram uma **dimensão de batch**. O `matmul` seguinte opera nas 8
em paralelo, sem loop. É matematicamente idêntico ao paper — há um teste que
compara essa versão com 8 atenções calculadas separadamente
(`test_multihead_equals_concat_of_heads`) e outro que compara com o PyTorch
(`test_multihead_matches_pytorch_reference`).

**Linha 84 — a equação 1.** Entramos em
[`scaled_dot_product_attention`](transformer/attention.py#L13), que tem 4 linhas
que importam:

**📄 [transformer/attention.py:23-34](transformer/attention.py#L23-L34)**

```python
d_k = query.size(-1)                                                   # linha 23
scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # linha 24

if mask is not None:
    scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)  # linha 29

attn = F.softmax(scores, dim=-1)                                       # linha 31
if dropout is not None:
    attn = dropout(attn)
return torch.matmul(attn, value), attn                                 # linha 34
```

*Linha 24.* `Q @ Kᵀ` produz `(1, 8, 6, 6)`: para cada cabeça, uma matriz 6×6 onde
a entrada `[i][j]` diz o quanto a posição `i` se interessa pela posição `j`.

A divisão por `√d_k = √64 = 8` importa de verdade. Medindo nos nossos dados:

```text
scores sem escala:  desvio padrão 11.820
scores ÷ 8:         desvio padrão  1.478
```

Com desvio 11.8, o softmax fica quase one-hot e o gradiente morre. Com 1.5, ele
distribui. Essa é toda a motivação do "scaled" no nome.

Os scores da **cabeça 0** (linha = quem pergunta, coluna = quem é olhado):

```text
          <bos>    Ein   Mann   geht      .  <eos>
 <bos>     0.11  -3.32  -0.87  -1.85  -0.14   0.12
 Ein      -0.16   0.70   0.23   0.69   0.19   0.37
 Mann     -1.60  -1.67  -1.79  -1.12  -1.55   0.70
 geht     -0.43  -0.62   0.05  -0.81   0.89   2.60
 .        -3.58  -3.30  -1.90  -3.77  -1.74  -2.26
 <eos>    -0.11  -0.34  -0.37  -0.23  -1.49  -1.26
```

*Linha 29.* Aplica a máscara. Aqui não muda nada (não há padding), mas repare no
valor usado: `torch.finfo(scores.dtype).min`, que vale `-3.40e+38` — e **não**
`-inf`. Se uma linha inteira fosse mascarada (acontece quando um batch tem uma
sequência 100% padding), `-inf` em todas as colunas faria o softmax dividir
`0/0` e produzir `NaN`. O valor finito evita isso. Existe um teste só para esse
caso: `test_all_padding_row_does_not_produce_nan`.

*Linha 31.* Softmax na última dimensão — normaliza **sobre as keys**, ou seja,
cada linha passa a somar 1:

```text
          <bos>    Ein   Mann   geht      .  <eos>
 <bos>     0.30   0.01   0.11   0.04   0.23   0.30
 Ein       0.10   0.23   0.14   0.23   0.14   0.16
 Mann      0.07   0.06   0.05   0.10   0.07   0.65
 geht      0.04   0.03   0.06   0.02   0.13   0.73
 .         0.05   0.07   0.29   0.04   0.34   0.20
 <eos>     0.25   0.20   0.19   0.22   0.06   0.08
```

Olhe a linha `geht`: o score 2.60 para `<eos>` virou peso 0.73, e os scores
negativos viraram quase zero. O softmax amplifica diferenças.

Cada cabeça aprende um padrão diferente. A **cabeça 1**, nos mesmos dados:

```text
          <bos>    Ein   Mann   geht      .  <eos>
 <bos>     0.10   0.09   0.08   0.16   0.12   0.45
 Ein       0.11   0.07   0.22   0.03   0.04   0.53
 Mann      0.12   0.11   0.08   0.67   0.01   0.02
 geht      0.55   0.20   0.03   0.05   0.02   0.15
 .         0.13   0.12   0.07   0.02   0.02   0.65
 <eos>     0.48   0.27   0.19   0.01   0.02   0.03
```

Na cabeça 0, `Mann` olha para `<eos>` (0.65). Na cabeça 1, `Mann` olha para
`geht` (0.67) — o sujeito olhando para o seu verbo. São views diferentes da mesma
frase; é para isso que servem múltiplas cabeças.

*Linha 34.* `attn @ value`: `(1,8,6,6) @ (1,8,6,64)` → `(1,8,6,64)`. Cada posição
vira uma **média dos vetores V**, ponderada pelos pesos de atenção. É aqui que a
informação de fato se move entre posições.

**Linha 85 — juntar as cabeças e projetar.** `_merge_heads`
([attention.py:70](transformer/attention.py#L70)) desfaz o que `_split_heads`
fez, e `w_o` é a matriz `W^O` do paper:

```text
concat  = [+0.2302, +0.3758, -0.0635, -0.1955, -0.3662, ...]   (1, 6, 512)
× W_O   = [-0.1092, +0.1929, +0.0083, -0.2339, -0.0044, ...]   (1, 6, 512)
```

Saímos da atenção com a mesma forma com que entramos: `(1, 6, 512)`.

### Passo 9: Add & Norm

Voltamos para a linha 50 do `EncoderLayer`, que estava dentro de
[`ResidualConnection.forward`](transformer/layers.py#L36):

**📄 [transformer/layers.py:36-37](transformer/layers.py#L36-L37)**

```python
def forward(self, x, sublayer):                              # linha 36
    return self.norm(x + self.dropout(sublayer(x)))          # linha 37
```

A linha 37 é o post-LN do paper: `LayerNorm(x + Sublayer(x))`. Só existe essa
variante no código, porque só existe essa no paper. Acompanhe:

```text
x            = [-1.3500, +1.4963, -0.9160, +1.1685, +2.9118, ...]   entrada
Sublayer(x)  = [-0.1092, +0.1929, +0.0083, -0.2339, -0.0044, ...]   saída da atenção
               ↓ soma (a conexão residual)
x + Sub(x)   = [-1.4592, +1.6892, -0.9076, +0.9346, +2.9074, ...]
               ↓ LayerNorm
saída        = [-1.7799, +1.3347, -1.2348, +0.5934, +2.5487, ...]
```

O que o LayerNorm fez, medido em cada uma das 6 posições:

```text
média:   [-0.00311, -0.00323, -0.00396, -0.00306, -0.00292, -0.00309]
desvio:  [ 1.0019,   1.0017,   1.0013,   1.0015,   1.0021,   1.0025 ]
```

Média ~0 e desvio ~1, posição por posição. É isso que mantém os valores em uma
faixa estável mesmo empilhando muitas camadas.

A conexão residual (`x +`) é o que garante que o gradiente tenha um caminho
direto até as camadas iniciais. Sem ela, redes com 6 camadas de atenção não
treinam.

### Passo 10: Feed Forward

Linha 51 do `EncoderLayer`, que chama
[`PositionwiseFeedForward.forward`](transformer/layers.py#L24):

**📄 [transformer/layers.py:24-25](transformer/layers.py#L24-L25)**

```python
def forward(self, x):                                            # linha 24
    return self.w_2(self.dropout(self.w_1(x).relu()))            # linha 25
```

Lendo de dentro para fora: `w_1` expande, ReLU corta, `w_2` volta. É a equação 2
do paper, `max(0, xW₁ + b₁)W₂ + b₂` — e aqui **tem** bias, porque a equação tem.

```text
entrada        (1, 6,  512)   [-1.7799, +1.3347, -1.2348, +0.5934, +2.5487, ...]
  w_1          (1, 6, 2048)   [-1.2624, -0.6319, -1.3542, -0.1702, +0.1607, ...]
  ReLU         (1, 6, 2048)   [+0.0000, +0.0000, +0.0000, +0.0000, +0.1607, ...]
  w_2          (1, 6,  512)   [-0.1407, +0.3320, -0.1092, +0.1795, +0.4123, ...]
```

Os quatro primeiros valores após o ReLU são zero porque os quatro primeiros de
`w_1` eram negativos; o quinto (+0.1607) passou. No total, **93.3% das ativações
foram zeradas**. Essa esparsidade é normal e é parte do que dá capacidade ao
modelo.

O nome "positionwise" quer dizer: o **mesmo** MLP roda em cada uma das 6
posições, independentemente. Ele não mistura posições. Quem mistura posições é
só a atenção — essa divisão de papéis é o coração da arquitetura.

Depois disso vem outro Add & Norm (o `sublayers[1]`), e a camada 0 termina.

### Passo 11: repetir e obter a memória

O `for` da linha 67 repete tudo mais cinco vezes. Acompanhando o vetor de `<bos>`:

```text
entrada (x_pos)     = [-1.3500, +1.4963, -0.9160, +1.1685, +2.9118, ...]
depois da camada 0  = [-1.7487, +1.5351, -1.2325, +0.7161, +2.7201, ...]
depois da camada 1  = [-1.5243, +1.7379, -1.2241, +1.0645, +2.7998, ...]
depois da camada 2  = [-1.6249, +1.9580, -1.2751, +1.2742, +2.5804, ...]
depois da camada 3  = [-1.6942, +2.0189, -1.4240, +1.5086, +2.7746, ...]
depois da camada 4  = [-1.9069, +1.8060, -1.0063, +1.6768, +2.7660, ...]
depois da camada 5  = [-1.5464, +2.0007, -0.8919, +1.6409, +3.0255, ...]
```

**A forma nunca muda: `(1, 6, 512)` do começo ao fim.** Empilhar camadas
enriquece a representação, não a redimensiona. Esse tensor final é a **memória**
(`H_enc` no diagrama).

> **Um detalhe que custou caro.** Com 6 camadas post-LN, cada camada mistura as
> posições pela atenção e o LayerNorm renormaliza a soma. Com inicialização
> Xavier pura, a distância média entre as 6 posições na saída do encoder já era
> só 16% da entrada **antes de qualquer treino** — e o treino convergia para um
> encoder em que todas as posições eram o mesmo vetor, com cross-attention
> uniforme (o modelo ainda traduzia, como um "saco de palavras", mas sem
> alinhamento). A correção está em
> [model.py:143-149](transformer/model.py#L143-L149): as projeções de saída de
> cada sublayer (`W^O` e `W_2`) são escaladas por `1/√(2N)` na inicialização.
> Com isso, em 20 frases de validação, a distância entre posições na saída fica
> em 95% da entrada. O paper não especifica a inicialização; o teste
> `test_encoder_keeps_positions_distinct_at_init` protege essa escolha.

O encoder rodou **uma vez**. O decoder vai consultar essa mesma memória em todos
os passos de geração.

---

## Parte 3 — O decoder gera pedaço por pedaço

Agora [`greedy_decode`](transformer/decoding.py#L12), em `transformer/decoding.py`:

**📄 [transformer/decoding.py:23-36](transformer/decoding.py#L23-L36)**

```python
memory = model.encode(src, src_mask)                       # linha 23
batch = src.size(0)
ys = torch.full((batch, 1), bos_idx, dtype=torch.long)     # linha 25
finished = torch.zeros(batch, dtype=torch.bool)            # linha 26

for _ in range(max_len - 1):                               # linha 28
    out = model.decode(ys, memory, src_mask, make_target_mask(ys, pad_idx))  # linha 29
    next_token = model.generator(out[:, -1]).argmax(dim=-1)                  # linha 30
    next_token = torch.where(finished, torch.full_like(next_token, pad_idx), next_token)  # 31
    ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)   # linha 32
    finished |= next_token == eos_idx                      # linha 33
    if bool(finished.all()):                               # linha 34
        break                                              # linha 35
return ys                                                  # linha 36
```

**Linha 23 está FORA do loop.** Esse é o ponto mais importante do arquivo: a
memória não depende do que já foi gerado, então recalculá-la a cada passo seria
desperdício puro.

**Linha 25:** começamos com uma sequência de um token só, o `<bos>`.

```text
ys = [[1]]
```

Vamos acompanhar cada volta do `for`. Em cada uma, além da máscara e das
probabilidades, vou mostrar **para onde a cross-attention da última camada está
olhando** (média das 8 cabeças, na posição que está gerando).

### Volta 0 — gerando o primeiro pedaço

`ys = [[1]]` = `['<bos>']`

A linha 29 chama [`make_target_mask`](transformer/masking.py#L27):

**📄 [transformer/masking.py:27-29](transformer/masking.py#L27-L29)**

```python
def make_target_mask(tgt, pad_idx=0):                                      # linha 27
    return padding_mask(tgt, pad_idx) & subsequent_mask(tgt.size(1), device=tgt.device)  # 29
```

Com um token só, a máscara é trivial:

```text
tgt_mask = [[1]]
```

Depois `model.decode` roda a pilha do decoder. Cada
[`DecoderLayer`](transformer/layers.py#L64) tem **três** sublayers (o encoder
tem dois):

**📄 [transformer/layers.py:64-69](transformer/layers.py#L64-L69)**

```python
def forward(self, x, memory, src_mask=None, tgt_mask=None, need_weights=False):  # linha 64
    x = self.sublayers[0](x, lambda y: self.self_attn(y, y, y, tgt_mask, need_weights))  # 65
    x = self.sublayers[1](
        x, lambda y: self.cross_attn(y, memory, memory, src_mask, need_weights)  # linha 67
    )
    return self.sublayers[2](x, self.feed_forward)                                # linha 69
```

Compare as linhas 65 e 67 — **é aqui que mora a diferença toda**:

**📄 [transformer/layers.py:65-67](transformer/layers.py#L65-L67)** — trecho simplificado das linhas 65 e 67

```python
self.self_attn(y, y, y, tgt_mask)                 # linha 65: Q, K, V vêm todos do decoder
self.cross_attn(y, memory, memory, src_mask)      # linha 67: Q do decoder; K e V da memória
#               ^Q   ^K      ^V
```

A classe é a mesma `MultiHeadAttention` que já lemos. O que muda é só quem são
os argumentos. Na cross-attention, `Q` vem do que estamos gerando e `K, V` vêm
do que a frase alemã diz.

Cross-attention da última camada, na posição `<bos>`:

```text
        <bos>   Ein   Mann   geht     .   <eos>
         0.03  0.44   0.37   0.05  0.07   0.03
```

Olhando para `Ein` (0.44) e `Mann` (0.37) — os dois pedaços de "Ein Mann". Faz
sentido: vai gerar o artigo.

Linha 30: `model.generator(out[:, -1])`. O
[`generator`](transformer/model.py#L120) é um `nn.Linear(512, 9740)` sem bias,
cujo peso **é o mesmo tensor** do embedding
([model.py:122](transformer/model.py#L122)). O `out[:, -1]` pega **só a última
posição** — as anteriores não servem para prever o próximo pedaço.

Os 5 pedaços mais prováveis:

| pedaço | logit | prob |
|---|---|---|
| **A** | 12.262 | **91.83%** |
| Man | 8.397 | 1.93% |
| The | 6.838 | 0.41% |
| One | 6.771 | 0.38% |
| There | 5.917 | 0.16% |

Repare que é "A" maiúsculo: o vocabulário preserva caixa, e o modelo aprendeu
que frases começam assim. `argmax` escolhe `A` (id 7). Linha 32 concatena:

```text
ys = [[1, 7]]
```

### Volta 1

`ys = [[1, 7]]` = `['<bos>', 'A']`

Agora a máscara causal começa a fazer trabalho:

```text
tgt_mask = [[1, 0],
            [1, 1]]
```

Linha 0 (`<bos>`) só vê a si mesma. Linha 1 (`A`) vê as duas. É a
[`subsequent_mask`](transformer/masking.py#L14), que é literalmente um `.tril()`:

**📄 [transformer/masking.py:19](transformer/masking.py#L19)**

```python
mask = torch.ones(size, size, dtype=torch.bool, device=device).tril()   # linha 19
```

Cross-attention:

```text
        <bos>   Ein   Mann   geht     .   <eos>
         0.01  0.21   0.66   0.10  0.01   0.01
```

0.66 em `Mann`. Ele sabe qual pedaço alemão está traduzindo agora.

| pedaço | logit | prob |
|---|---|---|
| **man** | 11.520 | **88.42%** |
| guy | 7.039 | 1.00% |
| male | 6.791 | 0.78% |
| A | 5.645 | 0.25% |
| Man | 5.374 | 0.19% |

```text
ys = [[1, 7, 18]]
```

### Volta 2 — o passo mais interessante

`ys = [[1, 7, 18]]` = `['<bos>', 'A', 'man']`

```text
tgt_mask = [[1, 0, 0],
            [1, 1, 0],
            [1, 1, 1]]
```

Cross-attention:

```text
        <bos>   Ein   Mann   geht     .   <eos>
         0.02  0.00   0.02   0.72  0.11   0.14
```

**0.72 em `geht`.** Terminou de traduzir o sujeito e mudou o foco para o verbo,
sozinho. Ninguém programou esse alinhamento: ele emergiu do treino.

| pedaço | logit | prob |
|---|---|---|
| **is** | 11.140 | **43.40%** |
| walking | 10.707 | 28.15% |
| walks | 10.352 | 19.74% |
| goes | 6.530 | 0.43% |
| going | 6.046 | 0.27% |

Aqui dá para ver o modelo hesitando entre três formas legítimas de traduzir
`geht`: "is" (começando "is walking"), "walking" e "walks". Somadas, 91% da massa
está no verbo. O greedy pega a maior, "is", e segue.

> É por causa de casos assim que existe o beam search: com beam 4, o modelo
> mantém "walking" e "walks" vivos por mais alguns passos e compara as frases
> inteiras no fim. Aqui o beam chega à mesma resposta; veja
> [`beam_search`](transformer/decoding.py#L40).

```text
ys = [[1, 7, 18, 17]]
```

### Volta 3 — um pedaço alemão, dois ingleses

`ys` = `['<bos>', 'A', 'man', 'is']`

```text
        <bos>   Ein   Mann   geht     .   <eos>
         0.02  0.04   0.01   0.73  0.03   0.17
```

**Ainda 0.73 em `geht`.** O verbo alemão "geht" vira dois pedaços em inglês
("is" + "walking"), e a cross-attention fica parada em `geht` durante os dois.
É um alinhamento um-para-dois, e ele aparece sozinho na matriz.

| pedaço | logit | prob |
|---|---|---|
| **walking** | 13.369 | **96.29%** |
| going | 9.097 | 1.34% |
| walks | 6.775 | 0.13% |
| running | 6.022 | 0.06% |
| walk | 5.935 | 0.06% |

Depois de "is", a dúvida sumiu: 96% em "walking".

```text
ys = [[1, 7, 18, 17, 73]]
```

### Voltas 4 e 5 — fechando

`ys` = `['<bos>', 'A', 'man', 'is', 'walking']`, cross-attention:

```text
        <bos>   Ein   Mann   geht     .   <eos>
         0.09  0.02   0.00   0.12  0.22   0.55
```

Atenção agora em `<eos>` (0.55) e `.` (0.22): a frase alemã está acabando.

| pedaço | logit | prob |
|---|---|---|
| **.** | 10.356 | **62.10%** |
| `<eos>` | 8.069 | 6.31% |
| in | 8.045 | 6.16% |

Gera `.`. Na volta seguinte:

| pedaço | logit | prob |
|---|---|---|
| **`<eos>`** | 12.108 | **93.04%** |
| A | 3.873 | 0.02% |

Gera `<eos>`. A linha 33 marca `finished = True`, a linha 34 verifica, e a
linha 35 dá `break`.

```text
ys = [[1, 7, 18, 17, 73, 4, 2]]
```

### O alinhamento completo

Juntando as cross-attentions de todos os passos numa matriz só (linha = o que o
decoder acabou de ler, coluna = origem):

```text
            <bos>   Ein   Mann   geht     .   <eos>
 <bos>       0.03  0.44   0.37   0.05  0.07   0.03
 A           0.01  0.21   0.66   0.10  0.01   0.01
 man         0.02  0.00   0.02   0.72  0.11   0.14
 is          0.02  0.04   0.01   0.73  0.03   0.17
 walking     0.09  0.02   0.00   0.12  0.22   0.55
 .           0.24  0.23   0.03   0.03  0.11   0.36
```

Leia por linha: depois de ler `A`, olha para `Mann` (vai gerar "man"); depois de
`man` e de `is`, olha para `geht` (vai gerar "is" e depois "walking"); depois de
`walking`, olha para o fim da frase. O modelo aprendeu a alinhar as duas línguas
sem nunca ter recebido um único alinhamento anotado — só pares de frases.

### Passo final: voltar para texto

[`Vocab.decode`](transformer/data.py#L57):

**📄 [transformer/data.py:57-66](transformer/data.py#L57-L66)**

```python
def decode(self, ids, strip_special=True):                    # linha 57
    pieces = []
    for i in ids:
        i = int(i)
        if strip_special and i in (PAD_IDX, BOS_IDX):         # linha 61
            continue
        if strip_special and i == EOS_IDX:                     # linha 63
            break
        pieces.append(self.itos[i] if i < len(self.itos) else UNK)
    return self.detokenize(pieces)                             # linha 66
```

`continue` pula `<pad>` e `<bos>`; `break` **para** no `<eos>` (tudo depois é
lixo). E `detokenize` chama [`BPE.decode`](transformer/bpe.py#L151), que desfaz o
`@@ ` dos pedaços — aqui não há nenhum, mas "Schneem@@ ob@@ il@@ en" viraria
"Schneemobilen" de volta.

```text
[1, 7, 18, 17, 73, 4, 2]   -->   "A man is walking ."
```

Fim. Da string de entrada à string de saída.

---

## Parte 4 — E o treino?

O forward é **o mesmo**; o que muda é o que se faz com os logits. Em
[`run_epoch`](train_multi30k.py#L73):

**📄 [train_multi30k.py:83-92](train_multi30k.py#L83-L92)**

```python
logits = model(src, tgt_in)              # linha 83
loss = loss_fn(logits.float(), tgt_out)  # linha 84
loss.backward()                          # linha 88
optimizer.step()                         # linha 91
scheduler.step()                         # linha 92 — por PASSO, não por época
```

A diferença central está em como `tgt_in` e `tgt_out` são montados:

**📄 [transformer/data.py:192](transformer/data.py#L192)**

```python
return src, tgt[:, :-1], tgt[:, 1:]
```

Deslocados em uma posição. Com nossa frase alvo `[<bos>, A, man, is, walking, ., <eos>]`:

```text
tgt_in  = [<bos>,  A,    man,  is,       walking,  .    ]   o que o decoder lê
tgt_out = [A,      man,  is,   walking,  .,        <eos>]   o que ele deve prever
```

Na posição 0 ele lê `<bos>` e deve prever `A`; na posição 1 lê `A` e deve prever
`man`; e assim por diante. Isso é o **teacher forcing**: em vez de alimentar o
decoder com o que ele mesmo gerou (como na Parte 3), alimentamos com a resposta
correta.

E é justamente a máscara causal que torna isso válido: **as 6 previsões acontecem
em um único forward, em paralelo**, e nenhuma delas conseguiu espiar a resposta
das seguintes. Sem a máscara, prever `A` na posição 0 seria trivial — bastaria
copiar o `A` da posição 1 da entrada.

Compare com a Parte 3: lá foram 6 forwards sequenciais, porque em inferência não
existe resposta para dar de comer ao decoder.


O resto da receita do §5 está nos mesmos lugares de sempre: batches por tokens
([`TokenBatchSampler`](transformer/data.py#L140)), Adam com warmup
([`noam_lambda`](transformer/optim.py#L9)), label smoothing
([`LabelSmoothingLoss`](transformer/optim.py#L37)) e média dos últimos
checkpoints ([`average_checkpoints`](transformer/utils.py#L13)).

---

## Onde parar com o debugger

```bash
python walkthrough.py --stage 8 --pause    # abre o pdb antes do passo
```

No pdb: `n` avança uma linha, `s` entra na função, `p nome` imprime, `c`
continua, `q` sai.

Se preferir depurar o código direto, estes são os pontos que ensinam mais:

| Onde | O que inspecionar |
|---|---|
| [bpe.py:137](transformer/bpe.py#L137) | `p symbols` a cada volta: a palavra sendo fundida pedaço a pedaço |
| [attention.py:24](transformer/attention.py#L24) | `p scores[0,0]` — a matriz 6×6 antes do softmax |
| [attention.py:31](transformer/attention.py#L31) | `p attn[0,0].sum(-1)` — confirmar que cada linha soma 1 |
| [attention.py:80](transformer/attention.py#L80) | `p query is key` — `True` em self-attention, `False` em cross |
| [layers.py:67](transformer/layers.py#L67) | ver `memory` entrando como K e V, não como Q |
| [model.py:186](transformer/model.py#L186) | o ponto único onde encoder e decoder se encontram |
| [decoding.py:32](transformer/decoding.py#L32) | `p ys` a cada volta: a sequência crescendo |

No VS Code, `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "walkthrough",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/walkthrough.py",
      "args": ["--stage", "8"],
      "console": "integratedTerminal",
      "justMyCode": false
    }
  ]
}
```

`justMyCode: false` é o que permite entrar (`F11`) dentro do `nn.Linear` e do
`LayerNorm` do PyTorch quando quiser ir ainda mais fundo.

---

## Quebre de propósito

A forma mais rápida de confirmar que você entendeu uma linha é estragá-la e
prever qual teste quebra. Os resultados abaixo foram **medidos**, não previstos
(suíte de 38 testes):

| Mude | Testes que quebram |
|---|---|
| Tire o `/ math.sqrt(d_k)` de [attention.py:24](transformer/attention.py#L24) | **2** — `test_attention_scaling_matches_formula` e `test_multihead_matches_pytorch_reference` |
| Troque `.tril()` por `.triu()` em [masking.py:19](transformer/masking.py#L19) | **3** — entre eles `test_decoder_cannot_see_the_future`: "vazamento de informacao futura" |
| Passe `None` no lugar de `src_mask` em [layers.py:67](transformer/layers.py#L67) | **1** — `test_padding_does_not_change_predictions` |
| Troque `finfo.min` por `float("-inf")` em [attention.py:29](transformer/attention.py#L29) | **1** — `test_all_padding_row_does_not_produce_nan` |
| Ponha `bias=True` nas projeções em [attention.py:48](transformer/attention.py#L48) | **2** — `test_attention_projections_have_no_bias` e a comparação com o PyTorch |
| Troque a escala da init por `1.0` em [model.py:143](transformer/model.py#L143) | **1** — `test_encoder_keeps_positions_distinct_at_init` |
| Tire o `* math.sqrt(self.d_model)` de [embeddings.py:18](transformer/embeddings.py#L18) | **nenhum** |

A última linha é a mais instrutiva. A suíte protege a **correção** da
arquitetura, não a **qualidade** do treino: remover a escala do embedding deixa
tudo verde, mas piora o aprendizado. Para ver esse efeito você precisa rodar
`python train_copy.py` antes e depois e comparar as curvas de loss.

---

## Resumo: a ordem de execução

A árvore de chamadas de ponta a ponta. Cada endereço é um link para a linha.

| Chamada | Onde | O que faz |
|---|---|---|
| `Vocab.encode` | [data.py:53-55](transformer/data.py#L53-L55) | `<bos>` + ids + `<eos>` |
| └&nbsp;`pretokenize` | [bpe.py:29](transformer/bpe.py#L29) | separa palavras, preserva caixa |
| └&nbsp;`BPE.segment_word` | [bpe.py:128-137](transformer/bpe.py#L128-L137) | funde pares por rank, até não sobrar nenhum |
| `make_source_mask` | [masking.py:24](transformer/masking.py#L24), [masking.py:11](transformer/masking.py#L11) | `True` = visível; `(B,1,1,L)` para o broadcast |
| `Transformer.encode` | [model.py:166-167](transformer/model.py#L166-L167) | embedding + posição, depois a pilha |
| └&nbsp;`TokenEmbedding.forward` | [embeddings.py:18](transformer/embeddings.py#L18) | `lut(ids)` × √d_model |
| └&nbsp;`PositionalEncoding.forward` | [embeddings.py:45](transformer/embeddings.py#L45) | + `pe[:, :L]` |
| └&nbsp;`Encoder.forward` | [model.py:67](transformer/model.py#L67) | `for` nas 6 camadas |
| &nbsp;&nbsp;&nbsp;&nbsp;└&nbsp;`EncoderLayer.forward` | [layers.py:50-51](transformer/layers.py#L50-L51) | dois sublayers |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└&nbsp;`MultiHeadAttention.forward` | [attention.py:80-85](transformer/attention.py#L80-L85) | projeções (sem bias), cabeças, `W_O` |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└&nbsp;`scaled_dot_product_attention` | [attention.py:24](transformer/attention.py#L24), [attention.py:29](transformer/attention.py#L29), [attention.py:31](transformer/attention.py#L31), [attention.py:34](transformer/attention.py#L34) | `QKᵀ/√d_k`, máscara, softmax, `× V` |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└&nbsp;`ResidualConnection.forward` | [layers.py:37](transformer/layers.py#L37) | Add & Norm |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└&nbsp;`PositionwiseFeedForward.forward` | [layers.py:25](transformer/layers.py#L25) | 512 → 2048 → ReLU → 512 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└&nbsp;`ResidualConnection.forward` | [layers.py:37](transformer/layers.py#L37) | Add & Norm |
| `greedy_decode` | [decoding.py:28](transformer/decoding.py#L28) | loop por pedaço; memória calculada fora |
| └&nbsp;`make_target_mask` | [masking.py:29](transformer/masking.py#L29) | padding `&` causal |
| └&nbsp;`Transformer.decode` | [model.py:173-174](transformer/model.py#L173-L174) | embedding + posição do alvo, depois a pilha |
| &nbsp;&nbsp;&nbsp;&nbsp;└&nbsp;`DecoderLayer.forward` | [layers.py:65](transformer/layers.py#L65), [layers.py:67](transformer/layers.py#L67), [layers.py:69](transformer/layers.py#L69) | masked self-attn · cross-attn (K,V da memória) · FFN |
| └&nbsp;`generator` | [model.py:120](transformer/model.py#L120) | Linear 512 → 9740 (peso = embedding) |
| └&nbsp;`argmax` | [decoding.py:30](transformer/decoding.py#L30) | só a última posição |
| `Vocab.decode` | [data.py:66](transformer/data.py#L66), [bpe.py:153](transformer/bpe.py#L153) | pula especiais, para no `<eos>`, desfaz o `@@ ` |

---

## Referências e linhagem

A arquitetura é do paper abaixo. A organização do código e vários nomes
(`clones`, `subsequent_mask`, `PositionwiseFeedForward`, `generator`, `memory`)
seguem a convenção de *The Annotated Transformer* (Harvard NLP), que é a
referência didática padrão para esta implementação. O BPE segue Sennrich et al.

```bibtex
@inproceedings{vaswani2017attention,
  title     = {Attention Is All You Need},
  author    = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob
               and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2017}
}

@inproceedings{sennrich2016bpe,
  title     = {Neural Machine Translation of Rare Words with Subword Units},
  author    = {Sennrich, Rico and Haddow, Barry and Birch, Alexandra},
  booktitle = {Proceedings of ACL},
  year      = {2016}
}
```
