"""Traducao alemao -> ingles no Multi30k com a receita do paper (secao 5).

Arquitetura = modelo base da Table 3 (N=6, d_model=512, h=8, d_ff=2048, P_drop=0.1).
Dados = BPE com vocabulario compartilhado (5.1), batches por tokens (5.1).
Otimizacao = Adam(0.9, 0.98, 1e-9) + warmup 4000 (5.3), label smoothing 0.1 (5.4).
Avaliacao = media dos ultimos 5 checkpoints, beam 4, alpha 0.6, max_len = |src|+50 (6.1).

O que NAO da para copiar do paper e o tamanho: WMT14 tem 4.5M pares e batches de
25k tokens; o Multi30k tem 29k pares e cabe numa GPU de 6 GB com ~2.5k tokens.

    python train_multi30k.py                         # ~30 min numa GPU modesta
    python train_multi30k.py --epochs 2 --limit 2000  # smoke test
"""

import argparse
import math
import sys
import time
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):  # console do Windows nao e UTF-8 por padrao
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
from torch.utils.data import DataLoader

from transformer import (
    LabelSmoothingLoss,
    Transformer,
    TransformerConfig,
    average_checkpoints,
    beam_search,
    make_optimizer,
)
from transformer.data import (
    BOS_IDX,
    EOS_IDX,
    PAD_IDX,
    TokenBatchSampler,
    TranslationDataset,
    Vocab,
    collate_batch,
    download_multi30k,
    tokenize,
)


# --------------------------------------------------------------------------- #
# BLEU (BLEU-4 com suavizacao e brevity penalty)
# --------------------------------------------------------------------------- #

def bleu_score(hypotheses, references, max_n=4):
    """BLEU de corpus. hypotheses/references sao listas de listas de tokens."""
    precisions = []
    for n in range(1, max_n + 1):
        matches = total = 0
        for hyp, ref in zip(hypotheses, references):
            hyp_ngrams = Counter(tuple(hyp[i:i + n]) for i in range(len(hyp) - n + 1))
            ref_ngrams = Counter(tuple(ref[i:i + n]) for i in range(len(ref) - n + 1))
            matches += sum((hyp_ngrams & ref_ngrams).values())
            total += max(sum(hyp_ngrams.values()), 0)
        precisions.append((matches + 1) / (total + 1))  # suavizacao add-1

    hyp_len = sum(len(h) for h in hypotheses)
    ref_len = sum(len(r) for r in references)
    brevity = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / max(hyp_len, 1))
    return 100 * brevity * math.exp(sum(math.log(p) for p in precisions) / max_n)


# --------------------------------------------------------------------------- #

def run_epoch(model, loader, loss_fn, device, optimizer=None, scheduler=None,
              autocast=None, clip=0.0):
    train = optimizer is not None
    model.train(train)
    total_loss, total_tokens = 0.0, 0

    with torch.set_grad_enabled(train):
        for src, tgt_in, tgt_out in loader:
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
            with autocast:
                logits = model(src, tgt_in)
                loss = loss_fn(logits.float(), tgt_out)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                scheduler.step()

            n_tokens = int((tgt_out != PAD_IDX).sum())
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def translate(model, sentence, vocab, device, beam_size=4):
    ids = torch.tensor(vocab.encode(sentence), device=device).unsqueeze(0)
    out = beam_search(model, ids, beam_size=beam_size, max_len=ids.size(1) + 50,
                      bos_idx=BOS_IDX, eos_idx=EOS_IDX, length_penalty=0.6)
    return vocab.decode(out.tolist())


def save_checkpoint(path, model, vocab):
    torch.save({"model": model.state_dict(), "config": model.config.to_dict(),
                "vocab": vocab.to_dict()}, path)


def main():
    ap = argparse.ArgumentParser(description="Transformer base (paper) DE->EN no Multi30k")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--tokens-per-batch", type=int, default=2500,
                    help="tokens por batch, contando padding (paper: 25000)")
    ap.add_argument("--bpe-merges", type=int, default=10000, help="paper: vocab ~37k")
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--d-ff", type=int, default=2048)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=4000)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--average-last", type=int, default=5, help="checkpoints promediados (6.1)")
    ap.add_argument("--clip", type=float, default=0.0, help="0 = sem clipping (paper)")
    ap.add_argument("--max-len", type=int, default=100)
    ap.add_argument("--limit", type=int, default=None, help="usa so N pares de treino")
    ap.add_argument("--no-amp", action="store_true", help="desliga bfloat16 na GPU")
    ap.add_argument("--eval-sentences", type=int, default=300, help="frases de teste para o BLEU")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--save", default="checkpoints/multi30k.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp and torch.cuda.is_bf16_supported()
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp)

    print("== dados ==")
    corpus = download_multi30k(args.data_dir)
    train_de, train_en = corpus[("train", "de")], corpus[("train", "en")]
    if args.limit:
        train_de, train_en = train_de[: args.limit], train_en[: args.limit]

    t0 = time.time()
    vocab = Vocab.build(train_de + train_en, bpe_merges=args.bpe_merges)
    print(f"  BPE: {args.bpe_merges} merges aprendidos em {time.time() - t0:.0f}s "
          f"(DE+EN juntos) | vocabulario compartilhado: {len(vocab)} tokens")

    datasets = {
        "train": TranslationDataset(train_de, train_en, vocab, args.max_len),
        "valid": TranslationDataset(corpus[("valid", "de")], corpus[("valid", "en")], vocab, args.max_len),
        "test": TranslationDataset(corpus[("test", "de")], corpus[("test", "en")], vocab, args.max_len),
    }
    loaders = {
        name: DataLoader(ds, batch_sampler=TokenBatchSampler(ds.lengths(), args.tokens_per_batch,
                                                             shuffle=(name == "train"), seed=args.seed),
                         collate_fn=collate_batch)
        for name, ds in datasets.items()
    }
    print(f"  treino: {len(datasets['train'])} pares em {len(loaders['train'])} batches "
          f"de ate {args.tokens_per_batch} tokens")

    print("\n== modelo ==")
    model = Transformer(TransformerConfig(
        src_vocab_size=len(vocab),
        tgt_vocab_size=len(vocab),
        num_layers=args.layers,
        d_model=args.d_model,
        num_heads=args.heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        pad_idx=PAD_IDX,
        share_embeddings=True,   # secao 3.4: um vocabulario, tres pesos amarrados
        tie_generator=True,
    )).to(device)
    print(f"  N={args.layers} d_model={args.d_model} h={args.heads} d_ff={args.d_ff} "
          f"dropout={args.dropout} | {model.num_parameters():,} parametros | {device}"
          f"{' | bfloat16' if use_amp else ''}")

    loss_fn = LabelSmoothingLoss(len(vocab), pad_idx=PAD_IDX, smoothing=args.label_smoothing)
    optimizer, scheduler = make_optimizer(model, d_model=args.d_model, warmup=args.warmup)

    print(f"\n== treino ({args.epochs} epocas, warmup {args.warmup} passos) ==")
    save_path = Path(args.save)
    epoch_dir = save_path.parent / "epochs"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    window, best_window, best_valid, best_epoch = [], [], float("inf"), 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = run_epoch(model, loaders["train"], loss_fn, device, optimizer, scheduler,
                               autocast, args.clip)
        valid_loss = run_epoch(model, loaders["valid"], loss_fn, device, autocast=autocast)

        # Janela deslizante dos ultimos K checkpoints (secao 6.1). O paper treina um
        # numero fixo de passos; aqui o "fim" do treino e a melhor epoca de validacao,
        # e a janela que termina nela e a que sera promediada.
        path = epoch_dir / f"epoch_{epoch:02d}.pt"
        save_checkpoint(path, model, vocab)
        window = (window + [path])[-args.average_last:]
        flag = ""
        if valid_loss < best_valid:
            best_valid, best_epoch, best_window = valid_loss, epoch, list(window)
            flag = "  <- melhor"
        for old in epoch_dir.glob("epoch_*.pt"):        # apaga o que saiu das duas janelas
            if old not in window and old not in best_window:
                old.unlink()

        print(f"  epoca {epoch:2d}/{args.epochs}  treino {train_loss:.3f}  "
              f"valid {valid_loss:.3f}  ppl {math.exp(min(valid_loss, 20)):.1f}  "
              f"lr {scheduler.get_last_lr()[0]:.2e}  passo {scheduler.last_epoch}  "
              f"{time.time() - t0:.0f}s{flag}", flush=True)

    print(f"\n== media dos {len(best_window)} checkpoints ate a epoca {best_epoch} (secao 6.1) ==")
    model.load_state_dict(torch.load(best_window[-1], map_location=device, weights_only=False)["model"])
    single = run_epoch(model, loaders["valid"], loss_fn, device, autocast=autocast)
    model.load_state_dict(average_checkpoints(best_window))
    averaged = run_epoch(model, loaders["valid"], loss_fn, device, autocast=autocast)
    print(f"  valid loss  epoca {best_epoch} sozinha {single:.3f}  ->  media {averaged:.3f}")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(save_path, model, vocab)
    for old in epoch_dir.glob("epoch_*.pt"):
        old.unlink()
    epoch_dir.rmdir()

    print("\n== avaliacao (BLEU no teste, beam 4, alpha 0.6) ==")
    test_de, test_en = corpus[("test", "de")], corpus[("test", "en")]
    n_eval = min(args.eval_sentences, len(test_de))
    hyps, refs = [], []
    for de, en in zip(test_de[:n_eval], test_en[:n_eval]):
        hyps.append(tokenize(translate(model, de, vocab, device)))
        refs.append(tokenize(en))
    print(f"  BLEU-4 em {n_eval} frases: {bleu_score(hyps, refs):.2f}")

    print("\n== exemplos ==")
    for de, en in list(zip(test_de, test_en))[:5]:
        print(f"  DE   {de}")
        print(f"  REF  {en}")
        print(f"  GER  {translate(model, de, vocab, device)}\n")

    print(f"checkpoint salvo em {save_path}")


if __name__ == "__main__":
    main()
