"""Tarefa de copia: o modelo aprende a repetir a sequencia de entrada.

E o "hello world" do Transformer — treina em ~30s na CPU e mostra o pipeline
inteiro funcionando: mascaras, Noam warmup, label smoothing e decodificacao.

    python train_copy.py
"""

import argparse
import time

import torch

from transformer import LabelSmoothingLoss, TransformerConfig, Transformer, greedy_decode, make_optimizer

PAD, BOS, EOS = 0, 1, 2


def make_batch(batch_size, seq_len, vocab_size, device):
    """Sequencias aleatorias de <bos> tok... <eos>, iguais na origem e no destino."""
    body = torch.randint(4, vocab_size, (batch_size, seq_len), device=device)
    bos = torch.full((batch_size, 1), BOS, device=device)
    eos = torch.full((batch_size, 1), EOS, device=device)
    seq = torch.cat([bos, body, eos], dim=1)
    return seq, seq  # copia: destino identico a origem


def main():
    ap = argparse.ArgumentParser(description="Treina o Transformer na tarefa de copia")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=10)
    ap.add_argument("--vocab-size", type=int, default=32)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    model = Transformer(TransformerConfig(
        src_vocab_size=args.vocab_size,
        tgt_vocab_size=args.vocab_size,
        num_layers=args.layers,
        d_model=args.d_model,
        num_heads=args.heads,
        d_ff=args.d_model * 4,
        dropout=0.1,
    )).to(device)

    print(f"dispositivo: {device} | parametros: {model.num_parameters():,}")

    loss_fn = LabelSmoothingLoss(args.vocab_size, pad_idx=PAD, smoothing=0.1)
    optimizer, scheduler = make_optimizer(model, d_model=args.d_model, warmup=args.warmup)

    model.train()
    start = time.time()
    for step in range(1, args.steps + 1):
        src, tgt = make_batch(args.batch_size, args.seq_len, args.vocab_size, device)
        logits = model(src, tgt[:, :-1])
        loss = loss_fn(logits, tgt[:, 1:])

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % 50 == 0 or step == 1:
            lr = scheduler.get_last_lr()[0]
            print(f"passo {step:4d}/{args.steps}  loss {loss.item():.4f}  lr {lr:.2e}")

    print(f"\ntreino em {time.time() - start:.1f}s\n")

    # ------------------------------------------------------------------ #
    model.eval()
    src, _ = make_batch(5, args.seq_len, args.vocab_size, device)
    generated = greedy_decode(model, src, max_len=args.seq_len + 2, bos_idx=BOS, eos_idx=EOS)

    print("origem -> gerado")
    acertos = 0
    for i in range(src.size(0)):
        esperado = src[i, 1:-1].tolist()
        obtido = [t for t in generated[i, 1:].tolist() if t not in (PAD, EOS)][: len(esperado)]
        ok = esperado == obtido
        acertos += ok
        print(f"  {'OK ' if ok else 'ERR'} {esperado}\n      {obtido}")

    print(f"\nsequencias copiadas corretamente: {acertos}/{src.size(0)}")
    if acertos == src.size(0):
        print("O modelo aprendeu a tarefa de copia.")


if __name__ == "__main__":
    main()
