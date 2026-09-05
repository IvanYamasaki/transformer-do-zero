"""Inspeciona o Transformer: arquitetura, contagem de parametros, mascaras e atencao.

Roda em segundos na CPU e nao precisa de treino.

    python demo.py              # so texto
    python demo.py --plots      # salva figuras em figures/
"""

import argparse
import sys

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from transformer import Transformer, TransformerConfig, make_target_mask, noam_lambda

PAD, BOS = 0, 1


def section(title):
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def show_architecture():
    section("1. Modelo base do paper (Table 3: N=6, d_model=512, d_ff=2048, h=8)")
    model = Transformer(TransformerConfig(src_vocab_size=37000, tgt_vocab_size=37000))

    groups = {"embeddings": 0, "encoder": 0, "decoder": 0, "generator": 0}
    seen = set()
    for name, p in model.named_parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        for key in groups:
            if name.startswith(key) or (key == "embeddings" and "embed" in name):
                groups[key] += p.numel()
                break

    total = model.num_parameters()
    for name, count in groups.items():
        if count:
            print(f"  {name:<12} {count / 1e6:7.2f}M  ({100 * count / total:4.1f}%)")
    print(f"  {'TOTAL':<12} {total / 1e6:7.2f}M   <- o paper reporta 65M para o modelo base")

    big = Transformer(TransformerConfig.big(37000, 37000))
    print(f"\n  variante 'big'  {big.num_parameters() / 1e6:.1f}M   <- o paper reporta 213M")

    d = model.config
    print(f"\n  d_k = d_v = d_model / h = {d.d_model} / {d.num_heads} = {d.d_model // d.num_heads}")


def show_forward():
    section("2. Passagem para frente (encoder-decoder)")
    model = Transformer(TransformerConfig(1000, 1000, num_layers=2, d_model=64,
                                          num_heads=4, d_ff=128, dropout=0.0)).eval()
    src = torch.randint(4, 1000, (2, 9))
    tgt = torch.randint(4, 1000, (2, 7))

    memory = model.encode(src)
    logits = model(src, tgt)
    print(f"  src        {tuple(src.shape)}      ids de entrada")
    print(f"  memory     {tuple(memory.shape)}   saida do encoder")
    print(f"  tgt        {tuple(tgt.shape)}      entrada do decoder (teacher forcing)")
    print(f"  logits     {tuple(logits.shape)}   distribuicao sobre o vocabulario")

    probs = torch.softmax(logits, dim=-1)
    print(f"\n  soma das probabilidades por posicao: {probs.sum(-1).mean().item():.6f} (deve ser 1.0)")


def show_masks():
    section("3. Mascaras (secao 3.2.3): True = visivel, False = bloqueado")
    tgt = torch.tensor([[BOS, 11, 12, 13, PAD]])
    mask = make_target_mask(tgt, PAD)[0, 0]
    print("      alvo:  <bos>   11    12    13  <pad>")
    for i, row in enumerate(mask):
        cells = "  ".join(" X " if v else " . " for v in row)
        print(f"  pos {i}: {cells}")
    print("\n  A diagonal inferior impede olhar o futuro; a ultima coluna e padding.")


def show_causality():
    section("4. Prova de causalidade: mudar o futuro nao altera o passado")
    model = Transformer(TransformerConfig(100, 100, num_layers=2, d_model=32,
                                          num_heads=4, d_ff=64, dropout=0.0)).eval()
    src = torch.randint(4, 100, (1, 5))
    a = torch.tensor([[BOS, 10, 11, 12, 13]])
    b = a.clone()
    b[0, 3] = 77

    out_a, out_b = model(src, a), model(src, b)
    diff = (out_a - out_b).abs().max(dim=-1).values[0]
    for i, d in enumerate(diff.tolist()):
        marca = "inalterado" if d < 1e-6 else "MUDOU"
        print(f"  posicao {i}: variacao maxima {d:.2e}  {marca}")
    print("\n  Trocamos o token da posicao 3; posicoes 0-2 nao se moveram.")


def show_attention(save_plots=False):
    section("5. Mapas de atencao (nao treinado, so para ver os formatos)")
    model = Transformer(TransformerConfig(100, 100, num_layers=2, d_model=32,
                                          num_heads=4, d_ff=64, dropout=0.0)).eval()
    src = torch.randint(4, 100, (1, 6))
    tgt = torch.randint(4, 100, (1, 5))
    model(src, tgt, need_weights=True)

    for name, weights in model.attention_maps().items():
        print(f"  {name:<26} {tuple(weights.shape)}   (batch, cabecas, consultas, chaves)")

    cross = model.attention_maps()["decoder.1.cross_attn"][0, 0]
    print(f"\n  decoder.1.cross_attn, cabeca 0 — cada linha soma 1: {cross.sum(-1).tolist()}")

    if save_plots:
        plot_attention(cross)


def show_schedule():
    section("6. Schedule de learning rate (eq. 3 do paper, warmup = 4000)")
    for step in (1, 100, 1000, 4000, 8000, 100000):
        print(f"  passo {step:>6}: lr = {noam_lambda(step, 512, 4000):.3e}")
    print("\n  Sobe linear ate 4000 passos e cai com 1/sqrt(passo).")


def plot_attention(cross):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    from transformer import PositionalEncoding

    # Estas figuras saem de um modelo NAO treinado: servem so para ver os formatos.
    out_dir = Path("figures/demo")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4, 3.2))
    im = ax.imshow(cross.numpy(), cmap="viridis")
    ax.set_xlabel("posicao na origem")
    ax.set_ylabel("posicao no destino")
    ax.set_title("cross-attention (cabeca 0)")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_dir / "attention.png", dpi=130)
    plt.close(fig)

    pe = PositionalEncoding(128, dropout=0.0, max_len=100).pe[0]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    im = ax.imshow(pe.numpy().T, aspect="auto", cmap="RdBu")
    ax.set_xlabel("posicao")
    ax.set_ylabel("dimensao")
    ax.set_title("codificacao posicional senoidal")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_dir / "positional_encoding.png", dpi=130)
    plt.close(fig)

    steps = range(1, 20000, 20)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(list(steps), [noam_lambda(s, 512, 4000) for s in steps])
    ax.axvline(4000, ls="--", c="gray")
    ax.set_xlabel("passo")
    ax.set_ylabel("learning rate")
    ax.set_title("Noam schedule (warmup = 4000)")
    fig.tight_layout()
    fig.savefig(out_dir / "noam_schedule.png", dpi=130)
    plt.close(fig)

    print(f"\n  figuras salvas em {out_dir}/")


def main():
    ap = argparse.ArgumentParser(description="Inspeciona a implementacao do Transformer")
    ap.add_argument("--plots", action="store_true", help="salva figuras em figures/")
    args = ap.parse_args()

    torch.manual_seed(0)
    show_architecture()
    show_forward()
    show_masks()
    show_causality()
    show_attention(save_plots=args.plots)
    show_schedule()
    print("\nTudo certo. Rode `python tests/test_transformer.py` para a bateria de testes.")


if __name__ == "__main__":
    main()
