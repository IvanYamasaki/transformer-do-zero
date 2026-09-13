"""Compara as tres implementacoes de atencao em tempo e memoria.

    python bench_attention.py                 # na GPU, se houver
    python bench_attention.py --device cpu
    python bench_attention.py --backward      # inclui o passo reverso

O que se espera ver, e o motivo de o benchmark existir:

  - "math" cresce com L^2 em memoria, porque materializa a matriz de scores
    (B, h, L, L). Numa placa de 6 GB isso estoura por volta de L = 4096.
  - "flash" em PyTorch puro nao materializa nada disso e sobe devagar, mas
    perde em tempo: o laco de blocos roda no interpretador.
  - "sdpa" e o kernel fundido; em GPU vira FlashAttention-2 e ganha nos dois.
"""

import argparse
import math
import time

import torch

from transformer.attention import scaled_dot_product_attention
from transformer.flash import flash_attention, sdpa_attention


def roda(fn, q, k, v, mask, backward, repeticoes):
    """Devolve (ms por passagem, pico de memoria em MB) ou None se estourar."""
    dispositivo = q.device
    try:
        if dispositivo.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        else:
            torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

        for _ in range(2):                       # aquecimento
            saida, _ = fn(q, k, v, mask)
            if backward:
                saida.sum().backward()
                q.grad = k.grad = v.grad = None

        if dispositivo.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(repeticoes):
            saida, _ = fn(q, k, v, mask)
            if backward:
                saida.sum().backward()
                q.grad = k.grad = v.grad = None
        if dispositivo.type == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / repeticoes * 1000

        pico = (torch.cuda.max_memory_allocated() / 2 ** 20
                if dispositivo.type == "cuda" else float("nan"))
        return ms, pico
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None


def main():
    ap = argparse.ArgumentParser(description="Atencao: math x flash x sdpa")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--d-k", type=int, default=64)
    ap.add_argument("--lengths", type=int, nargs="*",
                    default=[128, 512, 1024, 2048, 4096])
    ap.add_argument("--block", type=int, default=256, help="bloco do flash em PyTorch")
    ap.add_argument("--backward", action="store_true")
    ap.add_argument("--repeticoes", type=int, default=5)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    args = ap.parse_args()

    dispositivo = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    if dispositivo.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}  |  torch {torch.__version__}")
    print(f"B={args.batch}  h={args.heads}  d_k={args.d_k}  dtype={args.dtype}  "
          f"{'forward+backward' if args.backward else 'so forward'}\n")

    impls = {
        "math": lambda q, k, v, m: scaled_dot_product_attention(q, k, v, mask=m),
        "flash": lambda q, k, v, m: flash_attention(q, k, v, mask=m,
                                                    block_q=args.block, block_k=args.block),
        "sdpa": lambda q, k, v, m: sdpa_attention(q, k, v, mask=m),
    }

    cab = f"{'L':>6}  {'matriz L x L':>13}  " + "  ".join(
        f"{n:>10}" for n in impls) + "     " + "  ".join(f"{n + ' MB':>11}" for n in impls)
    print(cab)
    print("-" * len(cab))

    for comprimento in args.lengths:
        forma = (args.batch, args.heads, comprimento, args.d_k)
        q, k, v = (torch.randn(forma, device=dispositivo, dtype=dtype,
                               requires_grad=args.backward) for _ in range(3))
        # Mascara causal, o caso mais comum num decoder.
        mask = torch.ones(comprimento, comprimento, dtype=torch.bool,
                          device=dispositivo).tril().view(1, 1, comprimento, comprimento)

        bytes_matriz = (args.batch * args.heads * comprimento ** 2
                        * torch.finfo(dtype).bits // 8)
        tempos, picos = [], []
        for nome, fn in impls.items():
            r = roda(fn, q, k, v, mask, args.backward, args.repeticoes)
            tempos.append(f"{r[0]:10.1f}" if r else f"{'estourou':>10}")
            picos.append(f"{r[1]:11.0f}" if r and not math.isnan(r[1]) else f"{'-':>11}")
        print(f"{comprimento:>6}  {bytes_matriz / 2 ** 20:10.0f} MB  "
              + "  ".join(tempos) + "     " + "  ".join(picos))
        del q, k, v, mask
        if dispositivo.type == "cuda":
            torch.cuda.empty_cache()

    print("\ntempos em ms por passagem; memoria = pico alocado na GPU")


if __name__ == "__main__":
    main()
