"""Traduz com um checkpoint salvo por train_multi30k.py.

    python translate.py --text "Ein Mann geht die Strasse entlang."
    python translate.py                      # modo interativo
    python translate.py --attention          # mostra a cross-attention da traducao
    python translate.py --pieces             # mostra a segmentacao BPE
"""

import argparse
import sys

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from transformer import Transformer, TransformerConfig, beam_search, greedy_decode
from transformer.data import BOS_IDX, EOS_IDX, Vocab


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "vocab" not in ckpt:
        raise ValueError(f"{path} e de uma versao antiga (vocabularios separados); "
                         "retreine com train_multi30k.py")
    model = Transformer(TransformerConfig.from_dict(ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, Vocab.from_dict(ckpt["vocab"])


@torch.no_grad()
def translate(model, text, vocab, device, beam_size=4, show_attention=False, show_pieces=False):
    ids = torch.tensor(vocab.encode(text), device=device).unsqueeze(0)
    max_len = ids.size(1) + 50          # secao 6.1: comprimento da entrada + 50

    if show_pieces:
        print(f"  BPE {' '.join(vocab.tokenize(text))}")

    if beam_size > 1:
        out = beam_search(model, ids, beam_size=beam_size, max_len=max_len,
                          bos_idx=BOS_IDX, eos_idx=EOS_IDX, length_penalty=0.6)
    else:
        out = greedy_decode(model, ids, max_len=max_len, bos_idx=BOS_IDX, eos_idx=EOS_IDX)[0]

    result = vocab.decode(out.tolist())

    if show_attention:
        model(ids, out.unsqueeze(0)[:, :-1], need_weights=True)
        attn = model.attention_maps()[f"decoder.{len(model.decoder.layers) - 1}.cross_attn"]
        src_pieces = [vocab.itos[i] for i in ids[0].tolist()]
        tgt_pieces = [vocab.itos[i] for i in out[1:].tolist()]
        print_alignment(attn[0].mean(0), src_pieces, tgt_pieces)

    return result


def print_alignment(attn, src_pieces, tgt_pieces):
    """Para cada pedaco gerado, mostra o pedaco de origem mais atendido."""
    print("\n  alinhamento (gerado <- origem mais atendida, media das cabecas):")
    for i, tgt_tok in enumerate(tgt_pieces):
        if i >= attn.size(0):
            break
        row = attn[i][: len(src_pieces)]
        j = int(row.argmax())
        print(f"    {tgt_tok:<18} <- {src_pieces[j]:<18} ({row[j].item():.2f})")


def main():
    ap = argparse.ArgumentParser(description="Traduz DE->EN com o modelo treinado")
    ap.add_argument("--checkpoint", default="checkpoints/multi30k.pt")
    ap.add_argument("--text", help="frase em alemao; sem isso entra no modo interativo")
    ap.add_argument("--beam", type=int, default=4, help="1 = greedy")
    ap.add_argument("--attention", action="store_true")
    ap.add_argument("--pieces", action="store_true", help="mostra a segmentacao BPE")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    model, vocab = load_checkpoint(args.checkpoint, device)
    cfg = model.config
    print(f"modelo: N={cfg.num_layers} d_model={cfg.d_model} h={cfg.num_heads} "
          f"| {model.num_parameters():,} parametros | vocab {len(vocab)} | beam {args.beam}\n")

    def go(text):
        print(f"  DE  {text}")
        print(f"  EN  {translate(model, text, vocab, device, args.beam, args.attention, args.pieces)}")

    if args.text:
        go(args.text)
        return

    print("Digite uma frase em alemao (linha vazia ou Ctrl+C para sair).")
    while True:
        try:
            text = input("\nDE> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            break
        go(text)


if __name__ == "__main__":
    main()
