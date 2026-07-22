"""
Training baseline VQA YOLO+LSTM dengan LOSS MASKING dua kepala.

  * Tiap sampel hanya menghukum kepala yang cocok answer_type-nya:
      open_ended  -> CrossEntropy pada kepala OE
      multi_label -> BCEWithLogits pada kepala ML
  * Trunk (badan bersama) belajar dari SEMUA sampel; tiap kepala dari jenisnya.
  * Validasi tiap epoch -> simpan checkpoint terbaik -> early stopping.

Jalankan:  python train.py
Prasyarat: build_vocab.py & extract_features.py sudah dijalankan.
"""

import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import cfg
from build_vocab import load as load_vocab
from dataset import VQADataset, collate
from model import VQALSTM

CKPT = os.path.join(cfg.CACHE_DIR, "model.pt")


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)


@torch.no_grad()
def evaluate_val(model, loader, device):
    """Skor validasi ringkas: rata-rata (akurasi OE, micro-F1 ML)."""
    model.eval()
    oe_correct = oe_total = 0
    tp = fp = fn = 0
    for b in loader:
        logit_oe, logit_ml = model(b["q_idx"].to(device), b["q_len"],
                                   b["visual"].to(device))
        atype = b["atype"]
        oe_m = atype == 0
        ml_m = atype == 1
        if oe_m.any():
            pred = logit_oe[oe_m].argmax(1).cpu()
            tgt = b["t_oe"][oe_m]
            oe_correct += (pred == tgt).sum().item(); oe_total += oe_m.sum().item()
        if ml_m.any():
            pred = (torch.sigmoid(logit_ml[ml_m]).cpu() > cfg.ML_THRESHOLD).float()
            tgt = b["t_ml"][ml_m]
            tp += (pred * tgt).sum().item()
            fp += (pred * (1 - tgt)).sum().item()
            fn += ((1 - pred) * tgt).sum().item()
    oe_acc = oe_correct / oe_total if oe_total else 0.0
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    ml_f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return oe_acc, ml_f1, (oe_acc + ml_f1) / 2


def main():
    set_seed(cfg.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    v = load_vocab()
    n_words, n_oe, n_ml = len(v["word2idx"]), len(v["oe2idx"]), len(v["ml2idx"])

    tr = VQADataset("train"); va = VQADataset("val")
    print(f"train={len(tr)}  val={len(va)}  | n_words={n_words} n_oe={n_oe} n_ml={n_ml}")
    tr_dl = DataLoader(tr, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=collate)
    va_dl = DataLoader(va, batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=collate)

    model = VQALSTM(n_words, n_oe, n_ml).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()

    best, best_ep, wait = -1.0, -1, 0
    for ep in range(1, cfg.EPOCHS + 1):
        model.train()
        run = 0.0
        for b in tr_dl:
            opt.zero_grad()
            logit_oe, logit_ml = model(b["q_idx"].to(device), b["q_len"],
                                       b["visual"].to(device))
            atype = b["atype"].to(device)
            oe_m = atype == 0
            ml_m = atype == 1
            loss = torch.zeros((), device=device)
            if oe_m.any():
                loss = loss + ce(logit_oe[oe_m], b["t_oe"].to(device)[oe_m])
            if ml_m.any():
                loss = loss + bce(logit_ml[ml_m], b["t_ml"].to(device)[ml_m])
            loss.backward(); opt.step()
            run += loss.item()

        oe_acc, ml_f1, score = evaluate_val(model, va_dl, device)
        flag = ""
        if score > best:
            best, best_ep, wait = score, ep, 0
            torch.save({"state": model.state_dict(),
                        "n_words": n_words, "n_oe": n_oe, "n_ml": n_ml}, CKPT)
            flag = "  <- best (saved)"
        else:
            wait += 1
        print(f"ep {ep:2d} | loss {run/len(tr_dl):.4f} | val OE_acc {oe_acc:.4f} "
              f"ML_F1 {ml_f1:.4f} | score {score:.4f}{flag}")
        if wait >= cfg.PATIENCE:
            print(f"early stopping (patience {cfg.PATIENCE}) — best epoch {best_ep} "
                  f"score {best:.4f}")
            break

    print(f"\nModel terbaik disimpan -> {CKPT} (epoch {best_ep}, score {best:.4f})")


if __name__ == "__main__":
    main()
