"""
Evaluasi baseline pada TEST + TABEL PERBANDINGAN lawan metode template.

  * Metrik memakai logika IDENTIK dengan vqa/eval_vqa.Acc:
      open_ended  -> accuracy (Exact-Match)
      multi_label -> Precision/Recall/F1 (micro) + Subset-EM
  * Gold multi_label dikanonikkan (split(':')[0]) agar setara output baseline.
      -> Untuk 2 tipe composite (grade_breakdown, color_breakdown) gold aslinya
         "Label:N"; baseline TIDAK bisa memancarkan jumlah, jadi dua tipe ini
         dinilai di LEVEL LABEL & ditandai (*) di tabel. Keterbatasan ini justru
         bukti argumen paper (template menghitung langsung dari YOLO).
  * Angka template dibaca dari data/vqa/eval_results_test_id.json (hasil paper).

Hasil:
  cache/baseline_eval_test_id.json   — metrik baseline (format seperti eval paper)
  cache/comparison_test_id.md        — tabel Template vs YOLO+LSTM

Jalankan:  python evaluate.py
"""

import json
import os
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

import cfg
from build_vocab import load as load_vocab
from dataset import VQADataset, collate
from model import VQALSTM

CKPT = os.path.join(cfg.CACHE_DIR, "model.pt")
TEMPLATE_EVAL = os.path.join(cfg.DATA_VQA, "eval_results_test_id.json")
COMPOSITE = {"grade_breakdown", "color_breakdown"}   # dinilai level-label (*)


# --- akumulator metrik: identik dengan vqa/eval_vqa.Acc ---
class Acc:
    def __init__(self):
        self.correct = self.total_oe = 0
        self.tp = self.fp = self.fn = self.subset_ok = self.total_ml = 0
        self.answer_type = None

    def add_open(self, s_ans, g_ans):
        self.answer_type = "open_ended"; self.total_oe += 1
        s = s_ans[0].strip() if s_ans else ""
        g = g_ans[0].strip() if g_ans else ""
        if s == g:
            self.correct += 1

    def add_multi(self, s_ans, g_ans):
        self.answer_type = "multi_label"; self.total_ml += 1
        s, g = set(s_ans), set(g_ans)
        self.tp += len(s & g); self.fp += len(s - g); self.fn += len(g - s)
        if s == g:
            self.subset_ok += 1

    def report(self):
        if self.answer_type == "open_ended":
            n = self.total_oe
            return {"answer_type": "open_ended", "n": n,
                    "accuracy": round(self.correct / n, 4) if n else None}
        n = self.total_ml
        p = self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0
        r = self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        return {"answer_type": "multi_label", "n": n,
                "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
                "subset_em": round(self.subset_ok / n, 4) if n else None}


@torch.no_grad()
def run(split="test", lang=cfg.LANG):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    v = load_vocab()
    oe_inv = {i: a for a, i in v["oe2idx"].items()}
    ml_inv = {i: a for a, i in v["ml2idx"].items()}

    ck = torch.load(CKPT, map_location=device)
    model = VQALSTM(ck["n_words"], ck["n_oe"], ck["n_ml"]).to(device)
    model.load_state_dict(ck["state"]); model.eval()

    ds = VQADataset(split, lang)
    dl = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=collate)

    # gold per qid (untuk jawaban emas kanonik)
    with open(cfg.gold_path(split, lang), encoding="utf-8") as fh:
        gold = {g["question_id"]: g for g in json.load(fh)}

    per_type = defaultdict(Acc)
    per_tl = defaultdict(Acc)   # (track, level, answer_type)

    for b in dl:
        logit_oe, logit_ml = model(b["q_idx"].to(device), b["q_len"],
                                   b["visual"].to(device))
        prob_ml = torch.sigmoid(logit_ml).cpu()
        pred_oe = logit_oe.argmax(1).cpu()
        for i, qid in enumerate(b["qid"]):
            g = gold[qid]
            qt = g["question_type"]; at = g["answer_type"]
            track, level = cfg.track_of(qt), cfg.level_of(qt)
            if at == "open_ended":
                s_ans = [oe_inv.get(int(pred_oe[i]), "<UNK>")]
                g_ans = [str(g["answer"])]
                per_type[qt].add_open(s_ans, g_ans)
                per_tl[(track, level, at)].add_open(s_ans, g_ans)
            else:
                s_ans = [ml_inv[j] for j in range(len(ml_inv))
                         if prob_ml[i, j] > cfg.ML_THRESHOLD]
                g_ans = [a.split(":")[0] for a in g["answers"]]   # kanonik
                per_type[qt].add_multi(s_ans, g_ans)
                per_tl[(track, level, at)].add_multi(s_ans, g_ans)

    results = {
        "config": {"split": split, "lang": lang, "model": "YOLO+LSTM baseline",
                   "yolo": cfg.MODEL_YOLO, "ml_threshold": cfg.ML_THRESHOLD},
        "per_question_type": {qt: per_type[qt].report() for qt in sorted(per_type)},
        "aggregate": {},
    }
    for (tr, lv, at), acc in sorted(per_tl.items()):
        results["aggregate"].setdefault(f"Jalur_{tr}", {}).setdefault(lv, {})[at] = acc.report()

    out = os.path.join(cfg.CACHE_DIR, f"baseline_eval_{split}_{lang}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    return results


def metric_str(rep):
    if rep is None:
        return "-"
    if rep["answer_type"] == "open_ended":
        return f"acc={rep['accuracy']}"
    return f"F1={rep['f1']}"


def metric_val(rep):
    if rep is None:
        return None
    return rep["accuracy"] if rep["answer_type"] == "open_ended" else rep["f1"]


def comparison(baseline):
    with open(TEMPLATE_EVAL, encoding="utf-8") as fh:
        templ = json.load(fh)["per_question_type"]
    base = baseline["per_question_type"]

    rows = []
    for qt in sorted(set(templ) | set(base)):
        t = templ.get(qt); bR = base.get(qt)
        tv = metric_val(t); bv = metric_val(bR)
        delta = (tv - bv) if (tv is not None and bv is not None) else None
        star = " *" if qt in COMPOSITE else ""
        metric_name = "acc" if (t and t["answer_type"] == "open_ended") else "F1"
        rows.append((qt + star, metric_name, tv, bv, delta))

    lines = [
        "# Perbandingan: Template (rule-based) vs YOLO+LSTM (baseline) — TEST split",
        "",
        "Metrik: open_ended = accuracy; multi_label = F1. Δ = Template − Baseline (positif = template unggul).",
        "`*` = tipe composite (grade/color_breakdown): baseline dinilai di level-label",
        "(tak bisa memancarkan jumlah) — keterbatasan bawaan VQA neural.",
        "",
        "| question_type | metrik | Template | YOLO+LSTM | Δ |",
        "|---|---|---:|---:|---:|",
    ]
    for qt, mn, tv, bv, d in rows:
        ds = f"{d:+.3f}" if d is not None else "-"
        lines.append(f"| {qt} | {mn} | {tv if tv is not None else '-'} "
                     f"| {bv if bv is not None else '-'} | {ds} |")

    # rata-rata makro (hindari composite di headline)
    def macro(vals):
        vals = [x for x in vals if x is not None]
        return round(sum(vals) / len(vals), 4) if vals else None
    tv_all = macro([tv for qt, mn, tv, bv, d in rows])
    bv_all = macro([bv for qt, mn, tv, bv, d in rows])
    lines += ["",
              f"**Rata-rata makro (semua tipe):** Template = {tv_all} | YOLO+LSTM = {bv_all}",
              ""]

    out = os.path.join(cfg.CACHE_DIR, f"comparison_test_{cfg.LANG}.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return out, "\n".join(lines)


if __name__ == "__main__":
    res = run("test")
    path, text = comparison(res)
    print(text)
    print("\nTersimpan ->", path)
