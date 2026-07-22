"""
Precompute fitur visual dari deteksi YOLO train16 -> cache/feat_{split}_{lang}.npz

YOLO dibekukan, jadi fitur dihitung SEKALI lalu dipakai ulang tiap epoch
(kalau tidak, YOLO jalan ribuan kali sia-sia).

Fitur = vektor 33-dim (lihat cfg): concat(GLOBAL 17, LOKAL 16).
  * pertanyaan global : bagian lokal = nol.
  * pertanyaan lokal  : objek gold dicocokkan ke box prediksi via IoU>=MATCH_IOU.
        - train : jika tak cocok -> sampel di-SKIP (keep=False) agar sinyal bersih.
        - val/test : jika tak cocok -> fitur lokal = nol (dihitung SALAH saat eval,
                     meniru vqa/eval_vqa.py: missed detection).

Enrichment box (obj, xc, area, color) MENIRU vqa/generate_vqa.generate_for_image
dan memakai vqa/color.py yang SAMA -> fitur warna identik dengan metode template.

Jalankan:  python extract_features.py           (semua split)
           python extract_features.py --split val
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

import cfg

# --- import modul vqa/ (dibaca, TIDAK diubah) ---
sys.path.insert(0, cfg.VQA_DIR)
import config as VC          # vqa/config.py  (CLASS_NAMES, REJECT_ID, ...)
import color as CL           # vqa/color.py   (HSV -> warna, identik)


# ---------------------------------------------------------------------------
# YOLO inference (setelan identik dengan vqa/eval_vqa.predict_boxes)
# ---------------------------------------------------------------------------
def predict_boxes(model, img_path):
    res = model.predict(img_path, conf=cfg.PRED_CONF, iou=cfg.PRED_NMS_IOU,
                        imgsz=cfg.PRED_IMGSZ, agnostic_nms=cfg.PRED_AGNOSTIC,
                        max_det=cfg.PRED_MAX_DET, verbose=False)[0]
    boxes = []
    if res.boxes is None:
        return boxes
    xywhn = res.boxes.xywhn.cpu().numpy()
    cls = res.boxes.cls.cpu().numpy().astype(int)
    for c, b in zip(cls, xywhn):
        boxes.append({"cls": int(c), "bbox": tuple(round(float(v), 5) for v in b)})
    return boxes


def enrich(boxes, img):
    """Tambah obj, xc, area, color — MENIRU generate_vqa.generate_for_image."""
    for i, b in enumerate(boxes, 1):
        b["obj"] = f"obj_{i}"
        b["xc"] = b["bbox"][0]
        b["area"] = b["bbox"][2] * b["bbox"][3]
        if img is None:
            b["color"] = "unknown"
        elif b["cls"] == VC.REJECT_ID:
            primary, _ids = CL.reject_color_analysis(img, b["bbox"])
            b["color"] = primary
        else:
            b["color"] = CL.mango_color_id(img, b["bbox"])
    return boxes


def find_image(images_dir, stem):
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        cand = os.path.join(images_dir, stem + ext)
        if os.path.exists(cand):
            return cand
    return None


def horizontal_pos_idx(xc):
    if xc < 0.40:
        return 0     # left
    if xc > 0.60:
        return 2     # right
    return 1         # center


# ---------------------------------------------------------------------------
# Geometri (IoU) untuk mencocokkan objek lokal gold <-> prediksi
# ---------------------------------------------------------------------------
def _xyxy(b):
    xc, yc, w, h = b
    return (xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2)


def iou(a, b):
    ax1, ay1, ax2, ay2 = _xyxy(a)
    bx1, by1, bx2, by2 = _xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


# ---------------------------------------------------------------------------
# Konstruksi vektor
# ---------------------------------------------------------------------------
def global_vec(boxes):
    v = np.zeros(cfg.GLOBAL_DIM, dtype=np.float32)
    if not boxes:
        return v
    # grade counts (4)
    for b in boxes:
        if b["cls"] in cfg.GRADE_IDS:
            v[cfg.GRADE_IDS.index(b["cls"])] += 1
    off = len(cfg.GRADE_IDS)
    # total (1)
    v[off] = len(boxes); off += 1
    # color counts (7)
    for b in boxes:
        c = b.get("color", "unknown")
        if c in cfg.COLOR_IDS:
            v[off + cfg.COLOR_IDS.index(c)] += 1
    off += len(cfg.COLOR_IDS)
    # area mean/max (2)
    areas = [b["area"] for b in boxes]
    v[off] = float(np.mean(areas)); v[off + 1] = float(np.max(areas)); off += 2
    # posisi frac L/C/R (3)
    pos = np.zeros(3, dtype=np.float32)
    for b in boxes:
        pos[horizontal_pos_idx(b["xc"])] += 1
    v[off:off + 3] = pos / len(boxes); off += 3
    # grade counts per-SISI: kiri (xc<0.5) lalu kanan (>=0.5) — untuk grades_on_side
    for b in boxes:
        if b["cls"] in cfg.GRADE_IDS:
            side = 0 if b["xc"] < cfg.SIDE_SPLIT else 1     # 0=kiri, 1=kanan
            v[off + side * len(cfg.GRADE_IDS) + cfg.GRADE_IDS.index(b["cls"])] += 1
    return v


def local_vec(box):
    v = np.zeros(cfg.LOCAL_DIM, dtype=np.float32)
    off = 0
    if box["cls"] in cfg.GRADE_IDS:
        v[cfg.GRADE_IDS.index(box["cls"])] = 1
    off += len(cfg.GRADE_IDS)
    c = box.get("color", "unknown")
    if c in cfg.COLOR_IDS:
        v[off + cfg.COLOR_IDS.index(c)] = 1
    off += len(cfg.COLOR_IDS)
    v[off + horizontal_pos_idx(box["xc"])] = 1
    off += 3
    v[off] = box["xc"]; off += 1
    v[off] = box["area"]
    return v


ZERO_LOCAL = np.zeros(cfg.LOCAL_DIM, dtype=np.float32)


def match_local(gold_bbox, pred_boxes):
    best, best_iou = None, 0.0
    for pb in pred_boxes:
        j = iou(gold_bbox, pb["bbox"])
        if j > best_iou:
            best, best_iou = pb, j
    return best if (best is not None and best_iou >= cfg.MATCH_IOU) else None


# ---------------------------------------------------------------------------
# Driver per split
# ---------------------------------------------------------------------------
def extract_split(model, split, lang):
    from collections import defaultdict
    with open(cfg.gold_path(split, lang), encoding="utf-8") as fh:
        gold = json.load(fh)

    by_image = defaultdict(list)
    for s in gold:
        by_image[s["image"]].append(s)

    images_dir = os.path.join(cfg.IMAGES_DIR, split)
    is_train = (split == "train")

    qids, feats, keeps = [], [], []
    n_missed = n_skip = 0

    for k, (img_name, qas) in enumerate(by_image.items(), 1):
        stem = os.path.splitext(img_name)[0]
        img_path = find_image(images_dir, stem)
        img = cv2.imread(img_path) if img_path else None
        pred = enrich(predict_boxes(model, img_path), img) if img_path else []
        gvec = global_vec(pred)

        for qa in qas:
            qid = qa["question_id"]
            if qa["level"] == "global":
                feat = np.concatenate([gvec, ZERO_LOCAL]); keep = True
            else:  # local
                mb = match_local(tuple(qa["bbox"]), pred)
                if mb is not None:
                    feat = np.concatenate([gvec, local_vec(mb)]); keep = True
                else:
                    n_missed += 1
                    if is_train:
                        keep = False; n_skip += 1
                        feat = np.concatenate([gvec, ZERO_LOCAL])
                    else:
                        feat = np.concatenate([gvec, ZERO_LOCAL]); keep = True
            qids.append(qid); feats.append(feat); keeps.append(keep)

        if k % 50 == 0:
            print(f"  [{split}] {k}/{len(by_image)} gambar diproses...")

    qids = np.array(qids, dtype=np.int64)
    feats = np.stack(feats).astype(np.float32)
    keeps = np.array(keeps, dtype=bool)
    out = os.path.join(cfg.CACHE_DIR, f"feat_{split}_{lang}.npz")
    np.savez_compressed(out, qids=qids, feats=feats, keeps=keeps)
    print(f"[{split}] disimpan -> {out} | n={len(qids)} "
          f"dim={feats.shape[1]} | local missed={n_missed} skip(train)={n_skip}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=cfg.SPLITS + ["all"], default="all")
    ap.add_argument("--lang", default=cfg.LANG)
    args = ap.parse_args()

    from ultralytics import YOLO
    print("Load YOLO:", cfg.MODEL_YOLO)
    model = YOLO(cfg.MODEL_YOLO)

    splits = cfg.SPLITS if args.split == "all" else [args.split]
    for sp in splits:
        extract_split(model, sp, args.lang)


if __name__ == "__main__":
    main()
