"""
Evaluasi MangoVQA — Jalur A (grade / ground-truth) & Jalur B (warna / pseudo-label HSV).

Alur:
  * Gold answer  : dari data/vqa/vqa_{split}_{lang}.json (dibangkitkan dari label GT).
  * System answer: dari INFERENSI YOLOv11n (train16) pada citra uji -> kotak prediksi
                   -> templat QA yang sama (generate_vqa.generate_for_image untuk global,
                   perhitungan langsung untuk local).
  * Global : dicocokkan per (question_type, teks pertanyaan).
  * Local  : tiap objek gold dicocokkan ke kotak prediksi via IoU >= --iou;
             bila tak ada yang cocok -> dianggap salah (missed detection).

Metrik:
  * open_ended  -> Exact-Match / akurasi (per tipe, per jalur, per level).
  * multi_label -> Precision/Recall/F1 berbasis himpunan (micro + macro) + Subset-EM.

Pemisahan Jalur A vs B menjaga evaluasi tetap jujur: warna adalah pseudo-label HSV,
sehingga TIDAK digabung ke metrik utama grading.

Contoh:
    python vqa/eval_vqa.py --lang id --split test --conf 0.25 --iou 0.5
"""

import argparse
import json
import os
from collections import Counter, defaultdict

import cv2

import config as C
import color as CL
import generate_vqa as G

MODEL_PATH = r"C:\Users\Lenovo\yolo-project\yolov8-live\runs\detect\train16\weights\best.pt"

# Setelan inferensi HARUS sama dengan main5.py (inferensi resmi train16).
# imgsz=480 wajib sama dengan training; agnostic_nms buang box dobel lintas-kelas.
PRED_IMGSZ = 480
PRED_NMS_IOU = 0.55
PRED_AGNOSTIC = True
PRED_MAX_DET = 300

# Tipe pertanyaan yang termasuk Jalur B (warna). Sisanya = Jalur A (grade/GT).
COLOR_TYPES = {"which_colors", "count_color", "color_breakdown", "object_color",
               "count_colors"}

# Jalur C = komposit grade+warna (campur GT & pseudo-label HSV), dilaporkan terpisah.
COMPOSITE_TYPES = {"grade_color_global", "object_color_grade"}

LOCAL_TYPES = {"object_grade", "object_color", "object_position", "object_marketable",
               "object_color_grade"}


def track_of(qtype):
    if qtype in COMPOSITE_TYPES:
        return "C"
    return "B" if qtype in COLOR_TYPES else "A"


# ---------------------------------------------------------------------------
# Geometri
# ---------------------------------------------------------------------------
def xywhn_to_xyxy(b):
    xc, yc, w, h = b
    return (xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2)


def iou(a, b):
    ax1, ay1, ax2, ay2 = xywhn_to_xyxy(a)
    bx1, by1, bx2, by2 = xywhn_to_xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


# ---------------------------------------------------------------------------
# System answers
# ---------------------------------------------------------------------------
def predict_boxes(model, img_path, conf):
    """Inferensi -> list of {'cls':int, 'bbox':(xc,yc,w,h)} ternormalisasi.

    Setelan disamakan dengan main5.py: imgsz=480, iou=0.55, agnostic_nms=True.
    """
    res = model.predict(img_path, conf=conf, iou=PRED_NMS_IOU, imgsz=PRED_IMGSZ,
                        agnostic_nms=PRED_AGNOSTIC, max_det=PRED_MAX_DET,
                        verbose=False)[0]
    boxes = []
    if res.boxes is None:
        return boxes
    xywhn = res.boxes.xywhn.cpu().numpy()
    cls = res.boxes.cls.cpu().numpy().astype(int)
    for c, b in zip(cls, xywhn):
        boxes.append({"cls": int(c), "bbox": tuple(round(float(v), 5) for v in b)})
    return boxes


def system_global_lookup(meta, pred_boxes, img, lang):
    """Jalankan generator pada kotak prediksi; kembalikan lookup global + kotak terenrich."""
    w = G.QAWriter()
    # generate_for_image meng-enrich pred_boxes in-place (color, xc, area, obj)
    G.generate_for_image(meta, pred_boxes, img, lang, w)
    lut = {}
    for s in w.samples:
        if s["level"] == "global":
            lut[(s["question_type"], s["question"])] = (s["answer_type"], s["answers"])
    return lut


def system_local_answer(qtype, box, lang):
    """Jawaban sistem untuk satu pertanyaan lokal dari kotak prediksi yang cocok."""
    cls = box["cls"]
    if qtype == "object_grade":
        return [C.GRADE_LABEL[lang][cls]]
    if qtype == "object_color":
        return [CL.color_name(box.get("color", "unknown"), lang)]
    if qtype == "object_position":
        return [G.horizontal_position(box["xc"], lang)]
    if qtype == "object_marketable":
        return [C.YESNO[lang][cls != C.REJECT_ID]]
    if qtype == "object_color_grade":
        clist = box.get("color_list", [box.get("color", "unknown")])
        names = [CL.color_name(c, lang) for c in clist if c != "unknown"]
        return names + [C.GRADE_LABEL[lang][cls]]
    return [""]


# ---------------------------------------------------------------------------
# Metric accumulators
# ---------------------------------------------------------------------------
class Acc:
    """Akumulator metrik per kunci (mis. question_type)."""
    def __init__(self):
        # open_ended
        self.correct = 0
        self.total_oe = 0
        # multi_label micro
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.subset_ok = 0
        self.total_ml = 0
        self.answer_type = None

    def add_open(self, sys_ans, gold_ans):
        self.answer_type = "open_ended"
        self.total_oe += 1
        s = sys_ans[0].strip() if sys_ans else ""
        g = gold_ans[0].strip() if gold_ans else ""
        if s == g:
            self.correct += 1

    def add_multi(self, sys_ans, gold_ans):
        self.answer_type = "multi_label"
        self.total_ml += 1
        s, g = set(sys_ans), set(gold_ans)
        inter = len(s & g)
        self.tp += inter
        self.fp += len(s - g)
        self.fn += len(g - s)
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


def new_acc_dict():
    return defaultdict(Acc)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def evaluate(data_root, split, lang, conf, iou_thr, out_dir):
    from ultralytics import YOLO

    gold_path = os.path.join(out_dir, f"vqa_{split}_{lang}.json")
    with open(gold_path, encoding="utf-8") as fh:
        gold = json.load(fh)

    by_image = defaultdict(list)
    for s in gold:
        by_image[s["image"]].append(s)

    images_dir = os.path.join(data_root, "images", split)

    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print(f"Images: {len(by_image)} | gold QA: {len(gold)} | conf={conf} iou={iou_thr}\n")

    per_type = new_acc_dict()          # question_type -> Acc
    per_track_level = new_acc_dict()   # (track, level, answer_type) -> Acc  (agregat)
    n_pred_total = n_missed_local = 0

    for img_name, qas in by_image.items():
        stem = os.path.splitext(img_name)[0]
        img_path = G.find_image(images_dir, stem)
        if img_path is None:
            continue
        img = cv2.imread(img_path)
        meta = {"image": img_name, "image_path": img_path, "split": split, "lang": lang}

        pred_boxes = predict_boxes(model, img_path, conf)
        n_pred_total += len(pred_boxes)
        glut = system_global_lookup(meta, pred_boxes, img, lang)  # enriches pred_boxes

        for qa in qas:
            qtype = qa["question_type"]
            atype = qa["answer_type"]
            gold_ans = qa["answers"]
            track = track_of(qtype)
            level = qa["level"]

            if level == "global":
                sys_at, sys_ans = glut.get((qtype, qa["question"]), (atype, []))
            else:  # local: cocokkan bbox gold -> kotak prediksi via IoU
                gbb = tuple(qa["bbox"])
                best, best_iou = None, 0.0
                for pb in pred_boxes:
                    j = iou(gbb, pb["bbox"])
                    if j > best_iou:
                        best, best_iou = pb, j
                if best is not None and best_iou >= iou_thr:
                    sys_ans = system_local_answer(qtype, best, lang)
                else:
                    sys_ans = []           # missed detection -> salah
                    n_missed_local += 1

            # akumulasi
            key_lvl = (track, level, atype)
            if atype == "open_ended":
                per_type[qtype].add_open(sys_ans, gold_ans)
                per_track_level[key_lvl].add_open(sys_ans, gold_ans)
            else:
                per_type[qtype].add_multi(sys_ans, gold_ans)
                per_track_level[key_lvl].add_multi(sys_ans, gold_ans)

    # ---- rakit hasil ----
    def macro_f1(track, level):
        f1s = [per_type[qt].report()["f1"]
               for qt in per_type
               if track_of(qt) == track
               and per_type[qt].answer_type == "multi_label"
               and (level is None or _level_of(qt) == level)]
        return round(sum(f1s) / len(f1s), 4) if f1s else None

    results = {
        "config": {"split": split, "lang": lang, "conf": conf, "iou": iou_thr,
                   "model": MODEL_PATH, "n_pred_boxes": n_pred_total,
                   "n_missed_local": n_missed_local},
        "per_question_type": {qt: per_type[qt].report() for qt in sorted(per_type)},
        "aggregate": {},
    }
    for (track, level, atype), acc in sorted(per_track_level.items()):
        results["aggregate"].setdefault(f"Jalur_{track}", {}).setdefault(level, {})[atype] = acc.report()

    return results, per_type, per_track_level


def _level_of(qtype):
    return "local" if qtype in LOCAL_TYPES else "global"


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------
def print_report(results, per_type):
    cfg = results["config"]
    print("=" * 74)
    print(f"HASIL EVALUASI MangoVQA  |  split={cfg['split']}  lang={cfg['lang']}  "
          f"conf={cfg['conf']}  iou={cfg['iou']}")
    print(f"Kotak prediksi: {cfg['n_pred_boxes']} | objek lokal tak tercocok (missed): "
          f"{cfg['n_missed_local']}")
    print("=" * 74)

    for track, label in (("A", "JALUR A — GRADE / GROUND-TRUTH (utama)"),
                         ("B", "JALUR B — WARNA / PSEUDO-LABEL HSV (terpisah)"),
                         ("C", "JALUR C — KOMPOSIT GRADE+WARNA (campur, terpisah)")):
        print(f"\n### {label}")
        print(f"{'question_type':<20}{'lvl':<7}{'type':<12}{'n':>5}   metrik")
        print("-" * 74)
        for qt in sorted(per_type):
            if track_of(qt) != track:
                continue
            rep = per_type[qt].report()
            lvl = _level_of(qt)
            if rep["answer_type"] == "open_ended":
                m = f"acc={rep['accuracy']}"
            else:
                m = (f"P={rep['precision']} R={rep['recall']} "
                     f"F1={rep['f1']} subsetEM={rep['subset_em']}")
            print(f"{qt:<20}{lvl:<7}{rep['answer_type']:<12}{rep['n']:>5}   {m}")

    print("\n### AGREGAT per Jalur / level")
    print(json.dumps(results["aggregate"], ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Evaluasi MangoVQA (Jalur A & B)")
    ap.add_argument("--data-root", default="data/4-class")
    ap.add_argument("--out", default="data/vqa")
    ap.add_argument("--lang", default=C.DEFAULT_LANG, choices=["id", "en"])
    ap.add_argument("--split", default="test", choices=["test", "val", "train"])
    ap.add_argument("--conf", type=float, default=0.30,
                    help="Ambang confidence deteksi (default 0.30, sama dgn main5.py)")
    ap.add_argument("--iou", type=float, default=0.5,
                    help="Ambang IoU untuk MENCOCOKKAN objek lokal gold<->prediksi "
                         "(bukan NMS; NMS iou tetap 0.55 seperti main5.py)")
    args = ap.parse_args()

    results, per_type, _ = evaluate(args.data_root, args.split, args.lang,
                                    args.conf, args.iou, args.out)
    print_report(results, per_type)

    out_path = os.path.join(args.out, f"eval_results_{args.split}_{args.lang}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print(f"\nHasil rinci disimpan -> {out_path}")


if __name__ == "__main__":
    main()
