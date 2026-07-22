"""
MangoVQA generator (template-based) — question types defined in config.TEMPLATES,
covering local + global levels.

Converts the 4-class YOLO detection labels into Visual Question Answering
samples. Uniform schema: `answers` is ALWAYS a list of strings; `answer_type`
is "open_ended" (single) or "multi_label" (set). Composite "breakdown" types
encode structure as "key:count" strings so the schema stays uniform.

Levels:
  * global  -> reasoning over the whole image (counts, presence, breakdowns, ...)
  * local   -> per-object questions carrying object_id + bbox

Colour is derived from the fruit region via color.py (HSV, calibrated bins).

Usage:
    python vqa/generate_vqa.py --lang id
    python vqa/generate_vqa.py --lang en
"""

import argparse
import json
import os
from collections import Counter

import cv2

import config as C
import color as CL

NONE_TOKEN = {"id": "Tidak ada", "en": "None"}


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def parse_label_file(path):
    boxes = []
    if not os.path.exists(path):
        return boxes
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            p = line.split()
            if len(p) < 5:
                continue
            boxes.append({"cls": int(float(p[0])),
                          "bbox": tuple(round(float(v), 5) for v in p[1:5])})
    return boxes


def find_image(images_dir, stem):
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        cand = os.path.join(images_dir, stem + ext)
        if os.path.exists(cand):
            return cand
    return None


def horizontal_position(xc, lang):
    pos = C.POSITION[lang]
    if xc < 0.40:
        return pos["left"]
    if xc > 0.60:
        return pos["right"]
    return pos["center"]


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------
class QAWriter:
    def __init__(self):
        self.samples = []
        self._id = 0

    def add(self, meta, level, qtype, atype, question, answers,
            object_id=None, bbox=None):
        answers = [str(a) for a in answers]
        self.samples.append({
            "question_id": self._id,
            "image": meta["image"],
            "image_path": meta["image_path"],
            "split": meta["split"],
            "lang": meta["lang"],
            "level": level,
            "object_id": object_id,
            "bbox": list(bbox) if bbox is not None else None,
            "question": question,
            "question_type": qtype,
            "answer_type": atype,
            "answers": answers,
            "answer": answers[0] if atype == "open_ended" else None,
        })
        self._id += 1


# ---------------------------------------------------------------------------
# Per-image generation
# ---------------------------------------------------------------------------
def generate_for_image(meta, boxes, img, lang, w):
    T, GL = C.TEMPLATES[lang], C.GRADE_LABEL[lang]
    yn, none = C.YESNO[lang], NONE_TOKEN[lang]

    if not boxes:
        return

    # enrich boxes: obj id, colour id, xc, area
    for i, b in enumerate(boxes, 1):
        b["obj"] = f"obj_{i}"
        b["xc"] = b["bbox"][0]
        b["area"] = b["bbox"][2] * b["bbox"][3]
        if img is None:
            b["color"] = "unknown"
            b["color_list"] = ["unknown"]
        elif b["cls"] == C.REJECT_ID:
            # warna cacat sadar-kondisi + multi-warna per-piksel (mangga campur)
            primary, ids = CL.reject_color_analysis(img, b["bbox"])
            b["color"] = primary            # 1 warna primer utk agregasi global
            b["color_list"] = ids           # [dasar?, cacat?] utk object_color
        else:
            b["color"] = CL.mango_color_id(img, b["bbox"])
            b["color_list"] = [b["color"]]

    total = len(boxes)
    gcount = Counter(b["cls"] for b in boxes)
    ccount = Counter(b["color"] for b in boxes if b["color"] != "unknown")
    grades_present = [c for c in C.QUALITY_ORDER if gcount.get(c, 0) > 0]
    colors_present = [c for c in CL.COLOR_ORDER if ccount.get(c, 0) > 0]

    # ================= GLOBAL =================
    w.add(meta, "global", "count_total", "open_ended", T["count_total"], [total])

    for cid in C.QUALITY_ORDER:                       # per-grade count + existence
        n, g = gcount.get(cid, 0), GL[cid]
        w.add(meta, "global", "count_class", "open_ended",
              T["count_class"].format(grade=g), [n])
        w.add(meta, "global", "exist_class", "open_ended",
              T["exist_class"].format(grade=g), [yn[n > 0]])

    w.add(meta, "global", "which_grades", "multi_label", T["which_grades"],
          [GL[c] for c in grades_present])            # multi-answer

    top = max(gcount.values())                        # dominant (unique max only)
    leaders = [c for c in grades_present if gcount[c] == top]
    if len(leaders) == 1:
        w.add(meta, "global", "dominant", "open_ended", T["dominant"], [GL[leaders[0]]])

    w.add(meta, "global", "best_quality", "open_ended", T["best_quality"],
          [GL[grades_present[0]]])
    w.add(meta, "global", "worst_quality", "open_ended", T["worst_quality"],
          [GL[grades_present[-1]]])

    # LIMITATION (size): "largest" is approximated by 2D bounding-box area
    # (normalized w*h from the YOLO label), NOT true fruit size. There is no
    # depth cue, so a mango closer to the camera looks larger, and under
    # occlusion/stacking or edge cropping a box may shrink or overestimate.
    # Also, the axis-aligned box area is not the segmented fruit area: box
    # corners include background and equal-area boxes may bound different real
    # fruit sizes (a faithful cue would need a segmentation mask / extent, not
    # the detection box). Hence largest_grade is approximate; a symmetric
    # smallest_grade would be even less reliable (occluded/cropped fruit
    # yields small boxes).
    largest = max(boxes, key=lambda b: b["area"])
    w.add(meta, "global", "largest_grade", "open_ended", T["largest_grade"],
          [GL[largest["cls"]]])

    w.add(meta, "global", "overall", "open_ended", T["overall"],
          [yn[gcount.get(C.REJECT_ID, 0) == 0]])

    # colours
    w.add(meta, "global", "which_colors", "multi_label", T["which_colors"],
          [CL.color_name(c, lang) for c in colors_present] or [none])   # multi-answer
    for cid in CL.COLOR_ORDER:
        w.add(meta, "global", "count_color", "open_ended",
              T["count_color"].format(color=CL.color_name(cid, lang)),
              [ccount.get(cid, 0)])

    # grades on each side (multi-answer)
    for side in ("left", "right"):
        on = [b for b in boxes if (b["xc"] < 0.5) == (side == "left")]
        gp = [c for c in C.QUALITY_ORDER if any(b["cls"] == c for b in on)]
        w.add(meta, "global", "grades_on_side", "multi_label",
              T["grades_on_side"].format(side=C.SIDE[lang][side]),
              [GL[c] for c in gp] or [none])

    # composite breakdowns (multi-answer, "key:count")
    w.add(meta, "global", "grade_breakdown", "multi_label", T["grade_breakdown"],
          [f"{GL[c]}:{gcount[c]}" for c in grades_present])
    if colors_present:
        w.add(meta, "global", "color_breakdown", "multi_label", T["color_breakdown"],
              [f"{CL.color_name(c, lang)}:{ccount[c]}" for c in colors_present])

    # jumlah kategori BERBEDA (bukan jumlah mangga): grade & warna
    w.add(meta, "global", "count_grades", "open_ended", T["count_grades"],
          [len(grades_present)])
    w.add(meta, "global", "count_colors", "open_ended", T["count_colors"],
          [len(colors_present)])

    # komposit grade+warna per mangga untuk seluruh gambar (multi-answer, "grade:warna")
    gc_tokens = []
    for b in boxes:
        cnames = [CL.color_name(c, lang) for c in b.get("color_list", [b["color"]])
                  if c != "unknown"]
        gc_tokens.append(f"{GL[b['cls']]}:{'/'.join(cnames) if cnames else none}")
    w.add(meta, "global", "grade_color_global", "multi_label", T["grade_color_global"],
          gc_tokens)

    # ================= LOCAL (per object) =================
    for b in boxes:
        w.add(meta, "local", "object_grade", "open_ended", T["object_grade"],
              [GL[b["cls"]]], object_id=b["obj"], bbox=b["bbox"])
        clist = b.get("color_list", [b["color"]])
        if clist and clist[0] != "unknown":
            names = [CL.color_name(c, lang) for c in clist]
            atype = "multi_label" if len(names) > 1 else "open_ended"
            w.add(meta, "local", "object_color", atype, T["object_color"],
                  names, object_id=b["obj"], bbox=b["bbox"])
        w.add(meta, "local", "object_position", "open_ended", T["object_position"],
              [horizontal_position(b["xc"], lang)], object_id=b["obj"], bbox=b["bbox"])
        w.add(meta, "local", "object_marketable", "open_ended", T["object_marketable"],
              [yn[b["cls"] != C.REJECT_ID]], object_id=b["obj"], bbox=b["bbox"])
        # komposit warna+kualitas untuk satu mangga (warna dulu, lalu grade)
        cg_names = [CL.color_name(c, lang) for c in b.get("color_list", [b["color"]])
                    if c != "unknown"]
        w.add(meta, "local", "object_color_grade", "multi_label", T["object_color_grade"],
              cg_names + [GL[b["cls"]]], object_id=b["obj"], bbox=b["bbox"])


# ---------------------------------------------------------------------------
# Split driver
# ---------------------------------------------------------------------------
def build_split(data_root, split, lang, writer):
    images_dir = os.path.join(data_root, "images", split)
    labels_dir = os.path.join(data_root, "labels", split)
    n = 0
    if not os.path.isdir(labels_dir):
        return 0
    for fname in sorted(os.listdir(labels_dir)):
        if not fname.endswith(".txt"):
            continue
        stem = fname[:-4]
        img_path = find_image(images_dir, stem)
        if img_path is None:
            continue
        boxes = parse_label_file(os.path.join(labels_dir, fname))
        if not boxes:
            continue
        img = cv2.imread(img_path)
        meta = {"image": os.path.basename(img_path),
                "image_path": os.path.relpath(img_path).replace("\\", "/"),
                "split": split, "lang": lang}
        generate_for_image(meta, boxes, img, lang, writer)
        n += 1
    return n


def compute_stats(samples):
    return {
        "total_samples": len(samples),
        "by_split": dict(Counter(s["split"] for s in samples)),
        "by_level": dict(Counter(s["level"] for s in samples)),
        "by_answer_type": dict(Counter(s["answer_type"] for s in samples)),
        "by_question_type": dict(Counter(s["question_type"] for s in samples)),
        "multi_label_share": round(
            sum(s["answer_type"] == "multi_label" for s in samples) / max(1, len(samples)), 4),
    }


def main():
    n_types = len(C.TEMPLATES[C.DEFAULT_LANG])
    ap = argparse.ArgumentParser(
        description=f"MangoVQA generator ({n_types} question types)")
    ap.add_argument("--data-root", default="data/4-class")
    ap.add_argument("--out", default="data/vqa")
    ap.add_argument("--lang", default=C.DEFAULT_LANG, choices=["id", "en"])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    all_samples = []
    for split in ("train", "test", "val"):
        w = QAWriter()
        n = build_split(args.data_root, split, args.lang, w)
        if not w.samples:
            continue
        out = os.path.join(args.out, f"vqa_{split}_{args.lang}.json")
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(w.samples, fh, ensure_ascii=False, indent=2)
        print(f"[{split}] {n} images -> {len(w.samples)} QA  ({out})")
        all_samples.extend(w.samples)

    stats = compute_stats(all_samples)
    with open(os.path.join(args.out, f"vqa_stats_{args.lang}.json"), "w",
              encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)
    print("\n=== Summary ===")
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
