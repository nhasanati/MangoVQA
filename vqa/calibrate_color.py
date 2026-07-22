"""
Calibrate mango colour thresholds on the 4-class dataset.

For every mango bbox in the dataset it computes the fruit-region median hue,
then reports the hue distribution overall and per grade, and plots histograms
so the COLOR_BINS in color.py can be set from data (not guessed).

Run:
    python vqa/calibrate_color.py
"""
import os, glob
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
import color as CL

DATA_ROOT = os.path.join("data", "4-class")
SPLITS = ["train", "test", "val"]
GRADE_COLORS = {0: "#4C9F70", 1: "#E1B12C", 2: "#2E86DE", 3: "#8E44AD"}


def find_image(images_dir, stem):
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        p = os.path.join(images_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None


def collect():
    """Return dict grade_id -> list of median hues, and a flat list."""
    per_grade = {cid: [] for cid in C.CLASS_NAMES}
    n_box = n_img = 0
    for split in SPLITS:
        labels_dir = os.path.join(DATA_ROOT, "labels", split)
        images_dir = os.path.join(DATA_ROOT, "images", split)
        if not os.path.isdir(labels_dir):
            continue
        for lbl in sorted(glob.glob(os.path.join(labels_dir, "*.txt"))):
            stem = os.path.splitext(os.path.basename(lbl))[0]
            img_path = find_image(images_dir, stem)
            if img_path is None:
                continue
            img = cv2.imread(img_path)
            if img is None:
                continue
            n_img += 1
            with open(lbl, encoding="utf-8") as fh:
                for line in fh:
                    p = line.split()
                    if len(p) < 5:
                        continue
                    cls = int(float(p[0]))
                    bbox = tuple(float(v) for v in p[1:5])
                    hue = CL.dominant_hue(img, bbox)
                    if hue is not None:
                        per_grade.setdefault(cls, []).append(hue)
                        n_box += 1
    return per_grade, n_img, n_box


def pct(a):
    a = np.asarray(a)
    return {q: round(float(np.percentile(a, q)), 1) for q in (5, 25, 50, 75, 95)}


def main():
    per_grade, n_img, n_box = collect()
    allh = [h for v in per_grade.values() for h in v]

    print(f"Images read: {n_img} | bboxes with hue: {n_box}\n")
    print(f"{'Grade':<14}{'n':>6}   hue percentiles (OpenCV 0-179)  [p5/p25/p50/p75/p95]")
    for cid, name in C.CLASS_NAMES.items():
        v = per_grade.get(cid, [])
        if v:
            q = pct(v)
            print(f"{name:<14}{len(v):>6}   {q[5]:>5} {q[25]:>5} {q[50]:>5} {q[75]:>5} {q[95]:>5}")
        else:
            print(f"{name:<14}{0:>6}   (none)")
    q = pct(allh)
    print(f"{'ALL':<14}{len(allh):>6}   {q[5]:>5} {q[25]:>5} {q[50]:>5} {q[75]:>5} {q[95]:>5}\n")

    # colour-name distribution under current default bins
    from collections import Counter
    dist = Counter(CL.color_name(CL.hue_to_color_id(h), "id") for h in allh)
    print("Colour distribution under current default bins:")
    for k, v in dist.most_common():
        print(f"  {k:<18}{v:>5}  ({100*v/len(allh):.1f}%)")

    # ---- plots ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bins = np.arange(0, 100, 2)
    axes[0].hist(allh, bins=bins, color="#7f8c8d", edgecolor="white")
    for lo, hi, lab in [(0,19,"orange/red"),(20,32,"yellow"),(33,40,"y-green"),(41,89,"green")]:
        axes[0].axvspan(lo, hi, alpha=0.08)
        axes[0].text((lo+hi)/2, axes[0].get_ylim()[1]*0.92, lab, ha="center", fontsize=8)
    axes[0].set_title("Median hue of all mango bboxes"); axes[0].set_xlabel("Hue (OpenCV 0-179)")
    for cid, name in C.CLASS_NAMES.items():
        v = per_grade.get(cid, [])
        if v:
            axes[1].hist(v, bins=bins, histtype="step", linewidth=2,
                         color=GRADE_COLORS.get(cid), label=f"{name} (n={len(v)})")
    axes[1].set_title("Hue distribution per grade"); axes[1].set_xlabel("Hue (OpenCV 0-179)")
    axes[1].legend()
    out = os.path.join("vqa", "color_calibration.png")
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print(f"\nHistogram saved -> {out}")


if __name__ == "__main__":
    main()
