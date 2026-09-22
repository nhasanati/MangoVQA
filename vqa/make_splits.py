"""
Rebuild the MangoVQA train / val / test image partition from the Mango-YOLO
4-class dataset.

The detection repository (https://github.com/nhasanati/Mango-YOLO) releases the
4-class dataset as a single 70:30 partition: images/train (327) and images/test
(141). MangoVQA needs a validation set for the learned LSTM baseline, so the
141-image evaluation partition is further divided into val (70) and test (71).
The assignment is fixed by the file lists in data/splits/{train,val,test}.txt;
this script copies images and labels into that layout.

Usage (from the repository root):
    python vqa/make_splits.py --source /path/to/Mango-YOLO/data/4-class --out data/4-class
Afterwards run vqa/generate_vqa.py as usual.
"""
import argparse
import os
import shutil

IMG_EXTS = (".jpg", ".jpeg", ".png")
SPLITS = ("train", "val", "test")


def read_list(path):
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def index_source(source):
    """Map every image / label file name in the source dataset to its path."""
    images, labels = {}, {}
    for sub in ("train", "val", "test"):
        idir = os.path.join(source, "images", sub)
        ldir = os.path.join(source, "labels", sub)
        if os.path.isdir(idir):
            for fn in os.listdir(idir):
                if fn.lower().endswith(IMG_EXTS):
                    images[fn] = os.path.join(idir, fn)
        if os.path.isdir(ldir):
            for fn in os.listdir(ldir):
                if fn.lower().endswith(".txt"):
                    labels[fn] = os.path.join(ldir, fn)
    return images, labels


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="Mango-YOLO data/4-class directory (images/ + labels/)")
    ap.add_argument("--out", default="data/4-class", help="destination root (default: data/4-class)")
    ap.add_argument("--splits", default="data/splits", help="directory holding train.txt / val.txt / test.txt")
    ap.add_argument("--link", action="store_true", help="hard-link instead of copy (same drive only)")
    args = ap.parse_args()

    images, labels = index_source(args.source)
    if not images:
        raise SystemExit(f"no images found under {args.source}/images/")

    place = os.link if args.link else shutil.copy2
    missing = []
    for split in SPLITS:
        names = read_list(os.path.join(args.splits, f"{split}.txt"))
        idir = os.path.join(args.out, "images", split)
        ldir = os.path.join(args.out, "labels", split)
        os.makedirs(idir, exist_ok=True)
        os.makedirs(ldir, exist_ok=True)
        n_img = n_lbl = 0
        for name in names:
            stem = os.path.splitext(name)[0]
            if name in images:
                dst = os.path.join(idir, name)
                if not os.path.exists(dst):
                    place(images[name], dst)
                n_img += 1
            else:
                missing.append(name)
            lbl = stem + ".txt"
            if lbl in labels:
                dst = os.path.join(ldir, lbl)
                if not os.path.exists(dst):
                    place(labels[lbl], dst)
                n_lbl += 1
        print(f"[{split}] {n_img}/{len(names)} images, {n_lbl} labels -> {idir}")

    if missing:
        print(f"WARNING: {len(missing)} listed files not found in source, e.g. {missing[:3]}")
    else:
        print("done: 327 train / 70 val / 71 test expected")


if __name__ == "__main__":
    main()
