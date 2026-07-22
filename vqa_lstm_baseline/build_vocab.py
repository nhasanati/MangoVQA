"""
Bangun 3 kamus untuk baseline, HANYA dari split TRAIN (hindari kebocoran):

  1. word2idx  : kata pertanyaan -> indeks   (untuk embedding + LSTM)
  2. oe2idx    : jawaban open_ended -> indeks (kepala SOFTMAX)
  3. ml2idx    : label multi_label kanonik -> indeks (kepala SIGMOID)

Catatan desain (penting untuk paper):
  * Kepala open_ended memakai SEMUA jawaban open_ended yang muncul di train,
    TERMASUK angka hitungan. Baseline neural memperlakukan counting sebagai
    klasifikasi kelas-yang-pernah-dilihat -> ini keterbatasan bawaan yang justru
    ingin kita tunjukkan vs metode template (yang menghitung langsung dari YOLO).
  * Label multi_label dikanonikkan dengan split(':')[0] (buang bagian ":N" pada
    grade_breakdown/color_breakdown) sehingga kepala sigmoid belajar KEANGGOTAAN
    label, bukan menghafal pasangan label:jumlah yang jarang.

Jalankan:  python build_vocab.py
"""

import json
import os
import re

import cfg

PAD, UNK = "<PAD>", "<UNK>"

_token_re = re.compile(r"[a-z0-9]+")


def tokenize(text):
    """Tokenisasi sederhana: lowercase + ambil token alfanumerik.

    Pertanyaan bersifat template sehingga kosakata kecil & stabil.
    """
    return _token_re.findall(text.lower())


def build(lang=cfg.LANG):
    with open(cfg.gold_path("train", lang), encoding="utf-8") as fh:
        data = json.load(fh)

    words, oe, ml = set(), set(), set()
    for x in data:
        for tok in tokenize(x["question"]):
            words.add(tok)
        if x["answer_type"] == "open_ended":
            oe.add(str(x["answer"]))
        else:
            for a in x["answers"]:
                ml.add(a.split(":")[0])          # kanonik: buang ":N"

    word2idx = {PAD: 0, UNK: 1}
    for w in sorted(words):
        word2idx[w] = len(word2idx)

    # UNK di indeks 0 kepala OE menampung jawaban asing di val/test
    oe2idx = {UNK: 0}
    for a in sorted(oe):
        oe2idx[a] = len(oe2idx)

    ml2idx = {a: i for i, a in enumerate(sorted(ml))}

    vocab = {
        "lang": lang,
        "word2idx": word2idx,
        "oe2idx": oe2idx,
        "ml2idx": ml2idx,
        "meta": {
            "n_words": len(word2idx),
            "n_oe": len(oe2idx),
            "n_ml": len(ml2idx),
        },
    }
    os.makedirs(cfg.CACHE_DIR, exist_ok=True)
    with open(cfg.VOCAB_PATH, "w", encoding="utf-8") as fh:
        json.dump(vocab, fh, ensure_ascii=False, indent=2)
    return vocab


def load():
    with open(cfg.VOCAB_PATH, encoding="utf-8") as fh:
        return json.load(fh)


if __name__ == "__main__":
    v = build()
    m = v["meta"]
    print("Vocab tersimpan ->", cfg.VOCAB_PATH)
    print(f"  kata pertanyaan (word2idx) : {m['n_words']}  (termasuk <PAD>,<UNK>)")
    print(f"  jawaban open_ended (oe2idx): {m['n_oe']}  (termasuk <UNK>)")
    print(f"  label multi_label (ml2idx) : {m['n_ml']}")
    print("\n  Contoh oe2idx:", list(v["oe2idx"].items())[:8], "...")
    print("  ml2idx        :", v["ml2idx"])
