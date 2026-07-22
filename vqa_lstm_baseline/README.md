# VQA YOLO+LSTM — Baseline Pembanding (learned encoder)

Baseline *learned* untuk dibandingkan dengan metode **template rule-based** di paper
MangoVQA. Tujuannya menunjukkan trade-off "rule-based vs learned", menjawab kritik
"tidak ada pembanding", dan mengisi *future work* yang disebut di paper (§2.4/§5).

**Terisolasi**: folder ini tidak mengubah apa pun di `vqa/`. Hanya meng-`import`
`vqa/color.py` (ekstraksi HSV) & meniru setelan YOLO `vqa/eval_vqa.py` agar fitur
visual + metrik **identik** → perbandingan adil (yang beda murni template vs LSTM).

## Arsitektur
```
pertanyaan --emb--> LSTM --------\
                                  concat --> trunk(FC) --> kepala OE (softmax, 34)
fitur visual (33) --proj--------/                      \-> kepala ML (sigmoid, 12)
```
- Fitur visual 33-dim = concat(GLOBAL 17, LOKAL 16), dari deteksi YOLO train16
  (di-cache; YOLO beku). Global questions → bagian lokal nol.
- Routing kepala ditentukan `answer_type` (loss masking), bukan diprediksi model.
- Counting = klasifikasi angka-yang-pernah-dilihat (keterbatasan bawaan, sengaja
  ditunjukkan vs template yang menghitung langsung dari YOLO).

## Cara jalan (urут)
```bash
python build_vocab.py        # kamus: 54 kata, 34 jawaban OE, 12 label ML
python extract_features.py   # jalankan YOLO -> cache/feat_{split}_id.npz  (semua split)
python train.py              # training + early stopping -> cache/model.pt
python evaluate.py           # metrik test + tabel perbandingan -> cache/comparison_test_id.md
```

## Hasil (test split, id) — fitur sudah di-fair-kan (grade per-sisi)
- Rata-rata makro: **Template 0.917 vs YOLO+LSTM 0.912** (praktis SERI, +0.005).
- Pembeda jelas tersisa: `count_total` (Template 0.986 vs 0.944, +0.042) = exact
  counting titik lemah neural. `grades_on_side` kini seri (0.843 vs 0.839).
- Framing: template MENYAMAI baseline terlatih, tapi training-free + interpretable
  + unggul counting + baseline tak bisa emit per-label counts (composite breakdown).
- Detail: `cache/comparison_test_id.md`, `cache/baseline_eval_test_id.json`.

## File
| File | Peran |
|---|---|
| `cfg.py` | path, setelan YOLO (fair), hyperparameter, dimensi fitur |
| `build_vocab.py` | bangun 3 kamus dari train |
| `extract_features.py` | YOLO -> vektor visual 33-dim (cache) |
| `dataset.py` | Dataset PyTorch (tokenisasi + target one-hot/multi-hot) |
| `model.py` | LSTM + fusion + 2 kepala |
| `train.py` | loop training (loss masking) + validasi + early stop |
| `evaluate.py` | metrik test + tabel perbandingan template vs LSTM |

## Keterbatasan (jujur, untuk paper)
- Fitur global teragregasi kehilangan info gabungan sisi×grade → `grades_on_side`
  rendah (bisa diperbaiki dgn menambah hitungan grade per-sisi ke fitur).
- Tipe composite `grade/color_breakdown` dinilai level-label (baseline tak bisa
  keluarkan jumlah); di level "Label:N" template menang telak.
- CPU-only, YOLO train16 beku; angka dari `eval_results_test_id.json` (template).
