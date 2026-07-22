# Mango Quality VQA Dataset Generator

Membangun dataset **Visual Question Answering (VQA)** untuk penilaian kualitas
mangga dari anotasi deteksi objek YOLO (dataset **4-class**). Berbasis
*template* (rule-based) — jawaban diturunkan langsung dari bounding box,
sehingga **100% akurat** tanpa anotasi manual.

## Kelas / Grade Mutu
Urutan mutu (terbaik → terburuk): `Extra Class > Class 1 > Class 2 > Reject`

| id | Grade       | Catatan          |
|----|-------------|------------------|
| 0  | Class 1     | mayoritas        |
| 1  | Class 2     |                  |
| 2  | Extra Class | grade tertinggi  |
| 3  | Reject      | rare-class       |

## Cara Menjalankan
```bash
python vqa/generate_vqa.py                 # 4-class, Bahasa Indonesia
python vqa/generate_vqa.py --lang en       # pertanyaan Bahasa Inggris
python vqa/generate_vqa.py --data-root data/4-class --out data/vqa
```

Output ke `data/vqa/`:
- `vqa_train_<lang>.json` — Q&A split train
- `vqa_test_<lang>.json`  — Q&A split test (dari images/test)
- `vqa_stats_<lang>.json` — statistik distribusi

## Format Sampel
```json
{
  "question_id": 42,
  "image": "1ab58976-harum_manis_2.jpg",
  "image_path": "data/4-class/images/train/1ab58976-harum_manis_2.jpg",
  "split": "train",
  "question": "Berapa banyak mangga dengan grade Class 1?",
  "question_type": "count_class",
  "answer_type": "open_ended",
  "answers": ["13"],
  "answer": "13"
}
```

- `answer_type: "open_ended"` → tepat satu jawaban pendek (`answers` panjang 1,
  di-mirror ke field `answer`).
- `answer_type: "multi_label"` → himpunan jawaban benar (`answers` bisa >1,
  field `answer` = `null`).

## Tipe Pertanyaan (10 jenis)
| question_type   | answer_type | Contoh jawaban          |
|-----------------|-------------|-------------------------|
| count_total     | open_ended  | `"15"`                  |
| count_class     | open_ended  | `"13"` / `"0"`          |
| exist_class     | open_ended  | `"Ya"` / `"Tidak"`      |
| dominant        | open_ended  | `"Class 1"`             |
| best_quality    | open_ended  | `"Extra Class"`         |
| worst_quality   | open_ended  | `"Reject"`              |
| largest_grade   | open_ended  | grade bbox terbesar     |
| location        | open_ended  | `"kiri"/"tengah"/"kanan"` |
| overall         | open_ended  | `"Ya"` / `"Tidak"` (layak jual) |
| which_grades    | multi_label | `["Class 1","Class 2"]` |

## Catatan Desain
- `count_class` & `exist_class` ditanyakan untuk **semua** grade (termasuk yang
  tidak ada) agar jawaban `"0"` / `"Tidak"` ikut terepresentasi → distribusi
  jawaban lebih seimbang untuk training.
- `location` hanya dibuat bila grade tsb muncul **tepat satu** kali (posisi
  tidak ambigu).
- `largest_grade` memakai luas bbox ternormalisasi (`w*h`).
- Konfigurasi kelas, urutan mutu, dan template terpusat di `config.py` —
  mudah menambah tipe pertanyaan atau bahasa baru.
