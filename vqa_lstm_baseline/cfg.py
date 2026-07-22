"""
Konfigurasi terpusat untuk BASELINE VQA YOLO+LSTM.

Baseline ini adalah PEMBANDING (learned encoder) untuk metode template rule-based
di paper MangoVQA. Kode di folder ini TIDAK mengubah apa pun di vqa/ — hanya
membaca/mengimpor color.py & eval metric agar fitur visual + metrik IDENTIK
(perbandingan adil: yang dibandingkan murni "template vs LSTM").

Semua path absolut supaya bisa dijalankan dari mana saja.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MANGO_ROOT = r"C:\Users\Lenovo\Mango-YOLO"
VQA_DIR    = os.path.join(MANGO_ROOT, "vqa")               # kode template lama (di-import, tak diubah)
DATA_VQA   = os.path.join(MANGO_ROOT, "data", "vqa")       # file vqa_{split}_{lang}.json
DATA_ROOT  = os.path.join(MANGO_ROOT, "data", "4-class")   # images/{split}/*.jpg
IMAGES_DIR = os.path.join(DATA_ROOT, "images")

HERE       = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR  = os.path.join(HERE, "cache")
VOCAB_PATH = os.path.join(CACHE_DIR, "vocab.json")

MODEL_YOLO = r"C:\Users\Lenovo\yolo-project\yolov8-live\runs\detect\train16\weights\best.pt"

# ---------------------------------------------------------------------------
# Setelan inferensi YOLO — HARUS sama dengan vqa/eval_vqa.py (fair!)
# ---------------------------------------------------------------------------
PRED_CONF     = 0.30
PRED_IMGSZ    = 480
PRED_NMS_IOU  = 0.55
PRED_AGNOSTIC = True
PRED_MAX_DET  = 300
MATCH_IOU     = 0.50      # cocokkan objek lokal gold<->prediksi (sama dgn eval)

# ---------------------------------------------------------------------------
# Bahasa & split
# ---------------------------------------------------------------------------
LANG   = "id"
SPLITS = ["train", "val", "test"]

# ---------------------------------------------------------------------------
# Ruang fitur visual (dari box hasil enrichment: cls, color, xc, area)
# Urutan grade mengikuti vqa/config.CLASS_NAMES: {0:Class1,1:Class2,2:Extra,3:Reject}
# Urutan warna mengikuti vqa/color.COLOR_ORDER (7 warna).
# ---------------------------------------------------------------------------
GRADE_IDS = [0, 1, 2, 3]                                    # Class1, Class2, Extra, Reject
COLOR_IDS = ["green", "yellow_green", "yellow", "orange_red", "coklat", "hitam", "abu"]

# Vektor GLOBAL (agregasi semua box) : counts grade(4) + total(1) + counts warna(7)
#                                      + area mean/max(2) + posisi frac L/C/R(3)
#                                      + grade counts per-SISI kiri(4) + kanan(4) = 25
#   Bagian per-sisi (batas xc<0.5 kiri, >=0.5 kanan — SAMA dgn generate_vqa
#   grades_on_side) memberi baseline info sisi×grade agar 'grades_on_side' FAIR.
# Vektor LOKAL (per objek)           : onehot grade(4) + onehot warna(7)
#                                      + posisi onehot L/C/R(3) + xc(1) + area(1) = 16
# Vektor visual final = concat(GLOBAL 25, LOKAL 16) = 41.
#   - pertanyaan global : bagian lokal = nol.
#   - pertanyaan lokal  : dua bagian terisi (objek + konteks global).
SIDE_SPLIT = 0.5                                            # batas sisi (generate_vqa)
GLOBAL_DIM = len(GRADE_IDS) + 1 + len(COLOR_IDS) + 2 + 3 + 2 * len(GRADE_IDS)   # 25
LOCAL_DIM  = len(GRADE_IDS) + len(COLOR_IDS) + 3 + 1 + 1    # 16
VISUAL_DIM = GLOBAL_DIM + LOCAL_DIM                         # 41

# ---------------------------------------------------------------------------
# Hyperparameter model & training
# ---------------------------------------------------------------------------
EMB_DIM     = 128        # dimensi embedding kata
LSTM_HID    = 128        # dimensi hidden LSTM (vektor pertanyaan)
LSTM_LAYERS = 1
FUSION_HID  = 256        # lebar trunk FC setelah fusion
DROPOUT     = 0.3

BATCH_SIZE  = 64
EPOCHS      = 40
LR          = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE    = 6          # early stopping
ML_THRESHOLD = 0.5       # ambang sigmoid utk multi_label
SEED        = 42

# ---------------------------------------------------------------------------
# Tipe pertanyaan: warna (Track B) — sama dengan eval_vqa (untuk pelaporan terpisah)
# ---------------------------------------------------------------------------
COLOR_TYPES = {"which_colors", "count_color", "color_breakdown", "object_color"}
LOCAL_TYPES = {"object_grade", "object_color", "object_position", "object_marketable"}


def gold_path(split, lang=LANG):
    return os.path.join(DATA_VQA, f"vqa_{split}_{lang}.json")


def track_of(qtype):
    return "B" if qtype in COLOR_TYPES else "A"


def level_of(qtype):
    return "local" if qtype in LOCAL_TYPES else "global"
