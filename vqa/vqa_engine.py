"""
vqa_engine.py — Mesin MangoVQA (berbasis templat) dengan dua jalur pertanyaan.

Modul ini menyatukan seluruh pipeline penjawaban ke dalam satu API yang jelas dan
BISA DITELUSURI, sehingga setiap langkah dapat dirujuk saat penulisan paper.

===========================================================================
ALGORITMA
===========================================================================
Masukan : citra I, bahasa L, ambang confidence c, dan pertanyaan q.
Keluaran: jawaban a beserta jejak penalaran (trace).

  Tahap T1 — Deteksi objek.
      D <- YOLOv11n(train16).predict(I; imgsz=480, conf=c, iou=0.55, agnostic)
      D = { (kelas_k, bbox_k, skor_k) }  untuk setiap buah k.

  Tahap T2 — Konstruksi answer-bank (template answering).
      Untuk objek D dan citra I, jalankan aturan templat (generate_vqa +
      color.py) sehingga terbentuk himpunan pasangan (pertanyaan, jawaban)
      B = { (tipe, teks_pertanyaan, jawaban, level, slot) }.
      B disebut "answer-bank": seluruh jawaban yang DAPAT dijawab sistem atas I.

  Tahap T3 — Routing pertanyaan (dua jalur, satu mesin):
      (a) JALUR TEMPLAT  [closed vocabulary]:
          pengguna memilih pertanyaan p dari daftar -> ambil (p) langsung dari B.
      (b) JALUR TERBUKA  [open free-text]:
          q -> parsing (qmatch): deteksi slot (grade/warna/sisi/objek) +
          deteksi intent -> tipe pertanyaan t -> ambil sampel B yang bertipe t
          dengan slot yang sesuai.

  Tahap T4 — Penyajian.
      Kembalikan jawaban a = jawaban(B, ...) beserta trace langkah T3.
===========================================================================

Poin kunci untuk paper: KEDUA jalur bermuara pada answer-bank B yang SAMA
(dibangun oleh mesin templat yang sama seperti gold QA). Jalur terbuka hanya
menambah lapisan pemetaan bahasa -> tipe; ia TIDAK mengubah cara menjawab,
sehingga konsistensi dengan dataset gold tetap terjaga.
"""
from dataclasses import dataclass, field
from typing import Optional
from collections import Counter

import config as C
import color as CL
import generate_vqa as G
import qmatch as QM

MODEL_PATH = r"C:\Users\Lenovo\yolo-project\yolov8-live\runs\detect\train16\weights\best.pt"

# Setelan inferensi — WAJIB sama dengan main5.py / eval_vqa.py.
PRED_IMGSZ = 480
PRED_NMS_IOU = 0.55
PRED_AGNOSTIC = True
PRED_MAX_DET = 300


# ---------------------------------------------------------------------------
# Struktur data hasil
# ---------------------------------------------------------------------------
@dataclass
class Analysis:
    """Hasil Tahap T1+T2 untuk sebuah citra."""
    boxes: list                      # [{cls,bbox}] hasil deteksi
    answer_bank: list                # samples (pertanyaan+jawaban) hasil template answering
    plot_bgr: object                 # citra beranotasi (BGR) untuk ditampilkan
    lang: str

    def detection_summary(self):
        counts = Counter(C.GRADE_LABEL[self.lang][b["cls"]] for b in self.boxes)
        return {"total": len(self.boxes), "per_grade": dict(counts)}


@dataclass
class Answer:
    """Hasil Tahap T3+T4 untuk satu pertanyaan."""
    ok: bool
    route: str                        # "template" | "open"
    question: str = ""
    answer: str = ""
    question_type: str = ""
    level: str = ""
    answer_type: str = ""
    trace: dict = field(default_factory=dict)   # jejak langkah (untuk paper/demo)
    message: str = ""                 # pesan bila tidak terjawab


# ---------------------------------------------------------------------------
# Mesin
# ---------------------------------------------------------------------------
class MangoVQAEngine:
    def __init__(self, model_path=MODEL_PATH):
        from ultralytics import YOLO
        self.model = YOLO(model_path)

    # ---- Tahap T1: deteksi ----
    def _detect(self, img_bgr, conf):
        res = self.model.predict(img_bgr, conf=conf, iou=PRED_NMS_IOU, imgsz=PRED_IMGSZ,
                                 agnostic_nms=PRED_AGNOSTIC, max_det=PRED_MAX_DET,
                                 verbose=False)[0]
        boxes = []
        if res.boxes is not None:
            xywhn = res.boxes.xywhn.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy().astype(int)
            for k, b in zip(cls, xywhn):
                boxes.append({"cls": int(k), "bbox": tuple(round(float(v), 5) for v in b)})
        return boxes, res

    # ---- Tahap T1+T2: analisis citra -> answer-bank ----
    def analyze(self, img_bgr, lang="id", conf=0.30) -> Analysis:
        """Jalankan deteksi (T1) lalu template answering (T2)."""
        boxes, res = self._detect(img_bgr, conf)
        w = G.QAWriter()
        meta = {"image": "input", "image_path": "input", "split": "input", "lang": lang}
        if boxes:
            G.generate_for_image(meta, boxes, img_bgr, lang, w)
        return Analysis(boxes=boxes, answer_bank=w.samples, plot_bgr=res.plot(), lang=lang)

    # ---- Aturan khusus: grade dominan dengan penanganan SERI ----
    def _dominant_answer(self, analysis: Analysis):
        """Kembalikan (teks_jawaban, is_tie). Saat seri -> sebut semua co-leaders.

        Aturan tie-break: TIDAK ada (definisi 'paling banyak' bersifat berdasarkan
        jumlah); bila jumlah teratas sama, keduanya dianggap sama-sama terbanyak.
        """
        gc = Counter(b["cls"] for b in analysis.boxes)
        if not gc:
            return None, False
        top = max(gc.values())
        leaders = [c for c in C.QUALITY_ORDER if gc.get(c, 0) == top]
        labels = [C.GRADE_LABEL[analysis.lang][c] for c in leaders]
        if len(leaders) == 1:
            return labels[0], False
        joined = ", ".join(labels)
        if analysis.lang == "id":
            return f"Seri — {joined} (masing-masing {top})", True
        return f"Tie — {joined} ({top} each)", True

    # ---- Tahap T3(a): JALUR TEMPLAT ----
    def ask_template(self, analysis: Analysis, sample_index: int) -> Answer:
        """Pengguna memilih pertanyaan dari daftar (indeks pada answer-bank)."""
        s = analysis.answer_bank[sample_index]
        return Answer(
            ok=True, route="template",
            question=s["question"], answer=", ".join(s["answers"]),
            question_type=s["question_type"], level=s["level"],
            answer_type=s["answer_type"],
            trace={"T3": "jalur templat: pilih langsung dari answer-bank",
                   "sample_index": sample_index},
        )

    # ---- Tahap T3(b): JALUR TERBUKA ----
    def ask_open(self, analysis: Analysis, text: str) -> Answer:
        """Pengguna mengetik pertanyaan bebas -> qmatch -> answer-bank."""
        s, f = QM.match_question(text, analysis.lang, analysis.answer_bank)
        fit = f["P3_fitur"]
        qtype = f["P4_intent"]
        trace = {
            "P1_normalisasi": f["P1_normalisasi"],
            "P2_token": f["P2_token"],
            "P3_slot": fit["slot"],
            "P3_kata_tanya": fit["kata_tanya"],
            "P3_isyarat_aktif": fit["isyarat_aktif"],
            "P3_lokal": fit["lokal"],
            "P4_intent(tipe)": qtype,
            "T3_pengambilan": "ambil sampel bertipe & berslot sesuai dari answer-bank",
        }
        if qtype is None:
            return Answer(ok=False, route="open", trace=trace,
                          message="Maaf, pertanyaan tidak dikenali. Coba lebih spesifik "
                                  "(mis. sebut 'grade', 'warna', 'jumlah', atau nama kelas).")
        if s is None:
            # Kasus khusus 'dominant' saat SERI: gold sengaja mengosongkan (definisi
            # 'strictly most frequent'), tetapi engine memberi jawaban co-leaders.
            if qtype == "dominant":
                dom, tie = self._dominant_answer(analysis)
                if dom is not None:
                    trace["T3.5_penanganan_seri"] = ("jumlah grade teratas sama (seri); "
                                                     "jawab seluruh co-leaders")
                    return Answer(ok=True, route="open", question_type="dominant",
                                  question=("Grade mutu apa yang paling banyak dalam gambar?"
                                            if analysis.lang == "id"
                                            else "Which quality grade is the most frequent?"),
                                  answer=dom, level="global",
                                  answer_type="multi_label" if tie else "open_ended",
                                  trace=trace)
            # intent dikenali tetapi memang tak berlaku untuk gambar ini
            return Answer(ok=False, route="open", question_type=qtype, trace=trace,
                          message=f"Pertanyaan dikenali sebagai '{qtype}', tetapi tidak "
                                  f"berlaku untuk gambar ini.")
        return Answer(
            ok=True, route="open",
            question=s["question"], answer=", ".join(s["answers"]),
            question_type=s["question_type"], level=s["level"],
            answer_type=s["answer_type"], trace=trace,
        )
