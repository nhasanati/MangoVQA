"""
HSV-based colour naming for mango bounding boxes (4-colour scheme, option B).

OpenCV HSV convention: H in [0,179], S,V in [0,255]. Mango skin colour shifts
green -> yellow-green -> yellow -> orange/red with ripeness; the dominant HUE of
the fruit region is the discriminative cue. Background (white/grey) has LOW
saturation and deep shadows/defects LOW value, so both are masked out.

Hue bins CALIBRATED on the 4-class dataset (n=745 bboxes) via calibrate_color.py.
Reference ranges (yellow ~20-40 OpenCV scale) follow mango ripeness colour studies
(e.g. Nandi et al., IEEE Sensors J., 2016).
"""

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 4-colour bins (option B). id -> (hue_low, hue_high). orange_red also wraps 160-179.
# ---------------------------------------------------------------------------
COLOR_BINS = [
    ("green",        41, 89),
    ("yellow_green", 33, 40),
    ("yellow",       20, 32),
    ("orange_red",    0, 19),
]
ORANGE_RED_WRAP = (160, 179)
# urutan listing stabil: kematangan (unripe->ripe) lalu warna cacat (Reject)
COLOR_ORDER = ["green", "yellow_green", "yellow", "orange_red", "coklat", "hitam", "abu"]

COLOR_LABEL = {
    "id": {"green": "Hijau", "yellow_green": "Kuning kehijauan",
           "yellow": "Kuning", "orange_red": "Oranye/Kemerahan",
           # warna cacat (khusus Reject, sadar-kondisi)
           "coklat": "Coklat (cacat)", "hitam": "Hitam/Gelap (cacat)",
           "abu": "Abu-abu (cacat)",
           "unknown": "Tidak diketahui"},
    "en": {"green": "Green", "yellow_green": "Yellow-green",
           "yellow": "Yellow", "orange_red": "Orange/Red",
           "coklat": "Brown (defect)", "hitam": "Black/Dark (defect)",
           "abu": "Grey (defect)",
           "unknown": "Unknown"},
}

# Fruit-pixel masking thresholds
SAT_MIN, VAL_MIN, VAL_MAX = 40, 40, 250
BBOX_SHRINK = 0.12  # use central (1-2*shrink) of the bbox to avoid edge background

# ---------------------------------------------------------------------------
# Warna cacat (defect) untuk kelas Reject — sadar-kondisi (dipicu grade Reject,
# bukan ambang murni, karena coklat-busuk & oranye-matang tumpang tindih di hue).
# Di dalam Reject, sub-kategori ditentukan oleh Value (gelap) & Saturation (pucat).
# ---------------------------------------------------------------------------
DEFECT_ORDER = ["coklat", "hitam", "abu"]
DEFECT_VAL_DARK = 70    # V median < ini -> hitam/gelap
DEFECT_SAT_PALE = 45    # S median < ini -> abu-abu


def crop_bbox(img_bgr, bbox_norm, shrink=BBOX_SHRINK):
    """Crop the central region of a YOLO-normalised bbox (xc,yc,w,h)."""
    h, w = img_bgr.shape[:2]
    xc, yc, bw, bh = bbox_norm
    x1 = (xc - bw / 2 + bw * shrink) * w
    x2 = (xc + bw / 2 - bw * shrink) * w
    y1 = (yc - bh / 2 + bh * shrink) * h
    y2 = (yc + bh / 2 - bh * shrink) * h
    x1, x2 = sorted((int(round(x1)), int(round(x2))))
    y1, y2 = sorted((int(round(y1)), int(round(y2))))
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, max(x2, x1 + 1)), min(h, max(y2, y1 + 1))
    return img_bgr[y1:y2, x1:x2]


def fruit_hue_pixels(crop_bgr):
    if crop_bgr.size == 0:
        return np.array([], dtype=np.uint8)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = (S >= SAT_MIN) & (V >= VAL_MIN) & (V <= VAL_MAX)
    hues = H[mask]
    if hues.size < 20:                      # too few -> relax to value-only mask
        hues = H[(V >= VAL_MIN) & (V <= VAL_MAX)]
    return hues


def dominant_hue(img_bgr, bbox_norm):
    """Median hue (OpenCV units) of the fruit region, or None."""
    hues = fruit_hue_pixels(crop_bbox(img_bgr, bbox_norm))
    return None if hues.size == 0 else float(np.median(hues))


def hue_to_color_id(hue):
    if hue is None:
        return "unknown"
    if hue >= ORANGE_RED_WRAP[0]:
        return "orange_red"
    for cid, lo, hi in COLOR_BINS:
        if lo <= hue <= hi:
            return cid
    nearest = min(COLOR_BINS, key=lambda b: abs(hue - (b[1] + b[2]) / 2))
    return nearest[0]


def color_name(cid, lang="id"):
    return COLOR_LABEL[lang].get(cid, COLOR_LABEL[lang]["unknown"])


def mango_color_id(img_bgr, bbox_norm):
    """bbox -> canonical colour id (green/yellow_green/yellow/orange_red)."""
    return hue_to_color_id(dominant_hue(img_bgr, bbox_norm))


def mango_color(img_bgr, bbox_norm, lang="id"):
    """bbox -> localized colour name."""
    return color_name(mango_color_id(img_bgr, bbox_norm), lang)


DEFECT_MIN_FRAC = 0.15   # fraksi minimal agar sebuah komponen (sehat/cacat) dilaporkan


def reject_color_analysis(img_bgr, bbox_norm):
    """Analisis warna buah Reject secara PER-PIKSEL (menangani mangga campur).

    Mangga Reject sering bercampur: sebagian kulit sehat (hijau/kuning) + sebagian
    area cacat (coklat/hitam/abu). Setiap piksel buah diklasifikasi:
      * gelap  (V < 70)                          -> cacat 'hitam'
      * pucat  (S < 45, V>=70)                    -> cacat 'abu'
      * coklat (hue<=25, S>=45, 70<=V<170)        -> cacat 'coklat' (busuk kecoklatan)
      * selain itu                               -> SEHAT (dipetakan ke bin kematangan)

    Kembalikan (primary_id, [color_ids...]):
      * primary_id : warna 1 komponen TERBESAR (untuk agregasi/penghitungan global).
      * color_ids  : daftar terurut [warna_dasar_sehat?, jenis_cacat?] untuk
                     pertanyaan per-objek (object_color). Komponen dimasukkan hanya
                     bila fraksinya >= DEFECT_MIN_FRAC.
    """
    crop = crop_bbox(img_bgr, bbox_norm)
    if crop.size == 0:
        return "coklat", ["coklat"]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    fruit = (V >= VAL_MIN) & (V <= VAL_MAX)
    n = int(fruit.sum())
    if n < 20:
        return "coklat", ["coklat"]

    dark = (V < DEFECT_VAL_DARK) & fruit
    pale = (S < DEFECT_SAT_PALE) & (V >= DEFECT_VAL_DARK) & fruit
    brown = (H <= 25) & (S >= DEFECT_SAT_PALE) & (V >= DEFECT_VAL_DARK) & (V < 170) & fruit
    defect = dark | pale | brown
    healthy = fruit & ~defect

    healthy_frac = healthy.sum() / n
    defect_frac = defect.sum() / n

    # warna dasar (bagian sehat) via median hue
    base_id = None
    if healthy.sum() >= 20:
        base_id = hue_to_color_id(float(np.median(H[healthy])))

    # jenis cacat = sub-kategori terbanyak di antara piksel cacat
    defect_id = None
    if defect.sum() > 0:
        counts = {"hitam": int(dark.sum()), "abu": int(pale.sum()), "coklat": int(brown.sum())}
        defect_id = max(counts, key=counts.get)

    # daftar untuk object_color (dasar lalu cacat), ambang fraksi
    ids = []
    if base_id is not None and healthy_frac >= DEFECT_MIN_FRAC:
        ids.append(base_id)
    if defect_id is not None and defect_frac >= DEFECT_MIN_FRAC:
        ids.append(defect_id)
    if not ids:                                   # fallback: pakai komponen terbesar
        ids = [defect_id or base_id or "coklat"]

    # primary = komponen area terbesar (untuk global counting yang tak ambigu)
    if defect_frac >= healthy_frac and defect_id is not None:
        primary = defect_id
    else:
        primary = base_id or defect_id or "coklat"
    return primary, ids
