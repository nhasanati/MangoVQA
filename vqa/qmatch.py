"""
Pencocok pertanyaan bebas -> tipe templat MangoVQA (rule-based, ID + EN).

Sistem MangoVQA berbasis templat, sehingga teks bebas dari pengguna dipetakan ke
salah satu dari 18 tipe pertanyaan beserta slotnya (grade / warna / sisi / objek).

===========================================================================
PIPELINE PEMROSESAN KALIMAT (dapat dikutip di paper)
===========================================================================
  P1. Normalisasi   : NFKD -> buang aksen -> huruf kecil -> buang tanda baca.
  P2. Tokenisasi    : pisah berdasarkan spasi menjadi daftar token.
  P3. Fiturisasi    : ekstraksi (a) SLOT  = grade/warna/sisi/objek, dan
                      (b) ISYARAT LEKSIKAL = kelompok kata kunci yang muncul,
                      termasuk KATA TANYA:
                          "apa"    (what/which) -> menanya identitas (grade/warna apa)
                          "berapa" (how many)   -> menanya kuantitas (hitung)
                          "apakah" (yes/no)     -> menanya keberadaan/kelayakan
  P4. Klasifikasi   : aturan (rule-based) memetakan fitur -> 1 dari 18 tipe.

Fungsi utama:
  extract_features(text) -> dict pipeline P1..P4 (untuk telusur/paper)
  match_question(text, lang, samples) -> (sample | None, features)
  explain(text) -> cetak pipeline untuk demonstrasi
===========================================================================
"""
import re
import unicodedata

import config as C
import color as CL


def normalize(text):
    """P1 — Normalisasi teks: NFKD, buang aksen, huruf kecil, buang tanda baca."""
    s = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9\s]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def tokenize(norm):
    """P2 — Tokenisasi: pisah string ternormalisasi berdasarkan spasi."""
    return norm.split()


_norm = normalize   # alias kompatibilitas


# ---------------------------------------------------------------------------
# Deteksi slot
# ---------------------------------------------------------------------------
def detect_grade(t):
    """Kembalikan class id (0..3) atau None."""
    if re.search(r"\bextra\b|\bsuper\b", t):
        return 2
    if re.search(r"\breject\b|\btolak\b|\bafkir\b", t):
        return 3
    if re.search(r"\bclass\s*(2|ii)\b|\bkelas\s*(2|ii)\b|\bgrade\s*2\b", t):
        return 1
    if re.search(r"\bclass\s*(1|i)\b|\bkelas\s*(1|i)\b|\bgrade\s*1\b", t):
        return 0
    return None


def detect_color(t):
    """Kembalikan color id atau None (termasuk warna cacat Reject: coklat/hitam/abu)."""
    # --- warna cacat (Reject) --- dicek lebih dulu
    if re.search(r"abu[\s-]*abu|\babu\b|keabuan|\bgrey\b|\bgray\b", t):
        return "abu"
    if re.search(r"hitam|gelap|\bdark\b|\bblack\b", t):
        return "hitam"
    if re.search(r"coklat|cokelat|kecoklatan|\bbrown\b", t):
        return "coklat"
    # --- warna kematangan ---
    if re.search(r"kuning\s*kehijauan|yellow[\s-]*green|hijau\s*kekuningan", t):
        return "yellow_green"
    if re.search(r"oranye|orange|merah|kemerahan|\bred\b", t):
        return "orange_red"
    if re.search(r"kuning|yellow", t):
        return "yellow"
    if re.search(r"hijau|green", t):
        return "green"
    return None


def detect_side(t):
    if re.search(r"\bkiri\b|\bleft\b", t):
        return "left"
    if re.search(r"\bkanan\b|\bright\b", t):
        return "right"
    return None


def detect_object(t):
    """Cari 'obj_2', 'objek 2', 'object 3', 'mangga ke-2', dst -> 'obj_N' atau None."""
    m = re.search(r"obj[\s_]*(\d+)|objek\s*(\d+)|object\s*(\d+)|mangga\s*(?:ke[\s-]*)?(\d+)|nomor\s*(\d+)|no\.?\s*(\d+)", t)
    if m:
        num = next(g for g in m.groups() if g)
        return f"obj_{num}"
    return None


# ---------------------------------------------------------------------------
# Kosakata terkontrol (kelompok sinonim). Dipakai konsisten di seluruh aturan
# sehingga mudah ditelusuri & dijelaskan di paper.
# ---------------------------------------------------------------------------
GRADE_WORDS    = ("grade", "mutu", "kualitas", "kelas", "quality", "class")  # konsep grade
COUNT_WORDS    = ("berapa", "jumlah", "banyak", "how many", "count", "total")  # kuantitas
EACH_WORDS     = ("masing", "each", "breakdown", "rincian", "rinci", "per grade", "per kelas")
WHICH_WORDS    = ("apa saja", "apa aja", "apasaja", "which", "sebutkan", "list", "apakah saja")
COLOR_WORDS    = ("warna", "color", "colour")
MARKET_WORDS   = ("layak", "jual", "marketable", "sellable", "layak jual")
EXIST_WORDS    = ("apakah ada", "adakah", "is there", "are there", "apakah terdapat", "terdapat")
POS_WORDS      = ("posisi", "position", "letak", "dimana", "di mana", "where", "horizontal", "sebelah mana")
SIDE_WORDS     = ("sisi", "side", "sebelah")   # dicocokkan sbg KATA UTUH (lihat _has_side)


def _has_side(t):
    """True bila teks memuat kata sisi UTUH (hindari 'sisi' di dalam 'po-SISI')."""
    return bool(re.search(r"\b(sisi|side|sebelah)\b", t))
DOMINANT_WORDS = ("paling banyak", "terbanyak", "dominan", "mayoritas", "most frequent", "most common")
LARGEST_WORDS  = ("paling besar", "terbesar", "largest", "biggest", "paling gede")
BEST_WORDS     = ("tertinggi", "terbaik", "highest", "best", "paling bagus", "paling tinggi")
WORST_WORDS    = ("terendah", "terburuk", "lowest", "worst", "paling jelek", "paling rendah")


# ---------------------------------------------------------------------------
# Deteksi intent -> question_type
# ---------------------------------------------------------------------------
def detect_intent(t, grade, color, side, is_local):
    g = lambda grp: any(w in t for w in grp)          # apakah teks memuat salah satu sinonim

    # kata tanya "apa" sebagai KATA UTUH (hindari salah cocok dgn 'berapa', 'siapa',
    # 'kenapa', 'mengapa'). "apa"/"apa saja" menandakan pertanyaan "... apa?".
    asks_what = bool(re.search(r"\bapa\b", t))
    which = g(WHICH_WORDS) or asks_what
    each = g(EACH_WORDS)
    count = g(COUNT_WORDS)                             # "berapa" atau "jumlah" -> kuantitas
    is_grade = g(GRADE_WORDS)                          # "grade"/"mutu"/"kualitas" -> sama
    # konteks warna aktif bila kata "warna" disebut ATAU sebuah warna spesifik terdeteksi
    # (mis. "apakah ada yang hitam?" -> tak sebut 'warna' tapi slot warna = hitam)
    is_color = g(COLOR_WORDS) or (color is not None)

    # --- lokal (menyebut objek "ini"/"this"/obj_N) ---
    if is_local:
        if g(MARKET_WORDS):
            return "object_marketable"
        if is_color:
            return "object_color"
        if g(POS_WORDS):
            return "object_position"
        return "object_grade"                          # default objek: tanya grade

    # --- WARNA --- (dibuat setara dengan alur GRADE)
    if is_color:
        if color is not None:
            return "count_color"          # warna spesifik -> hitung warna itu (mis. "berapa mangga kuning")
        if each or count:
            return "color_breakdown"      # "warna apa ... jumlahnya berapa/masing-masing" -> rincian+jumlah
        return "which_colors"             # "warna apa (saja)" tanpa hitung -> daftar warna

    # --- GRADE + rincian jumlah (breakdown) : "grade ... jumlahnya masing-masing" ---
    if each and is_grade:
        return "grade_breakdown"

    # --- kelayakan jual keseluruhan ---
    if g(MARKET_WORDS):
        return "overall"

    # --- pemeringkatan mutu ---
    if g(DOMINANT_WORDS):
        return "dominant"
    if g(LARGEST_WORDS):
        return "largest_grade"
    if g(BEST_WORDS):
        return "best_quality"
    if g(WORST_WORDS):
        return "worst_quality"

    # --- sisi ---
    if side is not None or _has_side(t):
        return "grades_on_side"

    # --- keberadaan grade tertentu ---
    if g(EXIST_WORDS) and grade is not None:
        return "exist_class"

    # --- HITUNG (dipicu "berapa"/"jumlah") ---
    if count:
        if grade is not None:
            return "count_class"                       # "berapa mangga class 2"
        if color is not None:
            return "count_color"
        if is_grade and (which or each):
            return "grade_breakdown"                    # "berapa jumlah tiap grade"
        return "count_total"                            # "berapa jumlah mangga"

    # --- grade apa saja (tanpa kata hitung) ---
    if which and is_grade:
        return "which_grades"

    # --- fallback: menyebut grade spesifik -> hitung grade itu ---
    if grade is not None:
        return "count_class"
    return None


# ---------------------------------------------------------------------------
# Pemetaan intent+slot -> sampel QA yang sudah dibangkitkan
# ---------------------------------------------------------------------------
def _find_sample(samples, qtype, lang, grade=None, color=None, side=None, obj=None):
    cands = [s for s in samples if s["question_type"] == qtype]
    if not cands:
        return None
    if qtype in ("count_class", "exist_class") and grade is not None:
        lab = C.GRADE_LABEL[lang][grade]
        cands = [s for s in cands if lab in s["question"]] or cands
    if qtype == "count_color" and color is not None:
        lab = CL.color_name(color, lang)
        cands = [s for s in cands if lab in s["question"]] or cands
    if qtype == "grades_on_side" and side is not None:
        lab = C.SIDE[lang][side]
        cands = [s for s in cands if lab in s["question"]] or cands
    if qtype in ("object_grade", "object_color", "object_position", "object_marketable"):
        want = obj or "obj_1"
        cands = [s for s in cands if s.get("object_id") == want] or cands
    return cands[0]


# ---------------------------------------------------------------------------
# P3 — Fiturisasi kalimat (slot + isyarat leksikal)
# ---------------------------------------------------------------------------
def extract_features(text):
    """Jalankan P1..P4 dan kembalikan seluruh jejaknya sebagai dict."""
    norm = normalize(text)                 # P1
    tokens = tokenize(norm)                # P2
    gg = lambda grp: any(w in norm for w in grp)

    # --- kata tanya (interrogative) ---
    asks_what = bool(re.search(r"\bapa\b", norm))          # what / which
    asks_howmany = bool(re.search(r"\bberapa\b", norm))    # how many
    asks_yesno = bool(re.search(r"\bapakah\b|\badakah\b", norm)) or gg(EXIST_WORDS)

    # --- slot ---
    grade = detect_grade(norm)
    color = detect_color(norm)
    side = detect_side(norm)
    obj = detect_object(norm)

    # --- lokal vs global ("ini" lemah; kalah oleh isyarat global kuat) ---
    # Isyarat global kuat, TIDAK bergantung urutan kata:
    #   - rincian (masing-masing/each), which (apa saja/which),
    #   - KUANTITAS (berapa/jumlah): pertanyaan lokal per-objek tak pernah menghitung,
    #   - agregat menyeluruh (semua/seluruh/keseluruhan).
    strong_global = (gg(COUNT_WORDS) or gg(EACH_WORDS) or gg(WHICH_WORDS)
                     or any(w in norm for w in ("semua", "seluruh", "keseluruhan", "all")))
    says_this = bool(re.search(r"\bini\b|\bthis\b", norm))
    is_local = (obj is not None) or (says_this and not strong_global)

    # --- isyarat leksikal (kelompok kosakata yang aktif) ---
    cues = {
        "tanya_apa(what/which)": asks_what,
        "tanya_berapa(how many)": asks_howmany,
        "tanya_apakah(yes/no)": asks_yesno,
        "which/apa saja": gg(WHICH_WORDS) or asks_what,
        "count/jumlah": gg(COUNT_WORDS),
        "each/masing-masing": gg(EACH_WORDS),
        "konsep_grade": gg(GRADE_WORDS),
        "konsep_warna": gg(COLOR_WORDS),
        "layak_jual": gg(MARKET_WORDS),
        "posisi": gg(POS_WORDS),
        "sisi": _has_side(norm),
        "rank_dominan": gg(DOMINANT_WORDS),
        "rank_terbesar": gg(LARGEST_WORDS),
        "rank_tertinggi": gg(BEST_WORDS),
        "rank_terendah": gg(WORST_WORDS),
    }
    active_cues = [k for k, v in cues.items() if v]

    intent = detect_intent(norm, grade, color, side, is_local)   # P4

    return {
        "raw": text,
        "P1_normalisasi": norm,
        "P2_token": tokens,
        "P3_fitur": {
            "slot": {"grade": grade, "warna": color, "sisi": side, "objek": obj},
            "kata_tanya": ("apa/what-which" if asks_what else
                           "berapa/how-many" if asks_howmany else
                           "apakah/yes-no" if asks_yesno else "-"),
            "isyarat_aktif": active_cues,
            "lokal": is_local,
        },
        "P4_intent": intent,
    }


def match_question(text, lang, samples):
    """Kembalikan (sample|None, features). `features` = jejak pipeline P1..P4."""
    f = extract_features(text)
    slot = f["P3_fitur"]["slot"]
    qtype = f["P4_intent"]
    if qtype is None:
        return None, f

    side = slot["sisi"]
    if qtype == "grades_on_side" and side is None:
        side = "left"
        f["P3_fitur"]["slot"]["sisi"] = "left (default)"

    s = _find_sample(samples, qtype, lang, grade=slot["grade"], color=slot["warna"],
                     side=side, obj=slot["objek"])
    return s, f


def explain(text):
    """Cetak pipeline pemrosesan sebuah kalimat (untuk demonstrasi/paper)."""
    f = extract_features(text)
    print(f"RAW        : {f['raw']}")
    print(f"P1 normal. : {f['P1_normalisasi']}")
    print(f"P2 token   : {f['P2_token']}")
    fit = f["P3_fitur"]
    print(f"P3 slot    : {fit['slot']}")
    print(f"   kata tny : {fit['kata_tanya']}")
    print(f"   isyarat  : {fit['isyarat_aktif']}")
    print(f"   lokal    : {fit['lokal']}")
    print(f"P4 intent  : {f['P4_intent']}")
    return f
