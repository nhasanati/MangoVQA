"""
Configuration for the Mango Quality VQA dataset generator.

Source: a 4-class YOLO detection dataset where each class is a mango quality
grade. This file centralises domain knowledge (grade names, quality ordering)
and the natural-language question templates (Indonesian + English) so
generate_vqa.py stays focused on generation logic. Colour handling lives in
color.py.
"""

# ---------------------------------------------------------------------------
# Domain: class ids -> grade names (order matches data.yaml `names`)
# ---------------------------------------------------------------------------
CLASS_NAMES = {0: "Class 1", 1: "Class 2", 2: "Extra Class", 3: "Reject"}

# Quality ranking BEST -> WORST (Extra Class > Class I > Class II > Reject)
QUALITY_ORDER = [2, 0, 1, 3]
REJECT_ID = 3

DEFAULT_LANG = "id"

# Grade names as they appear inside a sentence, per language.
GRADE_LABEL = {
    "id": {0: "Class 1", 1: "Class 2", 2: "Extra Class", 3: "Reject"},
    "en": {0: "Class 1", 1: "Class 2", 2: "Extra Class", 3: "Reject"},
}

YESNO = {"id": {True: "Ya", False: "Tidak"},
         "en": {True: "Yes", False: "No"}}

# object_position: 3-way; grades_on_side: 2-way
POSITION = {"id": {"left": "kiri", "center": "tengah", "right": "kanan"},
            "en": {"left": "left", "center": "center", "right": "right"}}
SIDE = {"id": {"left": "kiri", "right": "kanan"},
        "en": {"left": "left", "right": "right"}}

VERDICT = {"id": {"good": "Layak jual", "reject": "Tidak layak jual"},
           "en": {"good": "Marketable", "reject": "Not marketable"}}

# ---------------------------------------------------------------------------
# Question templates. Slots: {grade}, {color}, {side}. Filled at generation time.
# ---------------------------------------------------------------------------
TEMPLATES = {
    "id": {
        # --- global ---
        "count_total":       "Berapa jumlah mangga dalam gambar?",
        "count_class":       "Berapa banyak mangga dengan grade {grade}?",
        "exist_class":       "Apakah ada mangga dengan grade {grade}?",
        "which_grades":      "Grade mutu apa saja yang terdapat dalam gambar?",
        "dominant":          "Grade mutu apa yang paling banyak dalam gambar?",
        "best_quality":      "Apa grade mutu tertinggi yang ada dalam gambar?",
        "worst_quality":     "Apa grade mutu terendah yang ada dalam gambar?",
        "overall":           "Apakah keseluruhan mangga dalam gambar layak jual?",
        "which_colors":      "Warna apa saja yang terdapat pada mangga dalam gambar?",
        "count_color":       "Berapa banyak mangga berwarna {color}?",
        "grades_on_side":    "Grade mutu apa saja yang berada di sisi {side} gambar?",
        "largest_grade":     "Apa grade dari mangga yang paling besar dalam gambar?",
        "grade_breakdown":   "Grade mutu apa saja yang ada dan berapa jumlah masing-masing?",
        "color_breakdown":   "Warna apa saja yang ada dan berapa jumlah masing-masing?",
        "count_grades":      "Berapa jumlah kualitas dalam gambar?",
        "count_colors":      "Berapa jumlah warna mangga dalam gambar?",
        "grade_color_global": "Apa kualitas dan warna mangga dalam gambar ini?",
        # --- local (per object) ---cd
        "object_grade":      "Apa grade mutu mangga ini?",
        "object_color":      "Apa warna mangga ini?",
        "object_position":   "Di mana posisi horizontal mangga ini?",
        "object_marketable": "Apakah mangga ini layak jual?",
        "object_color_grade": "Apa warna dan kualitas mangga ini?",
    },
    "en": {
        "count_total":       "How many mangoes are in the image?",
        "count_class":       "How many mangoes are graded as {grade}?",
        "exist_class":       "Is there a mango graded as {grade}?",
        "which_grades":      "Which quality grades are present in the image?",
        "dominant":          "Which quality grade is the most frequent in the image?",
        "best_quality":      "What is the highest quality grade present in the image?",
        "worst_quality":     "What is the lowest quality grade present in the image?",
        "overall":           "Are all mangoes in the image marketable?",
        "which_colors":      "Which colours appear on the mangoes in the image?",
        "count_color":       "How many mangoes are {color} in colour?",
        "grades_on_side":    "Which quality grades are on the {side} side of the image?",
        "largest_grade":     "What is the grade of the largest mango in the image?",
        "grade_breakdown":   "Which quality grades are present and how many of each?",
        "color_breakdown":   "Which colours are present and how many of each?",
        "count_grades":      "How many different quality grades are in the image?",
        "count_colors":      "How many different mango colours are in the image?",
        "grade_color_global": "What are the quality and colour of the mangoes in this image?",
        "object_grade":      "What is the quality grade of this mango?",
        "object_color":      "What is the colour of this mango?",
        "object_position":   "Where is this mango located horizontally?",
        "object_marketable": "Is this mango marketable?",
        "object_color_grade": "What is the colour and quality of this mango?",
    },
}
