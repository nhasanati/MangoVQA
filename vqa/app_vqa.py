"""
MangoVQA — Antarmuka Tanya-Jawab Mutu Mangga (Streamlit).

UI tipis di atas MangoVQAEngine (vqa_engine.py). Menyediakan DUA jalur pertanyaan:
  (a) Jalur terbuka  : kotak ketik pertanyaan bebas  -> engine.ask_open
  (b) Jalur templat  : dropdown daftar pertanyaan     -> engine.ask_template
Keduanya bermuara pada answer-bank yang sama (lihat vqa_engine.py untuk algoritma).

Jalankan:
    cd C:\\Users\\Lenovo\\Mango-YOLO
    streamlit run vqa/app_vqa.py
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vqa_engine as E   # noqa: E402


def show_answer(st, ans):
    if ans.ok:
        st.success(f"**Jawaban:** {ans.answer}")
        st.caption(f"pertanyaan: “{ans.question}” · tipe: {ans.question_type} · "
                   f"level: {ans.level} · jenis jawaban: {ans.answer_type} · jalur: {ans.route}")
    else:
        st.warning(ans.message)
    with st.expander("🔎 Jejak penalaran (langkah algoritma)"):
        st.json(ans.trace)


def run_ui():
    import streamlit as st

    st.set_page_config(page_title="MangoVQA — Tanya Jawab Mutu Mangga", layout="centered")
    st.title("🥭 MangoVQA — Tanya Jawab Mutu Mangga")
    st.caption("YOLOv11n train16 + template answering (grading SNI 3164:2024). "
               "Setelan inferensi = main5.py (imgsz 480).")

    @st.cache_resource
    def load_engine():
        return E.MangoVQAEngine()

    st.sidebar.header("⚙️ Pengaturan")
    lang = st.sidebar.radio("Bahasa pertanyaan", ["id", "en"],
                            format_func=lambda x: "Indonesia" if x == "id" else "English")
    conf = st.sidebar.slider("Confidence (conf)", 0.05, 0.90, 0.30, 0.05,
                             help="Default 0.30 sesuai main5.py.")

    engine = load_engine()
    up = st.file_uploader("Unggah gambar mangga", type=["jpg", "jpeg", "png"])
    if up is None:
        st.info("Silakan unggah gambar untuk mulai bertanya.")
        return

    file_bytes = np.asarray(bytearray(up.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        st.error("Gambar gagal dibaca. Coba unggah ulang.")
        return

    # ---- Tahap T1+T2 ----
    analysis = engine.analyze(img_bgr, lang=lang, conf=conf)

    c1, c2 = st.columns(2)
    with c1:
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Gambar diunggah",
                 use_container_width=True)
    with c2:
        st.image(cv2.cvtColor(analysis.plot_bgr, cv2.COLOR_BGR2RGB), caption="Hasil deteksi",
                 use_container_width=True)

    if not analysis.boxes:
        st.warning("Tidak ada mangga terdeteksi. Turunkan Confidence di sidebar.")
        return

    summ = analysis.detection_summary()
    st.subheader("Ringkasan deteksi")
    st.write(f"**Total mangga terdeteksi: {summ['total']}**")
    st.write(" · ".join(f"{g}: {n}" for g, n in summ["per_grade"].items()))

    st.divider()

    # ---- Tahap T3: dua jalur ----
    tab_open, tab_tpl = st.tabs(["✍️ Ketik pertanyaan (terbuka)", "📋 Pilih dari daftar (templat)"])

    with tab_open:
        st.caption("Ketik pertanyaan bebas; sistem memetakannya ke tipe pertanyaan terdekat.")
        q = st.text_input("Pertanyaan", placeholder="mis. berapa mangga class 1? / warna apa saja?")
        if q.strip():
            show_answer(st, engine.ask_open(analysis, q))
        with st.expander("Contoh pertanyaan yang dikenali"):
            st.markdown(
                "- berapa jumlah mangga? · berapa mangga class 2? · apakah ada reject?\n"
                "- grade apa saja? · warna apa saja? · berapa mangga kuning?\n"
                "- grade paling banyak? · mutu tertinggi? · apakah semua layak jual?\n"
                "- grade di sisi kiri? · apa warna mangga ini? · apa grade obj_2?")

    with tab_tpl:
        def qlabel(i):
            s = analysis.answer_bank[i]
            tag = "🌐" if s["level"] == "global" else f"📍{s['object_id']}"
            multi = " (multi)" if s["answer_type"] == "multi_label" else ""
            return f"{tag}  {s['question']}{multi}"

        idx = st.selectbox("Pilih pertanyaan", range(len(analysis.answer_bank)),
                           format_func=qlabel)
        show_answer(st, engine.ask_template(analysis, idx))

    with st.expander("📋 Lihat semua pertanyaan & jawaban untuk gambar ini"):
        for s in analysis.answer_bank:
            tag = "🌐" if s["level"] == "global" else f"📍{s['object_id']}"
            st.markdown(f"- {tag} {s['question']} → **{', '.join(s['answers'])}**")


if __name__ == "__main__":
    run_ui()
