"""Antarmuka web Streamlit untuk deteksi & koreksi typo. Jalankan: streamlit run streamlit_app.py"""

import os

import streamlit as st

from typo_lib import TypoCorrector

MODELS_DIR = "models"

st.set_page_config(
    page_title="Deteksi & Koreksi Typo (Traditional ML)",
    page_icon="✏️",
    layout="centered",
)


@st.cache_resource(show_spinner="Memuat model & kamus koreksi ...")
def muat_corrector(models_dir: str):
    return TypoCorrector.load(models_dir)


def folder_model_siap(models_dir: str) -> bool:
    return os.path.isfile(os.path.join(models_dir, "word_model.joblib"))


st.title("✏️ Deteksi & Koreksi Typo Bahasa Indonesia")
st.caption(
    "Sistem berbasis **Machine Learning klasik** (TF-IDF n-gram + Linear SVM / "
    "Naive Bayes / Logistic Regression / XGBoost) — **tanpa** deep learning."
)

if not folder_model_siap(MODELS_DIR):
    st.error(
        "Artefak model belum ditemukan di folder `models/`.\n\n"
        "Jalankan dulu notebook **`typo_detection_correction.ipynb`** sampai "
        "selesai untuk membuat file model, lalu muat ulang halaman ini."
    )
    st.stop()

corrector = muat_corrector(MODELS_DIR)

with st.sidebar:
    st.header("ℹ️ Informasi Model")
    cfg = corrector.config
    st.markdown(
        f"- **Model kata**: {cfg.get('best_word_model', '-')}\n"
        f"- **Model kalimat**: {cfg.get('best_sentence_model', '-')}\n"
        f"- **Ukuran kosakata**: {len(corrector.vocab):,} kata"
    )
    mk = cfg.get("metrik_kata", {})
    ms = cfg.get("metrik_kalimat", {})
    if mk:
        st.markdown(
            f"**F1 model kata**: {mk.get('f1', 0):.3f}  \n"
            f"**F1 model kalimat**: {ms.get('f1', 0):.3f}"
        )

    st.divider()
    st.subheader("⚙️ Pengaturan")
    corrector.config["word_threshold"] = st.slider(
        "Ambang deteksi kata", 0.0, 1.0, float(cfg.get("word_threshold", 0.5)), 0.05,
        help="Makin rendah, makin sensitif menandai kata sebagai typo.",
    )
    corrector.config["sentence_threshold"] = st.slider(
        "Ambang deteksi kalimat", 0.0, 1.0, float(cfg.get("sentence_threshold", 0.5)), 0.05,
    )
    top_k = st.slider("Jumlah saran (top-k)", 1, 5, int(cfg.get("top_k", 3)))
    corrector.config["top_k"] = top_k

st.subheader("Masukkan kata atau kalimat")

contoh = ["Buni apa?", "Bunyi apa?", "bljar",
          "Tiap orang punya bend berharga.", "Siswa sedang menulis surat."]
kolom = st.columns(len(contoh))
for c, teks_contoh in zip(kolom, contoh):
    if c.button(teks_contoh, use_container_width=True):
        st.session_state["teks_input"] = teks_contoh

teks = st.text_area(
    "Teks",
    key="teks_input",
    height=90,
    placeholder="Contoh: Buni apa?",
    label_visibility="collapsed",
)

periksa = st.button("🔍 Periksa", type="primary", use_container_width=True)

if periksa:
    if not teks or not teks.strip():
        st.warning("⚠️ Input kosong tidak diperbolehkan. Silakan ketik sesuatu.")
        st.stop()

    teks = teks.strip()
    det = corrector.detect_typo(teks)
    hasil = corrector.suggest_correction(teks)

    st.divider()

    c1, c2 = st.columns(2)
    if det["is_typo"]:
        c1.error("### ❌ ADA TYPO terdeteksi")
    else:
        c1.success("### ✅ TIDAK ADA typo")
    c2.metric("Confidence", f"{det['confidence']:.1%}",
              help="Probabilitas pendukung kesimpulan dari model.")

    if det["level"] == "kata":
        prob = det["word_proba"]
        st.caption(f"Level: **kata** — probabilitas typo (model kata): {prob:.1%}")
    else:
        prob = det["sent_proba"]
        st.caption(f"Level: **kalimat** — probabilitas typo (model kalimat): {prob:.1%}")
    st.progress(min(max(prob, 0.0), 1.0))

    if hasil["is_typo"] and hasil["suggestions"]:
        st.subheader("✅ Hasil Koreksi")
        cc1, cc2 = st.columns(2)
        cc1.markdown("**Teks asli**")
        cc1.info(teks)
        cc2.markdown("**Teks terkoreksi**")
        cc2.success(hasil["corrected"])

        st.markdown("**Saran kandidat (top-k):**")
        for kata, kandidat in hasil["suggestions"].items():
            with st.expander(f"Kata salah: `{kata}`", expanded=True):
                for rank, (cand, skor) in enumerate(kandidat, start=1):
                    st.write(f"{rank}. **{cand}**  —  skor {skor:.3f}")
    elif hasil["is_typo"]:
        st.warning(
            "Terindikasi typo, namun tidak ada saran koreksi yang ditemukan "
            "(kemungkinan *real-word error* — kata salah berupa kata valid)."
        )
    else:
        st.info("👍 Tidak ada koreksi yang dibutuhkan.")

st.divider()
