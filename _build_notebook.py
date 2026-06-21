"""Generator notebook typo_detection_correction.ipynb (pakai nbformat).

Jalankan: python _build_notebook.py
"""

import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
cells = []


def md(text):
    cells.append(new_markdown_cell(text))


def code(text):
    cells.append(new_code_cell(text))


md(r"""# Deteksi & Koreksi Typo Bahasa Indonesia (Traditional ML)

Sistem untuk mendeteksi dan mengoreksi typo pada teks Bahasa Indonesia memakai
machine learning klasik, tanpa deep learning. Model yang dibandingkan: Logistic
Regression, Complement Naive Bayes, Linear SVM, dan XGBoost.

Alur: inspeksi & pembersihan data, bangun kamus kata benar dan pasangan kata
salah→benar, latih model deteksi level kata dan level kalimat, lalu koreksi
dengan edit distance + kemiripan n-gram + frekuensi, dan terakhir menu interaktif.
""")

md("## 1. Setup & Import Library")
code(r"""# Pasang pandas bila belum tersedia (library lain sudah ada di environment).
try:
    import pandas  # noqa: F401
except ImportError:
    %pip install -q pandas

import os
import re
import glob
import math
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score, ConfusionMatrixDisplay)
from xgboost import XGBClassifier

from typo_lib import (tokenize, damerau_levenshtein, word_shape_features,
                      TypoCorrector, run_menu)

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
print("Semua library berhasil diimpor.")
""")

md("""## 2. Memuat Dataset

18 file CSV di folder `dataset/` (3 mata pelajaran x 6 jenis kesalahan). Tiap baris
berisi `kalimat_awal` (benar) dan `kalimat_salah` (mengandung satu typo).
""")
code(r"""csv_files = sorted(glob.glob(os.path.join("dataset", "*.csv")))
print(f"Jumlah file CSV ditemukan: {len(csv_files)}")

frames = [pd.read_csv(f, encoding="utf-8") for f in csv_files]
df = pd.concat(frames, ignore_index=True)

print(f"Total baris gabungan : {len(df):,}")
print(f"Kolom                : {list(df.columns)}")
df.head()
""")

md("## 3. Inspeksi Dataset (EDA)")
code(r"""print("=== INFO DATAFRAME ===")
df.info()

print("\n=== JUMLAH NILAI KOSONG (missing) PER KOLOM ===")
print(df.isnull().sum())

print("\n=== JUMLAH BARIS DUPLIKAT ===")
print(int(df.duplicated().sum()))
""")
code(r"""print("=== Distribusi per Pelajaran ===")
print(df["Pelajaran"].value_counts())
print("\n=== Distribusi per tipe_kesalahan ===")
print(df["tipe_kesalahan"].value_counts())
""")
code(r"""fig, axes = plt.subplots(1, 2, figsize=(13, 4))
df["Pelajaran"].value_counts().plot(kind="bar", ax=axes[0], color="#4C72B0")
axes[0].set_title("Jumlah data per Pelajaran")
axes[0].set_ylabel("Jumlah baris")
df["tipe_kesalahan"].value_counts().plot(kind="bar", ax=axes[1], color="#DD8452")
axes[1].set_title("Jumlah data per tipe_kesalahan")
axes[1].set_ylabel("Jumlah baris")
plt.tight_layout()
plt.show()
""")
code(r"""for tipe in df["tipe_kesalahan"].unique():
    contoh = df[df["tipe_kesalahan"] == tipe].iloc[0]
    print(f"[{tipe}]")
    print(f"   BENAR : {contoh['kalimat_awal']}")
    print(f"   SALAH : {contoh['kalimat_salah']}\n")
""")
code(r"""df["panjang_kata"] = df["kalimat_awal"].apply(lambda s: len(tokenize(s)))
plt.figure(figsize=(8, 4))
plt.hist(df["panjang_kata"], bins=40, color="#55A868")
plt.title("Distribusi Panjang Kalimat (jumlah kata) — kalimat_awal")
plt.xlabel("Jumlah kata")
plt.ylabel("Frekuensi")
plt.tight_layout()
plt.show()
print(df["panjang_kata"].describe())
""")

md("""## 4. Pembersihan Data

Rapikan spasi, buang baris kosong dan duplikat, serta buang pasangan yang
`kalimat_awal` sama persis dengan `kalimat_salah`.
""")
code(r"""baris_awal = len(df)

for col in ["kalimat_awal", "kalimat_salah"]:
    df[col] = df[col].astype(str).str.strip()

df = df.dropna(subset=["kalimat_awal", "kalimat_salah"])
df = df.drop_duplicates()
df = df[df["kalimat_awal"] != df["kalimat_salah"]].reset_index(drop=True)

print(f"Baris sebelum : {baris_awal:,}")
print(f"Baris sesudah : {len(df):,}")
print(f"Baris dibuang : {baris_awal - len(df):,}")
""")

md("""## 5. Kamus Kata Benar

Semua kata pada `kalimat_awal` adalah kata benar. Kita kumpulkan jadi
`Counter(kata -> frekuensi)`, lalu dipakai sebagai kelas "benar" untuk model
deteksi kata sekaligus sumber kandidat koreksi.
""")
code(r"""kalimat_benar_unik = df["kalimat_awal"].unique()
freq_kata = Counter()
for kal in kalimat_benar_unik:
    freq_kata.update(tokenize(kal))

vocab = sorted(freq_kata.keys())
print(f"Jumlah kalimat benar unik : {len(kalimat_benar_unik):,}")
print(f"Ukuran kosakata (vocab)   : {len(vocab):,}")
print("Contoh 15 kata paling sering:", freq_kata.most_common(15))
""")

md("""## 6. Ekstraksi Pasangan Kata (salah -> benar)

Selaraskan token tiap pasangan kalimat dengan `difflib.SequenceMatcher`. Bagian
yang berubah memberi pasangan (kata_benar, kata_salah) yang dipakai untuk melatih
deteksi kata dan mengevaluasi koreksi.
""")
code(r"""from difflib import SequenceMatcher

def ekstrak_pasangan_kata(kalimat_benar, kalimat_salah):
    a = tokenize(kalimat_benar)
    b = tokenize(kalimat_salah)
    pasangan = []
    sm = SequenceMatcher(a=a, b=b, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace":
            for k in range(max(i2 - i1, j2 - j1)):
                wb = a[i1 + k] if i1 + k < i2 else ""
                ws = b[j1 + k] if j1 + k < j2 else ""
                if wb and ws and wb != ws:
                    pasangan.append((wb, ws))
        elif tag == "delete":
            for k in range(i1, i2):
                pasangan.append((a[k], ""))
        elif tag == "insert":
            for k in range(j1, j2):
                pasangan.append(("", b[k]))
    return pasangan

records = []
for benar, salah, tipe in zip(df["kalimat_awal"], df["kalimat_salah"], df["tipe_kesalahan"]):
    for wb, ws in ekstrak_pasangan_kata(benar, salah):
        records.append((wb, ws, tipe))

pairs = pd.DataFrame(records, columns=["kata_benar", "kata_salah", "tipe_kesalahan"])
print(f"Total pasangan kata terekstrak: {len(pairs):,}")
pairs.head(10)
""")

md("""## 7. Data Deteksi Level-Kata

Label 1 = kata typo dari error ortografis (Deletion, Insertion, Subtitution,
Transposition, Punctuation); label 0 = kata dari kosakata benar. Real word error
dikecualikan karena typonya kata valid (ditangani model kalimat). Kelas
diseimbangkan lalu disubsample agar pelatihan cepat.
""")
code(r"""ERROR_ORTOGRAFIS = ["Deletion Error", "Insertion Error", "Subtitution Error",
                    "Transposition Error", "Punctuation Error"]

typo_words = pairs[(pairs["tipe_kesalahan"].isin(ERROR_ORTOGRAFIS)) &
                   (pairs["kata_salah"].str.len() > 0)]["kata_salah"]
vocab_set = set(vocab)
typo_words = [w for w in typo_words if w not in vocab_set]

MAX_PER_KELAS = 40000
rng = np.random.default_rng(RANDOM_STATE)

typo_unique = np.array(sorted(set(typo_words)))
if len(typo_unique) > MAX_PER_KELAS:
    typo_sample = rng.choice(typo_unique, MAX_PER_KELAS, replace=False)
else:
    typo_sample = typo_unique

vocab_arr = np.array(vocab)
n_neg = min(len(vocab_arr), len(typo_sample))
benar_sample = rng.choice(vocab_arr, n_neg, replace=False)

X_word = np.concatenate([typo_sample, benar_sample])
y_word = np.concatenate([np.ones(len(typo_sample), dtype=int),
                         np.zeros(len(benar_sample), dtype=int)])

print(f"Jumlah contoh typo (label 1) : {len(typo_sample):,}")
print(f"Jumlah contoh benar (label 0): {len(benar_sample):,}")
print(f"Total data deteksi kata      : {len(X_word):,}")
""")

md("""## 8-9. Fitur & Pelatihan Model Deteksi Level-Kata

Fitur utama: TF-IDF n-gram karakter (`char_wb`, ngram 2-4), efektif menangkap
pola ejaan tak lazim akibat typo. Fitur bentuk-kata lain (panjang, rasio vokal,
edit distance, pola keyboard) dipakai pada tahap koreksi. Latih dan bandingkan 4
model: Logistic Regression, Complement NB, Linear SVM (dikalibrasi), XGBoost.
""")
code(r"""print("Contoh fitur bentuk-kata:")
for w in ["belajar", "bljar", "bunyti", "ybuni"]:
    print(f"  {w:10s} -> {word_shape_features(w)}")
""")
code(r"""Xw_train, Xw_test, yw_train, yw_test = train_test_split(
    X_word, y_word, test_size=0.2, random_state=RANDOM_STATE, stratify=y_word)

def buat_pipeline_kata(clf):
    return Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                                  min_df=2, max_features=50000)),
        ("clf", clf),
    ])

model_kata = {
    "Logistic Regression": buat_pipeline_kata(
        LogisticRegression(max_iter=1000, C=5.0)),
    "Complement NB": buat_pipeline_kata(ComplementNB()),
    "Linear SVM": buat_pipeline_kata(
        CalibratedClassifierCV(LinearSVC(C=1.0), cv=3)),
    "XGBoost": buat_pipeline_kata(
        XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.3,
                      tree_method="hist", n_jobs=-1, eval_metric="logloss",
                      random_state=RANDOM_STATE)),
}

hasil_kata = {}
for nama, pipe in model_kata.items():
    print(f"Melatih: {nama} ...")
    pipe.fit(Xw_train, yw_train)
    pred = pipe.predict(Xw_test)
    proba = pipe.predict_proba(Xw_test)[:, 1]
    hasil_kata[nama] = {
        "accuracy": accuracy_score(yw_test, pred),
        "precision": precision_score(yw_test, pred),
        "recall": recall_score(yw_test, pred),
        "f1": f1_score(yw_test, pred),
        "roc_auc": roc_auc_score(yw_test, proba),
    }
print("Selesai melatih semua model kata.")
""")

md("## 10. Evaluasi & Pemilihan Model Deteksi Level-Kata")
code(r"""tabel_kata = pd.DataFrame(hasil_kata).T.sort_values("f1", ascending=False)
print("=== Perbandingan Model Deteksi KATA ===")
display(tabel_kata.style.format("{:.4f}"))

best_word_name = tabel_kata.index[0]
best_word_model = model_kata[best_word_name]
print(f"\nModel KATA terbaik: {best_word_name} (F1={tabel_kata.loc[best_word_name,'f1']:.4f})")
""")
code(r"""pred_best = best_word_model.predict(Xw_test)
print(f"=== Classification Report — {best_word_name} (deteksi KATA) ===")
print(classification_report(yw_test, pred_best, target_names=["benar", "typo"]))

cm = confusion_matrix(yw_test, pred_best)
ConfusionMatrixDisplay(cm, display_labels=["benar", "typo"]).plot(cmap="Blues")
plt.title(f"Confusion Matrix — {best_word_name} (KATA)")
plt.show()
""")

md("""## 11. Data Deteksi Level-Kalimat

Label 0 = `kalimat_awal` (benar), label 1 = `kalimat_salah` (typo). Model kalimat
memakai konteks n-gram kata sehingga bisa menangkap real word error. Subsample
berstratifikasi per pelajaran dan tipe kesalahan.
""")
code(r"""N_PER_TIPE = 2500   # per (pelajaran x tipe_kesalahan)
contoh = (df.groupby(["Pelajaran", "tipe_kesalahan"], group_keys=False)
            .apply(lambda g: g.sample(min(len(g), N_PER_TIPE), random_state=RANDOM_STATE)))

benar_df = pd.DataFrame({"teks": contoh["kalimat_awal"], "label": 0,
                         "tipe": "Benar"})
salah_df = pd.DataFrame({"teks": contoh["kalimat_salah"], "label": 1,
                         "tipe": contoh["tipe_kesalahan"]})
data_kalimat = pd.concat([benar_df, salah_df], ignore_index=True)
data_kalimat = data_kalimat.drop_duplicates(subset=["teks", "label"]).reset_index(drop=True)

print(f"Total data kalimat: {len(data_kalimat):,}")
print(data_kalimat["label"].value_counts())
""")

md("""## 12-13. Fitur & Pelatihan Model Deteksi Level-Kalimat

Fitur gabungan (`FeatureUnion`): TF-IDF n-gram kata (1,2) dan n-gram karakter
(2,4). Latih dan bandingkan 4 model klasik yang sama.
""")
code(r"""Xs = data_kalimat["teks"].values
ys = data_kalimat["label"].values
tipe_s = data_kalimat["tipe"].values

Xs_train, Xs_test, ys_train, ys_test, tipe_train, tipe_test = train_test_split(
    Xs, ys, tipe_s, test_size=0.2, random_state=RANDOM_STATE, stratify=ys)

def buat_pipeline_kalimat(clf):
    fitur = FeatureUnion([
        ("kata", TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                                 min_df=2, max_features=50000)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                                 min_df=2, max_features=50000)),
    ])
    return Pipeline([("fitur", fitur), ("clf", clf)])

model_kalimat = {
    "Logistic Regression": buat_pipeline_kalimat(
        LogisticRegression(max_iter=1000, C=5.0)),
    "Complement NB": buat_pipeline_kalimat(ComplementNB()),
    "Linear SVM": buat_pipeline_kalimat(
        CalibratedClassifierCV(LinearSVC(C=1.0), cv=3)),
    "XGBoost": buat_pipeline_kalimat(
        XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.3,
                      tree_method="hist", n_jobs=-1, eval_metric="logloss",
                      random_state=RANDOM_STATE)),
}

hasil_kalimat = {}
for nama, pipe in model_kalimat.items():
    print(f"Melatih: {nama} ...")
    pipe.fit(Xs_train, ys_train)
    pred = pipe.predict(Xs_test)
    proba = pipe.predict_proba(Xs_test)[:, 1]
    hasil_kalimat[nama] = {
        "accuracy": accuracy_score(ys_test, pred),
        "precision": precision_score(ys_test, pred),
        "recall": recall_score(ys_test, pred),
        "f1": f1_score(ys_test, pred),
        "roc_auc": roc_auc_score(ys_test, proba),
    }
print("Selesai melatih semua model kalimat.")
""")
code(r"""tabel_kalimat = pd.DataFrame(hasil_kalimat).T.sort_values("f1", ascending=False)
print("=== Perbandingan Model Deteksi KALIMAT ===")
display(tabel_kalimat.style.format("{:.4f}"))

best_sent_name = tabel_kalimat.index[0]
best_sent_model = model_kalimat[best_sent_name]
print(f"\nModel KALIMAT terbaik: {best_sent_name} (F1={tabel_kalimat.loc[best_sent_name,'f1']:.4f})")

pred_s = best_sent_model.predict(Xs_test)
print(classification_report(ys_test, pred_s, target_names=["benar", "typo"]))
cm_s = confusion_matrix(ys_test, pred_s)
ConfusionMatrixDisplay(cm_s, display_labels=["benar", "typo"]).plot(cmap="Greens")
plt.title(f"Confusion Matrix — {best_sent_name} (KALIMAT)")
plt.show()
""")
code(r"""# Recall per jenis kesalahan: terlihat seberapa baik real word error tertangkap.
mask_typo = ys_test == 1
df_eval = pd.DataFrame({"tipe": tipe_test[mask_typo],
                        "benar_terdeteksi": pred_s[mask_typo] == 1})
recall_per_tipe = df_eval.groupby("tipe")["benar_terdeteksi"].mean().sort_values()
print("=== Recall per tipe kesalahan (model KALIMAT) ===")
print((recall_per_tipe * 100).round(2).astype(str) + " %")
""")

md("""## 14. Modul Koreksi (Non-Deep-Learning)

Kandidat koreksi diambil dari kosakata benar lalu diperingkat dengan kombinasi:
cosine TF-IDF n-gram karakter, edit distance Damerau-Levenshtein, frekuensi kata,
dan bonus pola keyboard. Kita bangun indeks kemiripan lalu rakit `TypoCorrector`.
""")
code(r"""index_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
index_matrix = index_vectorizer.fit_transform(vocab)   # L2-normalized
print(f"Indeks kemiripan dibangun: {index_matrix.shape[0]:,} kata, "
      f"{index_matrix.shape[1]:,} fitur n-gram.")

corrector = TypoCorrector(
    word_model=best_word_model,
    sentence_model=best_sent_model,
    correction_freq=dict(freq_kata),
    index_vectorizer=index_vectorizer,
    index_matrix=index_matrix,
    vocab=vocab,
)
print("TypoCorrector siap digunakan.")
""")
code(r"""for contoh in ["Buni apa?", "Bunyi apa?", "bljar", "Siswa sedang menulis surat."]:
    d = corrector.detect_typo(contoh)
    s = corrector.suggest_correction(contoh)
    print(f"INPUT   : {contoh}")
    print(f"  deteksi : is_typo={d['is_typo']} | confidence={d['confidence']:.2%}")
    print(f"  koreksi : {s['corrected']}")
    if s["is_typo"]:
        print(f"  saran   : {s['suggestions']}")
    print()
""")

md("""## 15. Evaluasi Koreksi (Top-1 & Top-3)

Uji koreksi pada pasangan (kata_salah -> kata_benar) yang ditahan. Top-1 = kandidat
peringkat 1 benar; Top-3 = kata benar ada di 3 teratas. Real word error
diperkirakan rendah karena typonya berupa kata valid.
""")
code(r"""eval_pairs = pairs[(pairs["kata_salah"].str.len() > 0) &
                   (pairs["kata_benar"].str.len() > 0)].copy()
eval_pairs = eval_pairs[eval_pairs["kata_salah"] != eval_pairs["kata_benar"]]
eval_sample = eval_pairs.sample(min(4000, len(eval_pairs)), random_state=RANDOM_STATE)

def eval_koreksi(row, k=3):
    cands = [c for c, _ in corrector.candidates(row["kata_salah"], k=k)]
    top1 = bool(cands and cands[0] == row["kata_benar"])
    topk = bool(row["kata_benar"] in cands)
    return pd.Series({"top1": top1, "top3": topk})

hasil_koreksi = eval_sample.join(eval_sample.apply(eval_koreksi, axis=1))
print("=== Akurasi Koreksi (SEMUA pasangan, termasuk real-word) ===")
print(f"Top-1 : {hasil_koreksi['top1'].mean():.2%}")
print(f"Top-3 : {hasil_koreksi['top3'].mean():.2%}")

# Skenario realistis: typo yang benar-benar di luar kamus (OOV) dengan target dikenal.
vset = set(vocab)
oov_mask = (~hasil_koreksi["kata_salah"].isin(vset)) & (hasil_koreksi["kata_benar"].isin(vset))
oov_res = hasil_koreksi[oov_mask]
print(f"\n=== Akurasi Koreksi (khusus typo di luar kamus/OOV, n={len(oov_res):,}) ===")
print(f"Top-1 : {oov_res['top1'].mean():.2%}")
print(f"Top-3 : {oov_res['top3'].mean():.2%}")

print("\n=== Akurasi Koreksi per tipe kesalahan ===")
display((hasil_koreksi.groupby("tipe_kesalahan")[["top1", "top3"]]
         .mean().mul(100).round(2)).style.format("{:.2f}%"))
""")

md("""## 16. Menyimpan Artefak

Simpan model, vectorizer, dan kamus ke folder `models/` dengan joblib agar bisa
dimuat ulang oleh `typo_app.py` dan `streamlit_app.py` tanpa pelatihan ulang.
""")
code(r"""corrector.config["best_word_model"] = best_word_name
corrector.config["best_sentence_model"] = best_sent_name
corrector.config["metrik_kata"] = hasil_kata[best_word_name]
corrector.config["metrik_kalimat"] = hasil_kalimat[best_sent_name]

corrector.save("models")
print("Artefak tersimpan di folder 'models/':")
for f in sorted(os.listdir("models")):
    ukuran = os.path.getsize(os.path.join("models", f)) / 1024
    print(f"  - {f:28s} ({ukuran:,.1f} KB)")
""")

md("""## 17. Menu Interaktif

Jalankan sel berikut untuk uji interaktif di dalam notebook, atau dari terminal:
`python typo_app.py`. Saat eksekusi non-interaktif menu berhenti dengan rapi.
""")
code(r"""corrector_loaded = TypoCorrector.load("models")
run_menu(corrector_loaded)
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

with open("typo_detection_correction.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook dibuat: typo_detection_correction.ipynb ({len(cells)} sel)")
