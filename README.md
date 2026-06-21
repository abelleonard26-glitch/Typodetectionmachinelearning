# Deteksi & Koreksi Typo Bahasa Indonesia (Traditional ML)

Sistem untuk **mendeteksi** apakah sebuah kata/kalimat mengandung typo dan
**mengoreksinya**, dibangun **sepenuhnya dengan Machine Learning klasik** —
tanpa deep learning (tanpa neural network, transformer, RNN, LSTM, atau BERT).

## Pendekatan singkat
- **Deteksi level-KATA** — klasifikasi *kata valid* vs *typo* memakai TF-IDF
  n-gram karakter + model klasik (Logistic Regression, Complement Naive Bayes,
  Linear SVM, XGBoost — dibandingkan, diambil yang terbaik).
- **Deteksi level-KALIMAT** — klasifikasi *kalimat benar* vs *mengandung typo*
  memakai TF-IDF n-gram kata + karakter. Mampu menangkap *real-word error*.
- **Koreksi (non-DL)** — kandidat dicari dari kosakata benar dengan kombinasi
  *edit distance* (Damerau-Levenshtein), kemiripan *cosine* TF-IDF n-gram
  karakter, frekuensi kata, dan pola tetangga keyboard QWERTY.

## Struktur file
| File | Keterangan |
|------|------------|
| `typo_detection_correction.ipynb` | Notebook utama: EDA → fitur → pelatihan → evaluasi → simpan model → menu. |
| `typo_lib.py` | Pustaka inti: tokenizer, edit distance, kelas `TypoCorrector` (deteksi & koreksi). |
| `typo_app.py` | Aplikasi **menu terminal** untuk pengujian interaktif. |
| `streamlit_app.py` | Aplikasi **web (Streamlit)** untuk deteksi & koreksi interaktif. |
| `requirements.txt` | Daftar dependensi. |
| `models/` | Artefak hasil pelatihan (model, vectorizer, kamus) — dibuat oleh notebook. |
| `dataset/` | 18 file CSV (pasangan `kalimat_awal` ↔ `kalimat_salah`). |

## Cara menjalankan
1. **Pasang dependensi**
   ```bash
   pip install -r requirements.txt
   ```
2. **Latih model** — buka & jalankan seluruh sel notebook:
   ```bash
   jupyter notebook typo_detection_correction.ipynb
   ```
   Notebook akan membuat folder `models/` berisi artefak terlatih.
3. **Uji lewat menu terminal**
   ```bash
   python typo_app.py
   ```
   Masukkan kata/kalimat → sistem menampilkan status typo, confidence, dan saran koreksi.

## Contoh
```
Input : Buni apa?
Status: ADA TYPO terdeteksi
Koreksi: Bunyi apa?

Input : Bunyi apa?
Status: TIDAK ADA typo  -> Tidak ada koreksi yang dibutuhkan.
```

## Keterbatasan (didokumentasikan di notebook)
- **Real-word error** (typo yang kebetulan kata valid, mis. *belajar* → *belanja*)
  sulit dideteksi dari kata tunggal; karena itu ditambahkan **model level-kalimat**.
- Koreksi terbatas pada kata yang pernah muncul pada `kalimat_awal` (kamus dataset).
