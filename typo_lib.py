"""Pustaka inti deteksi & koreksi typo Bahasa Indonesia (traditional ML).

Berisi util teks, edit distance, dan kelas TypoCorrector yang dipakai bersama
oleh notebook dan typo_app.py.
"""

import os
import re
import math
from collections import Counter

import numpy as np
import joblib


# Tokenizer: ambil kata termasuk kata berhubung seperti "siap-siap".
_TOKEN_RE = re.compile(r"\w+(?:-\w+)*", flags=re.UNICODE)
# Versi dengan grup penangkap agar spasi & tanda baca ikut terbawa saat split.
_SPLIT_RE = re.compile(r"(\w+(?:-\w+)*)", flags=re.UNICODE)

_VOWELS = set("aiueo")

# Tetangga tombol QWERTY untuk pola typo salah pencet tombol sebelah.
_KEYBOARD_NEIGHBORS = {
    "q": "wa", "w": "qeas", "e": "wrsd", "r": "etdf", "t": "ryfg",
    "y": "tugh", "u": "yihj", "i": "uojk", "o": "ipkl", "p": "ol",
    "a": "qwsz", "s": "awedxz", "d": "serfcx", "f": "drtgvc", "g": "ftyhbv",
    "h": "gyujnb", "j": "huikmn", "k": "jiolm", "l": "kop",
    "z": "asx", "x": "zsdc", "c": "xdfv", "v": "cfgb", "b": "vghn",
    "n": "bhjm", "m": "njk",
}


def tokenize(text):
    """Pecah teks jadi daftar token kata (lowercase)."""
    return _TOKEN_RE.findall(str(text).lower())


def split_keep_tokens(text):
    """Pecah teks jadi potongan kata & non-kata bergantian (untuk merangkai ulang)."""
    return _SPLIT_RE.split(str(text))


def damerau_levenshtein(a, b):
    """Jarak edit Damerau-Levenshtein (sisip, hapus, substitusi, transposisi)."""
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    d = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        d[i][0] = i
    for j in range(lb + 1):
        d[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,
                d[i][j - 1] + 1,
                d[i - 1][j - 1] + cost,
            )
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[la][lb]


def _is_keyboard_typo(a, b):
    """True bila a dan b beda 1 huruf yang bersebelahan di keyboard QWERTY."""
    if len(a) != len(b):
        return False
    diff = [(x, y) for x, y in zip(a, b) if x != y]
    if len(diff) != 1:
        return False
    x, y = diff[0]
    return y in _KEYBOARD_NEIGHBORS.get(x, "")


def word_shape_features(word):
    """Fitur bentuk kata: panjang, rasio vokal, deret konsonan, dll."""
    w = str(word).lower()
    n = len(w)
    n_vowel = sum(ch in _VOWELS for ch in w)
    max_cons = cur = 0
    for ch in w:
        if ch.isalpha() and ch not in _VOWELS:
            cur += 1
            max_cons = max(max_cons, cur)
        else:
            cur = 0
    has_repeat = any(w[i] == w[i + 1] for i in range(n - 1))
    return {
        "panjang": n,
        "rasio_vokal": (n_vowel / n) if n else 0.0,
        "deret_konsonan_max": max_cons,
        "ada_angka": int(any(ch.isdigit() for ch in w)),
        "ada_huruf_berulang": int(has_repeat),
    }


class TypoCorrector:
    """Pembungkus model deteksi (kata & kalimat) dan logika koreksi typo."""

    def __init__(self, word_model, sentence_model, correction_freq,
                 index_vectorizer, index_matrix, vocab, config=None):
        self.word_model = word_model
        self.sentence_model = sentence_model
        self.correction_freq = dict(correction_freq)
        self.index_vectorizer = index_vectorizer
        self.index_matrix = index_matrix
        self.vocab = list(vocab)
        self.vocab_set = set(self.vocab)
        self.config = config or self.default_config()
        self._max_log_freq = math.log1p(max(self.correction_freq.values())) if self.correction_freq else 1.0

    @staticmethod
    def default_config():
        # Bobot koreksi dipilih lewat grid-search kecil pada pasangan held-out.
        return {
            "word_threshold": 0.5,
            "sentence_threshold": 0.5,
            "top_n_candidates": 50,
            "top_k": 3,
            "w_cosine": 0.30,
            "w_editdist": 0.40,
            "w_freq": 0.25,
            "w_keyboard": 0.05,
            "correction_threshold": 0.30,
        }

    def _word_typo_proba(self, word):
        return float(self.word_model.predict_proba([word.lower()])[0][1])

    def _sentence_typo_proba(self, text):
        return float(self.sentence_model.predict_proba([text])[0][1])

    def detect_typo(self, text):
        """Deteksi apakah text (kata atau kalimat) mengandung typo."""
        text = str(text).strip()
        tokens = tokenize(text)
        wt = self.config["word_threshold"]
        st = self.config["sentence_threshold"]

        # Input satu kata: cukup pakai model level-kata.
        if len(tokens) <= 1:
            word = tokens[0] if tokens else ""
            if not word:
                return {"input": text, "level": "kata", "is_typo": False,
                        "confidence": 0.0, "word_proba": 0.0, "sent_proba": None,
                        "flagged": [], "word_probs": {}}
            in_vocab = word in self.vocab_set
            wp = self._word_typo_proba(word)
            is_typo = (not in_vocab) and (wp >= wt)
            return {
                "input": text, "level": "kata", "is_typo": is_typo,
                "confidence": wp if is_typo else (1.0 - wp),
                "word_proba": wp, "sent_proba": None,
                "flagged": [word] if is_typo else [],
                "word_probs": {word: wp},
            }

        # Input kalimat: gabungkan model kalimat dengan model kata per token.
        sp = self._sentence_typo_proba(text)
        flagged = []
        word_probs = {}
        for tok in tokens:
            if tok in self.vocab_set:
                word_probs[tok] = 0.0
                continue
            p = self._word_typo_proba(tok)
            word_probs[tok] = p
            if p >= wt:
                flagged.append(tok)
        is_typo = (sp >= st) or (len(flagged) > 0)
        return {
            "input": text, "level": "kalimat", "is_typo": is_typo,
            "confidence": sp if is_typo else (1.0 - sp),
            "word_proba": None, "sent_proba": sp,
            "flagged": flagged, "word_probs": word_probs,
        }

    def candidates(self, word, k=None):
        """Kandidat koreksi untuk sebuah kata, terurut dari skor tertinggi."""
        k = k or self.config["top_k"]
        w = str(word).lower()
        if not w:
            return []

        cfg = self.config
        # TF-IDF sudah ternormalisasi L2, jadi dot product = cosine similarity.
        qv = self.index_vectorizer.transform([w])
        sims = np.asarray(self.index_matrix.dot(qv.T).todense()).ravel()
        n = min(cfg["top_n_candidates"], len(self.vocab))
        top_idx = np.argpartition(-sims, n - 1)[:n]

        scored = []
        for idx in top_idx:
            cand = self.vocab[idx]
            cos = float(sims[idx])
            dl = damerau_levenshtein(w, cand)
            norm_dist = dl / max(len(w), len(cand), 1)
            freq = self.correction_freq.get(cand, 1)
            norm_freq = math.log1p(freq) / self._max_log_freq
            kb = 1.0 if _is_keyboard_typo(w, cand) else 0.0
            score = (cfg["w_cosine"] * cos
                     + cfg["w_editdist"] * (1.0 - norm_dist)
                     + cfg["w_freq"] * norm_freq
                     + cfg["w_keyboard"] * kb)
            scored.append((cand, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def suggest_correction(self, text):
        """Sarankan koreksi untuk text dan kembalikan teks yang sudah dikoreksi."""
        det = self.detect_typo(text)
        if not det["is_typo"]:
            return {"input": text, "is_typo": False, "corrected": text,
                    "suggestions": {}, "message": "Tidak ada typo terdeteksi. "
                                                   "Tidak ada koreksi yang dibutuhkan."}

        words_to_fix = list(det["flagged"])
        if not words_to_fix:
            # Kalimat terdeteksi typo tapi tak ada kata yang lolos ambang:
            # targetkan kata di luar kamus yang paling mencurigakan.
            ct = self.config.get("correction_threshold", 0.30)
            oov = [(w, p) for w, p in det.get("word_probs", {}).items()
                   if w not in self.vocab_set]
            oov.sort(key=lambda x: x[1], reverse=True)
            words_to_fix = [w for w, p in oov if p >= ct]
            if not words_to_fix and oov:
                words_to_fix = [oov[0][0]]

        suggestions = {}
        for w in words_to_fix:
            cands = self.candidates(w, k=self.config["top_k"])
            if cands and cands[0][0] != w:
                suggestions[w] = cands

        if not suggestions:
            return {"input": text, "is_typo": True, "corrected": text,
                    "suggestions": {},
                    "message": "Terindikasi typo, namun tidak ada saran koreksi yang ditemukan."}

        corrected = self._rebuild(text, list(suggestions.keys()), suggestions)
        return {"input": text, "is_typo": True, "corrected": corrected,
                "suggestions": suggestions,
                "message": "Typo terdeteksi. Berikut saran koreksinya."}

    def correct_sentence(self, text):
        return self.suggest_correction(text)["corrected"]

    def _rebuild(self, text, flagged, suggestions):
        # Ganti hanya token typo, pertahankan kapitalisasi & tanda baca asli.
        flagged_set = set(flagged)
        parts = split_keep_tokens(text)
        for i, part in enumerate(parts):
            low = part.lower()
            if low in flagged_set and suggestions.get(low):
                best = suggestions[low][0][0]
                parts[i] = _match_case(part, best)
        return "".join(parts)

    def save(self, models_dir="models"):
        os.makedirs(models_dir, exist_ok=True)
        joblib.dump(self.word_model, os.path.join(models_dir, "word_model.joblib"))
        joblib.dump(self.sentence_model, os.path.join(models_dir, "sentence_model.joblib"))
        joblib.dump({"freq": self.correction_freq, "vocab": self.vocab},
                    os.path.join(models_dir, "correction_dict.joblib"))
        joblib.dump({"vectorizer": self.index_vectorizer, "matrix": self.index_matrix,
                     "vocab": self.vocab},
                    os.path.join(models_dir, "correction_index.joblib"))
        joblib.dump(self.config, os.path.join(models_dir, "metadata.joblib"))

    @classmethod
    def load(cls, models_dir="models"):
        word_model = joblib.load(os.path.join(models_dir, "word_model.joblib"))
        sentence_model = joblib.load(os.path.join(models_dir, "sentence_model.joblib"))
        cdict = joblib.load(os.path.join(models_dir, "correction_dict.joblib"))
        cindex = joblib.load(os.path.join(models_dir, "correction_index.joblib"))
        config = joblib.load(os.path.join(models_dir, "metadata.joblib"))
        return cls(
            word_model=word_model,
            sentence_model=sentence_model,
            correction_freq=cdict["freq"],
            index_vectorizer=cindex["vectorizer"],
            index_matrix=cindex["matrix"],
            vocab=cindex["vocab"],
            config=config,
        )


def _match_case(original, replacement):
    if original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement.capitalize()
    return replacement


def run_menu(corrector=None, models_dir="models"):
    """Menu terminal interaktif. Berhenti rapi bila stdin tidak tersedia."""
    if corrector is None:
        corrector = TypoCorrector.load(models_dir)

    print("=" * 60)
    print("  SISTEM DETEKSI & KOREKSI TYPO (Traditional ML)")
    print("=" * 60)

    while True:
        print("\nMENU:")
        print("  1. Periksa kata / kalimat (deteksi + koreksi)")
        print("  2. Keluar")
        try:
            pilihan = input("Pilih menu (1/2): ").strip()
        except Exception:
            print("\n[Info] Input interaktif tidak tersedia. Menu dihentikan.")
            return

        if pilihan == "2":
            print("Terima kasih! Program selesai.")
            return
        if pilihan != "1":
            print("Pilihan tidak valid. Silakan masukkan 1 atau 2.")
            continue

        try:
            teks = input("Masukkan kata/kalimat: ").strip()
        except Exception:
            print("\n[Info] Input interaktif tidak tersedia. Menu dihentikan.")
            return

        if not teks:
            print(">> Input kosong tidak diperbolehkan. Silakan coba lagi.")
            continue

        hasil = corrector.suggest_correction(teks)
        det = corrector.detect_typo(teks)

        print("-" * 60)
        if det["is_typo"]:
            print("Status      : ADA TYPO terdeteksi")
        else:
            print("Status      : TIDAK ADA typo")
        if det["level"] == "kata":
            print("Level       : kata")
            print(f"Prob. typo  : {det['word_proba']:.2%} (model kata)")
        else:
            print("Level       : kalimat")
            print(f"Prob. typo  : {det['sent_proba']:.2%} (model kalimat)")
        print(f"Confidence  : {det['confidence']:.2%}")

        if hasil["is_typo"]:
            print(f"Kata salah  : {', '.join(hasil['suggestions'].keys())}")
            print(f"Koreksi     : {hasil['corrected']}")
            print("Saran kandidat (top-k):")
            for w, cands in hasil["suggestions"].items():
                opsi = ", ".join(f"{c} ({s:.2f})" for c, s in cands)
                print(f"   - {w} -> {opsi}")
        else:
            print(">> Tidak ada koreksi yang dibutuhkan.")
        print("-" * 60)


if __name__ == "__main__":
    run_menu()
