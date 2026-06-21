"""Menu terminal untuk menguji sistem deteksi & koreksi typo.

Jalankan setelah folder models/ terisi (lihat notebook): python typo_app.py
"""

import os
import sys

from typo_lib import TypoCorrector, run_menu

MODELS_DIR = "models"


def main():
    if not os.path.exists(os.path.join(MODELS_DIR, "word_model.joblib")):
        print("[ERROR] Folder 'models/' atau artefak model tidak ditemukan.")
        print("        Jalankan dulu notebook 'typo_detection_correction.ipynb'")
        print("        sampai selesai untuk membuat file model.")
        sys.exit(1)

    print("Memuat model dan kamus koreksi ...")
    corrector = TypoCorrector.load(MODELS_DIR)
    print("Model berhasil dimuat.\n")

    run_menu(corrector)


if __name__ == "__main__":
    main()
