"""Beranda aplikasi — Deteksi Penggunaan Helm Pengendara Motor.

Jalankan dari root repo:
    streamlit run app/streamlit_app.py
"""
import streamlit as st

st.set_page_config(page_title="Deteksi Helm — Demo & Hasil", page_icon="🏍️", layout="wide")

st.title("🏍️ Deteksi Penggunaan Helm Pengendara Motor")
st.caption("Demo deteksi interaktif + dashboard hasil eksperimen (Fase 1).")

st.markdown(
    """
Aplikasi ini menampilkan hasil riset deteksi objek **3 kelas** —
`helmet`, `license_plate`, `motorcyclist` — pada dataset **NCKH 2023 v19**.

**Gunakan menu di sidebar kiri:**

- 🔍 **Demo Deteksi** — unggah/pilih gambar, pilih model (YOLOv8s / YOLO11s /
  CBAM), dan lihat kotak deteksi beserta confidence-nya.
- 📊 **Hasil Eksperimen** — tabel perbandingan arsitektur (mAP, FPS),
  figur publikasi, dan performa per-kelas.
"""
)

c1, c2, c3 = st.columns(3)
c1.metric("Model terbaik", "YOLOv8s", "mAP@0.5 = 0.9592")
c2.metric("Kecepatan", "~296 FPS", "tercepat dari 4 model")
c3.metric("Temuan utama", "Atensi ✗", "tak mengalahkan CNN murni")

st.info(
    "**Kesimpulan riset:** pada dataset ini tidak ada mekanisme atensi "
    "(C2PSA, transformer, CBAM) yang mengalahkan YOLOv8s CNN-murni; CBAM justru "
    "turun signifikan (uji-t berpasangan p=0,008).",
    icon="💡",
)

st.divider()
st.caption(
    "Repo: github.com/salmanrachmadi/CV · Paper lengkap di `results/` "
    "(versi Inggris di `results/en/`)."
)
