"""Halaman dashboard — perbandingan model, figur publikasi, performa per-kelas."""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app_utils import FIGURES, PER_CLASS, load_summary  # noqa: E402

st.set_page_config(page_title="Hasil Eksperimen", page_icon="📊", layout="wide")
st.title("📊 Hasil Eksperimen")
st.caption("Perbandingan 4 arsitektur pada split uji NCKH 2023 v19 (multi-seed).")

# --- Tabel perbandingan ---
st.subheader("Perbandingan model")
rows = load_summary()
df = pd.DataFrame(rows)
disp = df.assign(**{
    "mAP@0.5": df.apply(lambda r: f"{r['mAP@0.5']:.4f} ± {r['mAP@0.5_std']:.4f}", axis=1),
    "mAP@[.5:.95]": df["mAP@[.5:.95]"].map("{:.4f}".format),
    "FPS": df["FPS"].map(lambda v: f"~{v}"),
})[["Model", "n_seed", "mAP@0.5", "mAP@[.5:.95]", "FPS"]]

best = df.loc[df["mAP@0.5"].idxmax(), "Model"]
st.dataframe(disp, hide_index=True, use_container_width=True)
st.success(f"**Pemenang akurasi & kecepatan: {best}** — mAP@0.5 tertinggi sekaligus FPS tertinggi.", icon="🏆")

col = st.columns(2)
with col[0]:
    st.caption("mAP@0.5 antar-model")
    st.bar_chart(df.set_index("Model")["mAP@0.5"])
with col[1]:
    st.caption("Kecepatan (FPS) antar-model")
    st.bar_chart(df.set_index("Model")["FPS"])

st.divider()

# --- Performa per-kelas ---
st.subheader("Performa per-kelas (YOLOv8s baseline)")
pc = pd.DataFrame({"kelas": list(PER_CLASS), "mAP@0.5": list(PER_CLASS.values())})
c1, c2 = st.columns([2, 3])
with c1:
    st.dataframe(pc.style.format({"mAP@0.5": "{:.3f}"}), hide_index=True, use_container_width=True)
with c2:
    st.bar_chart(pc.set_index("kelas"))
st.caption("Kelas `helmet` paling menantang (kecil & ambigu) — jadi penentu utama ruang perbaikan.")

st.divider()

# --- Figur publikasi ---
st.subheader("Figur publikasi")
figs = [
    ("fig1_attention_mAP50.png", "Pengaruh mekanisme atensi terhadap mAP@0.5"),
    ("fig2_per_class.png", "Performa per-kelas YOLOv8s"),
    ("fig3_accuracy_speed.png", "Trade-off akurasi vs kecepatan"),
    ("fig4_detection_example.png", "Contoh keluaran deteksi"),
]
cols = st.columns(2)
for i, (fname, cap) in enumerate(figs):
    p = FIGURES / fname
    with cols[i % 2]:
        if p.exists():
            st.image(str(p), caption=cap, use_container_width=True)
        else:
            st.warning(f"Figur belum ada: {fname} (jalankan `python results/make_figures.py`).")
