"""Halaman demo deteksi interaktif — pilih model, unggah/pilih gambar, lihat hasil."""
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app_utils import CLASS_NAMES, MODELS, load_model, sample_images  # noqa: E402

st.set_page_config(page_title="Demo Deteksi", page_icon="🔍", layout="wide")
st.title("🔍 Demo Deteksi Helm")


@st.cache_resource(show_spinner="Memuat model…")
def get_model(name: str):
    return load_model(name)


# --- Sidebar: pengaturan ---
with st.sidebar:
    st.header("Pengaturan")
    model_name = st.selectbox("Model", list(MODELS.keys()))
    st.caption(MODELS[model_name]["note"])
    conf = st.slider("Confidence minimum", 0.05, 0.95, 0.25, 0.05)
    iou = st.slider("IoU (NMS)", 0.1, 0.9, 0.45, 0.05)
    imgsz = st.select_slider("Ukuran inferensi", [640, 960, 1280], value=1280)

# --- Pilih sumber gambar ---
st.subheader("1. Pilih gambar")
up = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])

image = None
if up is not None:
    image = Image.open(up).convert("RGB")
else:
    samples = sample_images()
    if samples:
        st.caption("Atau pilih salah satu gambar contoh dari split uji:")
        names = [p.name[:24] + "…" for p in samples]
        idx = st.selectbox("Gambar contoh", range(len(samples)), format_func=lambda i: names[i])
        image = Image.open(samples[idx]).convert("RGB")
    else:
        st.warning("Tidak ada gambar contoh (dataset tidak tersedia). Unggah gambar untuk mencoba.")

# --- Deteksi ---
if image is not None:
    st.subheader("2. Hasil deteksi")
    run = st.button("▶️ Jalankan deteksi", type="primary")
    if run:
        try:
            model = get_model(model_name)
        except FileNotFoundError as e:
            st.error(f"Gagal memuat model: {e}")
            st.stop()

        with st.spinner("Mendeteksi…"):
            res = model.predict(np.array(image), conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]

        plotted = res.plot()[:, :, ::-1]  # BGR -> RGB
        col1, col2 = st.columns([3, 2])
        with col1:
            st.image(plotted, caption=f"{model_name} · conf≥{conf} · imgsz={imgsz}", use_container_width=True)
        with col2:
            n = len(res.boxes)
            st.metric("Objek terdeteksi", n)
            speed = res.speed.get("inference")
            if speed:
                st.metric("Waktu inferensi", f"{speed:.1f} ms", f"≈{1000/speed:.0f} FPS")

            if n:
                import pandas as pd

                rows = []
                for b in res.boxes:
                    cls = int(b.cls)
                    rows.append({
                        "kelas": CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else str(cls),
                        "confidence": float(b.conf),
                    })
                df = pd.DataFrame(rows).sort_values("confidence", ascending=False)
                st.dataframe(df.style.format({"confidence": "{:.3f}"}), hide_index=True, use_container_width=True)

                st.caption("Jumlah per kelas:")
                st.bar_chart(df["kelas"].value_counts())
            else:
                st.info("Tidak ada objek di atas ambang confidence. Coba turunkan slider.")
    else:
        st.image(image, caption="Pratinjau (klik 'Jalankan deteksi')", use_container_width=True)
