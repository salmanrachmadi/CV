# Deteksi Penggunaan Helm Pengendara Motor

Riset *computer vision* untuk mendeteksi penggunaan helm pada pengendara sepeda motor
(deteksi objek 3 kelas: `helmet`, `license_plate`, `motorcyclist`). Selain baseline
YOLOv8s, repo ini membandingkan beberapa **mekanisme atensi** dan merangkum hasilnya
dalam sebuah paper. Lihat latar riset di
[results/riset-deteksi-helm-pengendara-motor.md](results/riset-deteksi-helm-pengendara-motor.md)
dan panduan kerja di [CLAUDE.md](CLAUDE.md).

## Status & temuan utama

- **Dataset:** Roboflow **NCKH 2023 / helmet-detection-project v19** (1.563 train / 140 val / 100 test, lisensi MIT).
- **Eksperimen selesai:** YOLOv8s (baseline), YOLO11s (C2PSA), RT-DETR-l (transformer), YOLOv8s+CBAM — semua dievaluasi multi-seed.
- **Hasil test (mAP@0.5):** YOLOv8s **0,9592** > RT-DETR-l 0,9588 > YOLO11s 0,9569 > YOLOv8s+CBAM 0,9479.
- **Kesimpulan:** pada dataset ini **tidak ada mekanisme atensi yang mengalahkan YOLOv8s CNN-murni**; CBAM justru turun signifikan (uji-t berpasangan p=0,008). YOLOv8s = pemenang praktis (terakurat sekaligus tercepat, ~296 FPS).

## Struktur

```
configs/      # hyperparameter eksperimen (YAML): baseline_yolov8, ablation_yolo11, yolov8s_cbam
src/          # kode reusable: seeding, env, data, metrics
scripts/      # entrypoint CLI: prepare_data, train, eval, infer
notebooks/    # analisis hasil: 01 baseline, 02 ablation YOLO11, 03 multi-seed 3-model, 04 atensi CBAM
data/         # dataset (gitignored)
experiments/  # output per-run: checkpoint, log, metrik (gitignored)
results/      # laporan riset, paper (+ figur, peer-review); versi Inggris di results/en/
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
# Pasang torch sesuai CUDA GPU lokal — lihat https://pytorch.org/get-started/locally/
pip install -r requirements.txt

# Verifikasi GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python src/env.py
```

## Alur kerja

```bash
# 1) Siapkan dataset (export Roboflow format YOLOv8 ke data/helmet-roboflow)
#    Opsi A — unduh via SDK:
export ROBOFLOW_API_KEY=<kunci-anda>
python scripts/prepare_data.py --roboflow --workspace <ws> --project <proj> --version 1
#    Opsi B — validasi export manual yang sudah di data/:
python scripts/prepare_data.py --dataset-dir data/helmet-roboflow

# 2) Dry-run (WAJIB sebelum training panjang — CLAUDE.md)
python scripts/train.py --config configs/baseline_yolov8.yaml --epochs 2 --subset 0.05

# 3) Training penuh
python scripts/train.py --config configs/baseline_yolov8.yaml

# 4) Evaluasi di split test
python scripts/eval.py --config configs/baseline_yolov8.yaml \
    --checkpoint experiments/<run>/weights/best.pt

# 5) Inferensi satu gambar
python scripts/infer.py --checkpoint experiments/<run>/weights/best.pt --input contoh.jpg
```

## Demo & dashboard (Streamlit)

Aplikasi interaktif untuk menampilkan hasil model — demo deteksi (pilih model,
unggah/pilih gambar) + dashboard perbandingan eksperimen:

```bash
streamlit run app/streamlit_app.py
```

- `app/streamlit_app.py` — beranda; `app/pages/` — Demo Deteksi & Hasil Eksperimen.
- Demo memuat checkpoint dari `experiments/` (butuh `data/` & `experiments/` tersedia
  secara lokal; keduanya gitignored).

## Eksperimen perbandingan & analisis

Perbandingan arsitektur (YOLO11s, RT-DETR-l, YOLOv8s+CBAM) dijalankan multi-seed lewat
notebook di `notebooks/` (mengimpor util dari `src/`). Setiap notebook melatih bila
checkpoint belum ada (skip-if-exists) lalu mengevaluasi di split test:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/03_multiseed_yolov8_vs_yolo11.ipynb
```

- Config arsitektur khusus: `configs/ablation_yolo11.yaml`, `configs/yolov8s_cbam.yaml`
  (CBAM memerlukan registrasi modul ke parser Ultralytics — lihat sel setup notebook 04).
- Figur publikasi dihasilkan ulang via `python results/make_figures.py`.
- **Paper** lengkap (IMRaD, IEEE): `results/paper-deteksi-helm-revisi.md` (ID) dan
  `results/en/paper-deteksi-helm-revisi.md` (Inggris); peer-review & response-to-reviewers
  juga di `results/`.

## Reproducibility

- Seed dipatok (`src/seeding.py`); cuDNN deterministik via config `deterministic: true`.
- Tiap run menyimpan `config_used.yaml` + `env.json` (seed, versi torch/CUDA/GPU) di folder run.
- Konvensi nama run: `<task>_<arsitektur>_<dataset>_<tanggal>`.
- **Ubah satu variabel per run** untuk atribusi sebab-akibat.

## Rencana lanjutan

Sudah selesai pada Fase 1: baseline YOLOv8s + perbandingan atensi (YOLO11s, RT-DETR-l,
YOLOv8s+CBAM) multi-seed dan penulisan paper. Arah berikutnya:

- YOLOv8m sebagai pembanding *size-matched*; multi-seed 5+ untuk klaim publikasi.
- Hyperparameter tuning per-arsitektur (agar perbandingan benar-benar adil).
- Ablasi CBAM sistematis (penempatan, from-scratch, LR terpisah).
- Pipeline two-stage (motor→helm), deteksi plat nomor + OCR, fine-tuning ke data CCTV Indonesia.
