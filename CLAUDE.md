# CLAUDE.md — Computer Vision Research

Panduan untuk Claude Code saat bekerja di repositori riset *computer vision* ini.

## Tentang Proyek

Repositori ini berisi riset **deteksi penggunaan helm pada pengendara sepeda motor**:
pelatihan model, evaluasi, dan analisis. Fokus kerja adalah **reproducibility** dan
**eksperimen yang terlacak** (tercatat konfigurasi, data, dan metrik).

## Tujuan & Ruang Lingkup Riset

- **Task utama**: **deteksi objek** (3 kelas: `helmet`, `license_plate`, `motorcyclist`).
- **Pertanyaan riset**: apakah mekanisme atensi (C2PSA pada YOLO11, transformer pada
  RT-DETR, modul CBAM) meningkatkan akurasi deteksi helm dibanding CNN murni YOLOv8s?
- **Baseline**: **YOLOv8s** (Ultralytics, transfer learning dari COCO).
- **Metrik sukses**: **mAP@0.5** dan **mAP@[.5:.95]** pada split test; FPS untuk trade-off.
- **Temuan Fase 1**: tidak ada varian ber-atensi yang mengalahkan YOLOv8s; CBAM turun
  signifikan. Paper di `results/` (versi Inggris di `results/en/`).


## Struktur Repositori (konvensi yang diharapkan)

```
data/          # dataset mentah & terproses (TIDAK di-commit; lihat .gitignore)
notebooks/     # eksplorasi (.ipynb) — bukan untuk kode produksi
src/           # kode reusable: dataset, model, training loop, util
configs/       # file konfigurasi eksperimen (YAML/JSON)
experiments/   # output per-run: checkpoint, log, metrik
scripts/       # entrypoint CLI: train.py, eval.py, infer.py
results/       # tabel, figur, dan artefak untuk laporan/paper
```

## Aturan Kerja Penting

- **Jangan commit data atau bobot besar.** Dataset, checkpoint `.pt`/`.pth`/`.ckpt`,
  dan file `.npy` besar harus masuk `.gitignore`. Gunakan referensi path/URL atau
  DVC/Git-LFS bila perlu versi.
- **Setiap eksperimen harus reproducible**: catat *seed*, versi config, commit hash,
  dan versi dataset. Set seed untuk `random`, `numpy`, dan framework (PyTorch/TF).
- **Pisahkan eksplorasi dari kode reusable.** Logika yang dipakai ulang dari notebook
  harus dipindah ke `src/`, lalu di-*import* di notebook.
- **Konfigurasi lewat file, bukan hardcode.** Hyperparameter, path, dan flag ada di
  `configs/`, bukan tersebar di dalam kode.
- **Validasi sebelum klaim hasil.** Sebelum melaporkan metrik membaik, pastikan
  evaluasi memakai split test/val yang benar dan tidak ada kebocoran data (*leakage*).

## Lingkungan & Setup

```bash
# (sesuaikan dengan tooling repo: venv / conda / poetry / uv)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

- Catat versi kunci: `python`, framework DL, CUDA/cuDNN, dan driver GPU.
- Versi terkunci: Python, **PyTorch 2.8.0**, **Ultralytics 8.4.65**, **CUDA 12.8**, GPU **RTX 4090**.

## Alur Kerja Umum

```bash
# Training
python scripts/train.py --config configs/<eksperimen>.yaml

# Evaluasi
python scripts/eval.py --config configs/<eksperimen>.yaml --checkpoint experiments/<run>/best.pt

# Inferensi satu gambar
python scripts/infer.py --checkpoint <path> --input <gambar.jpg>
```

## Konvensi Eksperimen

- **Penamaan run**: `<task>_<arsitektur>_<dataset>_<tanggal>` (mis. `det_yolov8_coco_20260611`).
- **Logging**: gunakan satu sumber kebenaran untuk metrik (TensorBoard / Weights & Biases /
  CSV di `experiments/<run>/metrics.csv`). Jangan campur banyak sistem tanpa alasan.
- **Checkpoint**: simpan `last.pt` dan `best.pt`; sertakan config yang dipakai di folder run.
- **Membandingkan eksperimen**: ubah **satu variabel** per run bila ingin atribusi sebab-akibat.

## Penanganan Data

- Dokumentasikan asal dataset, lisensi, dan langkah preprocessing.
- Tetapkan split train/val/test yang **tetap** dan tercatat; jangan acak ulang antar run.
- Periksa keseimbangan kelas dan kualitas anotasi sebelum menyalahkan model.
- Augmentasi didefinisikan di config dan dilaporkan saat membandingkan hasil.

## Saat Membantu di Repo Ini, Claude Sebaiknya

- **Baca config & struktur dulu** sebelum menebak framework atau path.
- **Jaga determinisme**: jangan menambah sumber keacakan tanpa seed.
- **Jangan menjalankan training panjang** tanpa diminta; tawarkan *dry-run* / subset kecil dulu.
- **Laporkan hasil apa adanya**: jika metrik turun atau test gagal, sampaikan dengan angkanya.
- **Hindari menambah dependency berat** tanpa konfirmasi.
- Gunakan tipe data & device (CPU/GPU) yang konsisten; waspadai *silent* cast.

## Status Proyek

Fase 1 **selesai**: framework dikunci (PyTorch + Ultralytics), `requirements.txt` &
`.gitignore` ada, baseline YOLOv8s + ablation (YOLO11s, RT-DETR-l, CBAM) multi-seed
terlatih & dievaluasi, dan paper ditulis (ID + Inggris). Lihat `README.md` & `results/`.
