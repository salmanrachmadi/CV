# CLAUDE.md — Computer Vision Research

Panduan untuk Claude Code saat bekerja di repositori riset *computer vision* ini.

## Tentang Proyek

Repositori ini berisi riset dan eksperimen *computer vision* (CV): pelatihan model,
evaluasi, dan analisis. Fokus kerja adalah **reproducibility** (hasil dapat
diulang) dan **eksperimen yang terlacak** (tercatat konfigurasi, data, dan metrik).

> Catatan: bagian-bagian di bawah berisi nilai default yang umum. Perbarui sesuai
> kondisi nyata repo (framework, dataset, struktur folder) begitu kode pertama masuk.

## Tujuan & Ruang Lingkup Riset

- **Task utama**: (isi salah satu) klasifikasi citra / deteksi objek / segmentasi /
  estimasi pose / OCR / image retrieval / generative.
- **Pertanyaan riset**: rumuskan satu kalimat hipotesis yang sedang diuji.
- **Baseline**: model/paper pembanding yang menjadi acuan.
- **Metrik sukses**: mis. Top-1/Top-5 accuracy, mAP@[.5:.95], mIoU, F1, FID.

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
- Verifikasi GPU sebelum training panjang: `python -c "import torch; print(torch.cuda.is_available())"`.

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

## TODO Awal (hapus setelah diisi)

- [ ] Tetapkan task, dataset, dan baseline.
- [ ] Pilih framework (PyTorch / TensorFlow / JAX) dan kunci versinya.
- [ ] Buat `requirements.txt` / environment file.
- [ ] Tambahkan `.gitignore` untuk `data/`, `experiments/`, bobot model.
- [ ] Tulis `scripts/train.py` dan `configs/` contoh pertama.
