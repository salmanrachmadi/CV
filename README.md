# Deteksi Penggunaan Helm Pengendara Motor — Baseline YOLOv8

Riset *computer vision* untuk mendeteksi penggunaan helm pada pengendara sepeda
motor. **Fase 1**: baseline YOLOv8 single-stage (transfer learning dari COCO)
pada dataset publik Roboflow. Lihat latar riset di
[results/riset-deteksi-helm-pengendara-motor.md](results/riset-deteksi-helm-pengendara-motor.md)
dan panduan kerja di [CLAUDE.md](CLAUDE.md).

## Struktur

```
configs/      # hyperparameter eksperimen (YAML)
src/          # kode reusable: seeding, env, data, metrics
scripts/      # entrypoint CLI: prepare_data, train, eval, infer
data/         # dataset (gitignored)
experiments/  # output per-run: checkpoint, log, metrik (gitignored)
results/      # artefak inferensi & laporan
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

## Reproducibility

- Seed dipatok (`src/seeding.py`); cuDNN deterministik via config `deterministic: true`.
- Tiap run menyimpan `config_used.yaml` + `env.json` (seed, versi torch/CUDA/GPU) di folder run.
- Konvensi nama run: `<task>_<arsitektur>_<dataset>_<tanggal>`.
- **Ubah satu variabel per run** untuk atribusi sebab-akibat.

## Rencana lanjutan (di luar Fase 1)

YOLO11 / RT-DETR, pipeline two-stage (motor→helm), multi-task + tracking
(HELMET dataset), deteksi plat nomor + OCR, dan fine-tuning ke data CCTV Indonesia.
