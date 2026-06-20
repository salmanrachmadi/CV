# Deteksi Penggunaan Helm Pengendara Motor

Riset *computer vision* untuk mendeteksi penggunaan helm pada pengendara sepeda motor
(deteksi objek 3 kelas: `helmet`, `license_plate`, `motorcyclist`). Studi ini menguji
dua hal: **(1)** kemampuan *vanilla* (zero-shot COCO) YOLOv8s vs YOLO11s, dan **(2)** hasil
**fine-tuning** YOLOv8s vs YOLO11s vs YOLOv8s+CBAM. Seluruh pipeline berbasis **notebook**.

## Pengujian inti

1. **Baseline *vanilla* (zero-shot COCO)** — **YOLOv8s vs YOLO11s** dievaluasi *apa adanya*
   (bobot pretrained COCO, **tanpa** fine-tuning). Karena `helmet`/`license_plate` tidak ada
   di COCO, AP kelas itu ≈ 0 *by design* (mengukur **gap domain**); `motorcyclist` dipetakan
   dari COCO `motorcycle`. mAP dihitung dengan `pycocotools`.
2. **Fine-tuning (transfer learning)** — **YOLOv8s vs YOLO11s vs YOLOv8s+CBAM** dilatih
   multi-seed pada dataset helm; menguji apakah atensi (C2PSA / CBAM) mengalahkan CNN murni.

- **Dataset:** Roboflow **NCKH 2023 / helmet-detection-project v19** (1.563 train / 140 val / 100 test, lisensi MIT) — [tautan & cara akses](#dataset).
- **Metrik:** mAP@0.5 & mAP@[.5:.95] pada split **test**; FPS untuk trade-off.

> RT-DETR **tidak lagi** menjadi bagian studi.

## Struktur

```
notebooks/    # SELURUH pipeline studi (pure-notebook, self-contained):
              #   01_baseline_vanilla   — data → vanilla zero-shot → eval → figur
              #   02a_finetune_yolov8s  — fine-tuning multi-seed (sekuensial)
              #   02b_finetune_yolo11s  — fine-tuning multi-seed (sekuensial)
              #   02c_finetune_cbam     — fine-tuning multi-seed (sekuensial)
              #   03_comparative_study  — agregasi mean±std + figur perbandingan
configs/      # HANYA arsitektur model non-standar: yolov8s_cbam.yaml
data/         # dataset (gitignored)
experiments/  # output per-run: weights, metrics.json/csv (vanilla_*, ft_*) — gitignored
results/      # figur & artefak laporan/paper
app/          # Streamlit: demo inferensi + dashboard (di luar pipeline studi)
scripts/      # alat bantu lepas (remap_verify_datasets.py untuk uji lintas-dataset)
```

## Dataset

Studi ini memakai dataset publik **Helmet Detection Project v19** dari Roboflow Universe
(workspace `nckh-2023`), lisensi **MIT**, format anotasi **YOLOv8**.

| Atribut | Nilai |
|---|---|
| Nama / versi | Helmet Detection Project — v19 (ekspor 2 Juni 2023) |
| Workspace | `nckh-2023` |
| Halaman dataset | <https://universe.roboflow.com/nckh-2023/helmet-detection-project> |
| URL versi (v19) | <https://universe.roboflow.com/nckh-2023/helmet-detection-project/dataset/19> |
| Lisensi | MIT |
| Total citra | 1.803 (1.563 train / 140 valid / 100 test) |
| Kelas (3) | `helmet`, `license_plate`, `motorcyclist` |
| Pra-proses | auto-orient (strip EXIF), resize 1280×720 (stretch) |

> Dataset **tidak di-commit** (lihat `.gitignore`). Unduh sendiri ke `data/helmet-roboflow/`.

**Cara akses — via Roboflow API (disarankan):**

```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")  # ambil dari akun Roboflow; simpan di .env, JANGAN hardcode
project = rf.workspace("nckh-2023").project("helmet-detection-project")
dataset = project.version(19).download("yolov8", location="data/helmet-roboflow")
```

**Cara akses — manual:** buka [URL versi v19](https://universe.roboflow.com/nckh-2023/helmet-detection-project/dataset/19)
→ *Download Dataset* → format **YOLOv8** → ekstrak ke `data/helmet-roboflow/` sehingga
strukturnya `train/`, `valid/`, `test/`, dan `data.yaml`.

## Bobot & hasil eksperimen

Folder `experiments/` (bobot `.pt`, `metrics.json/csv` per-run; ~459 MB) **tidak di-commit**
(lihat `.gitignore`). Unduh dari Google Drive berikut lalu letakkan di `experiments/`:

- **Google Drive:** <https://drive.google.com/drive/folders/1htGJTApUoEK-hpO8UgM941Rozn93s_Eb?usp=sharing>

Isi run (penamaan `vanilla_<arsitektur>` & `ft_<arsitektur>_seed<N>`):

```
experiments/
  vanilla_yolov8s/  vanilla_yolo11s/                       # zero-shot COCO (metrics)
  ft_yolov8s_seed{42,0,1}/                                 # fine-tuning YOLOv8s
  ft_yolo11s_seed{42,0,1}/                                 # fine-tuning YOLO11s
  ft_cbam_seed{42,0,1}/                                    # fine-tuning YOLOv8s+CBAM
  yolov8s.pt  yolo11s.pt                                   # bobot pra-latih COCO
```

Diperlukan untuk demo Streamlit dan reproduksi figur tanpa melatih ulang.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
# Pasang torch sesuai CUDA GPU lokal — lihat https://pytorch.org/get-started/locally/
pip install -r requirements.txt

# Verifikasi GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

## Alur kerja (notebook berurutan)

```bash
# 1) Baseline vanilla (cepat — zero-shot, tanpa training)
jupyter nbconvert --to notebook --execute --inplace notebooks/01_baseline_vanilla.ipynb

# 2) Fine-tuning per-model, BERURUTAN & MANDIRI (jangan paralel; clear GPU cache antar-run)
jupyter nbconvert --to notebook --execute --inplace notebooks/02a_finetune_yolov8s.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02b_finetune_yolo11s.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02c_finetune_cbam.ipynb

# 3) Studi komparatif (agregasi mean±std + figur)
jupyter nbconvert --to notebook --execute --inplace notebooks/03_comparative_study.ipynb
```

Tiap notebook *self-contained* (sel *setup* memuat util sendiri: seed, env, metrik,
`clear_gpu`). Hyperparameter ada di sel *config* tiap notebook fine-tuning.

## Demo & dashboard (Streamlit)

```bash
streamlit run app/streamlit_app.py
```

- `app/streamlit_app.py` — beranda; `app/pages/` — Demo Deteksi & Hasil Eksperimen.
- Demo memuat checkpoint dari `experiments/` (butuh `data/` & `experiments/` tersedia
  lokal; keduanya gitignored — lihat [Dataset](#dataset) & [Bobot & hasil eksperimen](#bobot--hasil-eksperimen)).

## Reproducibility

- Seed dipatok per run (`set_seed` di sel setup); cuDNN deterministik.
- Konvensi nama run: `vanilla_<arsitektur>` (zero-shot) & `ft_<arsitektur>_seed<N>`
  (fine-tuning). Metrik tersimpan di `experiments/<run>/metrics.{json,csv}`.
- **Fine-tuning multi-seed** seed `[42, 0, 1]` dijalankan berurutan + `clear_gpu()` antar-run.
- **Ubah satu variabel per run** untuk atribusi sebab-akibat.

## Rencana lanjutan

- YOLOv8m sebagai pembanding *size-matched*; multi-seed 5+ untuk klaim publikasi.
- Hyperparameter tuning per-arsitektur (agar perbandingan benar-benar adil).
- Ablasi CBAM sistematis (penempatan, from-scratch, LR terpisah).
- Pipeline two-stage (motor→helm), deteksi plat nomor + OCR, fine-tuning ke data CCTV Indonesia.
- Uji generalisasi lintas-dataset (`scripts/remap_verify_datasets.py`).
