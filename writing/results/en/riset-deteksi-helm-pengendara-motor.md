# Object Detection of Helmet Use among Motorcyclists: A Review of Datasets, Methods, and Replication Candidates

*Deep Research Report (ARS) — full mode | APA 7.0 | Date: 2026-06-11*

---

## Abstract

Enforcing mandatory helmet use among motorcyclists is a significant road-safety problem, particularly in developing countries where motorcycles are the primary mode of transportation. This report reviews the *computer vision* literature on detecting motorcyclist helmet use with three objectives: (1) to identify the datasets used, (2) to detail the object-detection methods/architectures together with their metrics, and (3) to select the most *reproducible* research candidate for re-implementation. Key findings: two benchmark resources stand out and are publicly accessible, namely the **HELMET dataset** (Myanmar; 91,000 annotated frames, code available) and the **AI City Challenge 2023 Track 5** (100+100 videos, 7 annotation classes). The methods are dominated by the **YOLO** family (v5–v11) for real-time detection, with **RetinaNet** and a **multi-task learning + tracking** approach for cases that distinguish drivers from passengers. For replication, it is recommended to start from a **YOLOv8 single-stage pipeline on a subset of AI City Track 5** as a baseline, then move up to **multi-task learning on the HELMET dataset** for full reproduction including *tracking*. *Keywords:* helmet detection, motorcyclist, object detection, YOLO, RetinaNet, dataset.

> ⚠️ **Citation integrity note:** All references below are derived from web searches and verified for existence (title + venue + URL). Numerical details that **cannot** yet be fully verified from primary sources (e.g., journal page numbers, the full author names of certain preprints) are marked `[further verification]`. In accordance with the ARS rules, numbers that remain in doubt are **not** claimed as certain.

---

## 1. Introduction

Motorcycles account for a large proportion of traffic-accident fatalities, and helmet use is the single most important protective factor. In many developing countries, data on helmet use are scarce, which hampers enforcement and targeted campaigns (Siebert & Lin, 2020). *Computer vision* offers a way to automate the observation of helmet use from CCTV/traffic video. The research question guiding this report is:

> *What datasets and object-detection methods are used in research on detecting motorcyclist helmet use, and which study is the most feasible to replicate through re-implementation?*

Sub-questions: (a) the datasets & their annotations; (b) model architectures & metrics; (c) replication feasibility.

---

## 2. Review Methodology

Searches were conducted on academic and web search engines (arXiv, IEEE Xplore, ScienceDirect, MDPI, Indonesian university repositories, OSF, GitHub) with the keywords: *motorcycle/motorcyclist helmet detection, helmet violation, YOLO, RetinaNet, AI City Challenge Track 5, deteksi helm pengendara motor*. Sources were assessed based on an evidence hierarchy (*peer-reviewed* journals > proceedings > preprints > theses/student journals) and relevance to the three objectives. Priority was given to studies with public datasets and/or open code, because the user's ultimate goal is replication.

---

## 3. Finding 1 — Datasets

### 3.1 Public benchmark datasets (priority for replication)

| Dataset | Source & Location | Scale | Classes / Annotations | Access |
|---|---|---|---|---|
| **HELMET dataset** | 12 observation sites in Myanmar (2016) | 910 video clips (10 s, 10 fps, 1920×1080); **91,000 annotated frames**; **10,006 individual** motorcycles | Motorcycle bounding box + rider count + per-rider helmet use; supports inter-frame *tracking* | Public via OSF (osf.io/4pwj8); code on GitHub (Lin, 2020) |
| **AI City Challenge 2023 — Track 5** | Traffic video (India) | **100 training videos + 100 test videos** (≈20 s, 10 fps, 1920×1080) | **7 classes**: Motorbike (29,827), DHelmet (22,233), DNoHelmet (6,885), P1Helmet (97), P1NoHelmet (4,460), P2Helmet (0), P2NoHelmet (138) — distinguishing *driver* (D) vs *passenger* (P1/P2) | Public via challenge registration (aicitychallenge.org) |

Note: the AI City Track 5 class scheme explicitly separates drivers (D) and passengers (P1, P2) along with each one's helmet status — useful for per-individual violation detection. The class imbalance is highly conspicuous (e.g., P2Helmet = 0 instances), which constitutes a training challenge in its own right (Naphade et al., 2023).

### 3.2 Custom datasets from other studies

- **Indian context:** a video dataset of 100 clips at 1920×1080 @10 fps; plus a custom dataset of approximately 3,146 labeled images with high reported mAP (see §4).
- **Indonesian context (directly relevant to the user):**
  - Samarinda City ATCS CCTV → frames processed & annotated with 4 classes (*helmet, no-helmet, rider, motorcycle*), trained via *transfer learning* with YOLO11 (Universitas Jambi, 2025).
  - Yogyakarta traffic CCTV (6 intersections) + public Roboflow dataset → **9,882 images**, trained with YOLOv8n (UPN "Veteran" Yogyakarta, 2024).
  - A comparative study of **YOLOv8 vs RT-DETR** with Indonesian-context images (300 training images + 60 validation) (Kurniawan et al., 2025) `[further verification: authors]`.

> **Data implications:** For clean replication, use public datasets with clear licensing (HELMET / AI City). Indonesian CCTV datasets are good for local *domain adaptation*, but they are often not publicly released and their licensing is not explicit — document their provenance & permissions before use (see the repo's CLAUDE.md: *Data Handling*).

---

## 4. Finding 2 — Methods & Architectures

### 4.1 Dominant architectural patterns

1. **Single-stage detector (YOLO family)** — the most common choice because it is real-time. Used from YOLOv2 through YOLOv8/v10/v11. Suitable for direct detection of the *helmet/no-helmet* classes.
2. **Two-stage pipeline (motorcycle detection → helmet detection)** — the motorcycle is detected first, then the motorcycle region is *cropped* and its helmet is classified/detected. This reduces *false positives* from pedestrian helmets/other objects (Jia et al., 2021).
3. **High-accuracy one-stage (RetinaNet)** — a multi-scale *feature pyramid* + *focal loss* to handle class imbalance; used in the large-scale Myanmar study (Siebert & Lin, 2020).
4. **Multi-task learning + tracking** — a single model for identifying & *tracking* motorcycles across frames while simultaneously registering helmet use per rider; distinguishing drivers from passengers (Lin et al., 2020).
5. **Law-enforcement pipeline (helmet + license plate)** — detect violators → localize the license plate → OCR (e.g., Tesseract) for vehicle identification; often augmented with *triple riding* detection (three people on one motorcycle).

### 4.2 Summary of reported metrics

| Study | Architecture | Dataset | Main metric (reported) |
|---|---|---|---|
| Siebert & Lin (2020) | RetinaNet | HELMET (Myanmar), 91k frames | Large-scale helmet detection approach; *Accident Analysis & Prevention* |
| Lin et al. (2020) | CNN multi-task learning + tracking | HELMET | **Weighted avg F-measure ≈ 67.3%**, **>8 FPS** (consumer hardware) |
| Jia et al. (2021) | Improved YOLOv5 (triplet attention + soft-NMS), two-stage | Urban traffic | Real-time helmet detection; *IET Image Processing* |
| GA-Enhanced YOLOv5 (2023) | YOLOv5 + genetic algorithm | AI City Track 5 | **mAP ≈ 0.6667**, 4th place on the *public leaderboard* `[further verification: authors]` |
| Few-Shot + YOLOv8 (2023) | YOLOv8 + few-shot sampling | AI City Track 5 | Real-time multi-class detection; arXiv 2304.08256 |
| "Legends" — YOLOv8+TTA (2023) | YOLOv8 + test-time augmentation | AI City Track 5 | **mAP ≈ 0.5861**, **95 FPS**, 7th place |
| MDPI Algorithms (2024) | YOLOv8 + DCGAN (synthetic augmentation) | Custom | Addresses minority classes via synthetic images; high mAP |
| UPN Yogyakarta (2024) | YOLOv8n (transfer learning) | Yogyakarta CCTV, 9,882 img | **Accuracy 79.49%, precision 94.79%, recall 76.47%** |
| Top team CTCAI (2023) | (ensemble) | AI City Track 5 | **Score 0.8340** (*public leaderboard* winner) |

*General pattern:* precision is often high (>90%) while *recall* is lower under real-world conditions (occlusion, fast motion, poor lighting, wide viewing angles). Class imbalance (few "no-helmet" / passenger examples) is a recurring obstacle, addressed through augmentation, *focal loss*, GANs, or *few-shot sampling*.

---

## 5. Finding 3 — Synthesis & Gaps

- **Convergent:** real-time YOLO = the de-facto standard; benchmark accuracy rises with each YOLO generation; *transfer learning* from COCO weights is standard practice.
- **Divergent:** there is a *trade-off* between direct detection (fast, simple) vs. a two-stage/multi-task pipeline (more accurate for per-rider attribution & tracking, more complex).
- **Research gaps:**
  1. Cross-domain generalization (a model trained in one city/country drops in performance elsewhere) — relevant to the Indonesian context.
  2. Minority classes (passengers without helmets, triple riding) remain weak.
  3. Non-uniform reporting (mAP@0.5 vs mAP@[.5:.95], differing definitions of "accuracy") makes fair comparison difficult.
  4. Few openly licensed Indonesian datasets exist — an opportunity for contribution.

---

## 6. Replication Recommendations (for re-implementation from scratch)

### 🥇 Path A — Fast & most reproducible baseline: **YOLOv8 single-stage on AI City Track 5**
- **Why:** a public dataset with ready-to-use 7-class annotations; many open methods (arXiv 2304.08256 few-shot YOLOv8; arXiv 2304.09248 GA-YOLOv5; GitHub repo `vnptai/AI-City-Challenge-2023`) as benchmarks; YOLOv8 has a mature official implementation (Ultralytics).
- **What is needed:**
  - *Data:* register for the AI City Challenge → download Track 5 (100+100 videos) → extract frames.
  - *Framework:* Python + PyTorch + Ultralytics YOLOv8.
  - *Initial hyperparameters:* imgsz=1280–1920 (small objects), epochs 50–100, batch according to VRAM, mosaic/HSV augmentation, transfer learning from `yolov8s/m.pt`.
  - *Hardware:* 1 GPU (≥8–12 GB VRAM, e.g., RTX 3060/3090) is sufficient for s/m models.
  - *Metrics:* mAP@0.5 and mAP@[.5:.95] per class; report FPS.
  - *Challenge:* extreme class imbalance → apply *few-shot sampling*/augmentation/oversampling.

### 🥈 Path B — Full reproduction with tracking: **Multi-task learning on the HELMET dataset**
- **Why:** both the dataset **and the code** are available (OSF + GitHub `LinHanhe/Helmet_use_detection`); it reproduces the benchmark metrics (F-measure 67.3%, >8 FPS) and the ability to distinguish drivers/passengers + *tracking* — features absent from a simple baseline.
- **What is needed:** download HELMET from OSF; follow the repo (deep learning, CNN multi-task–based); set up the *tracking* pipeline; a GPU equivalent to Path A. *Risk:* the code is older → it may require dependency/framework-version adjustments.

> **Suggested workflow:** Start with **Path A** to obtain a running and measurable YOLOv8 baseline in a short time, then proceed to **Path B** if *tracking* & per-rider attribution are needed. For the local context, *fine-tune* on Indonesian CCTV data once the baseline is stable (change **one variable per run**, in line with the repo's experiment conventions).

---

## 7. Limitations

- Some metrics are derived from abstracts/summaries rather than full-text reading; differences in evaluation protocols across studies limit direct comparison.
- Some Indonesian sources are theses/student journals (a lower tier) — relevant to the local context but to be read critically.
- Details marked `[further verification]` (preprint author names, page numbers) require confirmation from primary sources before being cited in a formal paper.

## 8. AI Usage Statement

This report was prepared with the assistance of AI-based research tools (Claude Code + the Academic Research Skills/deep-research skill). All sources were verified for existence through web searches; the user remains responsible for verifying primary sources before publication.

---

## References (APA 7.0)

> Status: the existence of each entry has been verified via title + venue + URL. `[further verification]` = certain details (authors/pages/volume) require confirmation from primary sources.

Jia, W., et al. (2021). Real-time automatic helmet detection of motorcyclists in urban traffic using improved YOLOv5 detector. *IET Image Processing, 15*(14). https://doi.org/10.1049/ipr2.12295 `[further verification: full author list & pages]`

Lin, H., Deng, J. D., Albers, D., & Siebert, F. W. (2020). Helmet use detection of tracked motorcycles using CNN-based multi-task learning. *IEEE Access, 8*. https://doi.org/10.1109/ACCESS.2020.3021357 `[further verification: page numbers]` — Code: https://github.com/LinHanhe/Helmet_use_detection

Lin, H. (2020). *HELMET dataset* [Data set]. Open Science Framework. https://osf.io/4pwj8/

Naphade, M., et al. (2023). The 7th AI City Challenge. *CVPR Workshops (CVPRW)*. https://www.aicitychallenge.org/2023-challenge-tracks/ `[further verification: author list & official citation]`

Siebert, F. W., & Lin, H. (2020). Detecting motorcycle helmet use with deep learning. *Accident Analysis & Prevention, 134*, 105319. https://doi.org/10.1016/j.aap.2019.105319 (Preprint: arXiv:1910.13232)

Tsai, C.-Y., et al. (2023). Video analytics for detecting motorcyclist helmet rule violations. *CVPR Workshops (CVPRW)*. https://openaccess.thecvf.com/content/CVPR2023W/AICity/papers/Tsai_Video_Analytics_for_Detecting_Motorcyclist_Helmet_Rule_Violations_CVPRW_2023_paper.pdf `[further verification: authors]`

*Real-time multi-class helmet violation detection using few-shot data sampling technique and YOLOv8.* (2023). arXiv:2304.08256. https://arxiv.org/abs/2304.08256 `[further verification: authors]`

*Real-time helmet violation detection in AI City Challenge 2023 with genetic algorithm-enhanced YOLOv5.* (2023). arXiv:2304.09248. https://arxiv.org/abs/2304.09248 `[further verification: authors]`

*Enforcing traffic safety: A deep learning approach for detecting motorcyclists' helmet violations using YOLOv8 and DCGAN-generated images.* (2024). *Algorithms, 17*(5), 202. https://doi.org/10.3390/a17050202 `[further verification: authors]`

*Computer-vision based automatic rider helmet violation detection and vehicle identification in Indian smart city scenarios using NVIDIA TAO toolkit and YOLOv8.* (2025). PMC12321817. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12321817/ `[further verification: authors & journal]`

### Indonesian context sources
Universitas Jambi. (2025). *Deteksi penggunaan helm pada pengendara sepeda motor dengan YOLO11* [Skripsi]. Repository Unja. https://repository.unja.ac.id/83381/

UPN "Veteran" Yogyakarta. (2024). *Deteksi pengendara motor tanpa menggunakan helm dengan algoritma YOLOv8n berbasis CNN* [Skripsi]. eprints UPNYK. http://eprints.upnyk.ac.id/41638/

*Deteksi helm pengendara dan plat nomor kendaraan pada CCTV lampu lalu lintas menggunakan algoritma YOLO.* (2024). ResearchGate. https://www.researchgate.net/publication/379489611

*Sistem deteksi penggunaan helm pada pengendara sepeda motor di Indonesia menggunakan perbandingan model YOLOv8 dan RT-DETR.* (2025). ResearchGate. https://www.researchgate.net/publication/398377632
