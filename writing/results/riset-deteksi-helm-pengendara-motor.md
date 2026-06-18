# Deteksi Objek Penggunaan Helm pada Pengendara Sepeda Motor: Tinjauan Dataset, Metode, dan Kandidat Replikasi

*Laporan Deep Research (ARS) — full mode | APA 7.0 | Tanggal: 2026-06-11*

---

## Abstrak

Penegakan kewajiban penggunaan helm pada pengendara sepeda motor merupakan masalah keselamatan jalan yang signifikan, khususnya di negara berkembang tempat sepeda motor menjadi moda transportasi utama. Laporan ini meninjau literatur *computer vision* untuk deteksi penggunaan helm pengendara motor dengan tiga sasaran: (1) mengidentifikasi dataset yang digunakan, (2) merinci metode/arsitektur deteksi objek beserta metriknya, dan (3) memilih kandidat penelitian yang paling *reproducible* untuk re-implementasi. Temuan utama: dua sumber daya benchmark menonjol dan dapat diakses publik, yaitu **HELMET dataset** (Myanmar; 91.000 frame teranotasi, kode tersedia) dan **AI City Challenge 2023 Track 5** (100+100 video, 7 kelas anotasi). Metode didominasi keluarga **YOLO** (v5–v11) untuk deteksi real-time, dengan **RetinaNet** dan pendekatan **multi-task learning + tracking** untuk kasus yang membedakan pengemudi vs. penumpang. Untuk replikasi, direkomendasikan memulai dari pipeline **YOLOv8 single-stage pada subset AI City Track 5** sebagai baseline, lalu naik ke **multi-task learning pada HELMET dataset** untuk reproduksi penuh termasuk *tracking*. *Kata kunci:* helmet detection, motorcyclist, object detection, YOLO, RetinaNet, dataset.

> ⚠️ **Catatan integritas sitasi:** Semua referensi di bawah berasal dari pencarian web dan diverifikasi keberadaannya (judul + venue + URL). Detail numerik yang **belum** dapat diverifikasi penuh dari sumber primer (mis. nomor halaman jurnal, nama lengkap penulis preprint tertentu) ditandai `[verifikasi lanjut]`. Sesuai aturan ARS, angka yang masih ragu **tidak** diklaim sebagai pasti.

---

## 1. Pendahuluan

Sepeda motor menyumbang proporsi besar kematian akibat kecelakaan lalu lintas, dan penggunaan helm adalah faktor protektif paling penting. Di banyak negara berkembang, data penggunaan helm minim sehingga menghambat penegakan dan kampanye yang tertarget (Siebert & Lin, 2020). *Computer vision* menawarkan otomatisasi pengamatan helm dari video CCTV/lalu lintas. Pertanyaan riset yang memandu laporan ini:

> *Dataset dan metode deteksi objek apa yang digunakan dalam penelitian deteksi penggunaan helm pengendara sepeda motor, dan penelitian mana yang paling layak direplikasi melalui implementasi ulang?*

Sub-pertanyaan: (a) dataset & anotasinya; (b) arsitektur model & metrik; (c) kelayakan replikasi.

---

## 2. Metodologi Tinjauan

Pencarian dilakukan pada mesin pencari akademik dan web (arXiv, IEEE Xplore, ScienceDirect, MDPI, repositori universitas Indonesia, OSF, GitHub) dengan kata kunci: *motorcycle/motorcyclist helmet detection, helmet violation, YOLO, RetinaNet, AI City Challenge Track 5, deteksi helm pengendara motor*. Sumber dinilai berdasarkan hierarki bukti (jurnal *peer-reviewed* > prosiding > preprint > tesis/jurnal mahasiswa) dan relevansi terhadap tiga sasaran. Fokus diberikan pada penelitian dengan dataset publik dan/atau kode terbuka karena tujuan akhir pengguna adalah replikasi.

---

## 3. Temuan 1 — Dataset

### 3.1 Dataset benchmark publik (prioritas untuk replikasi)

| Dataset | Sumber & Lokasi | Skala | Kelas / Anotasi | Akses |
|---|---|---|---|---|
| **HELMET dataset** | 12 titik observasi di Myanmar (2016) | 910 klip video (10 dtk, 10 fps, 1920×1080); **91.000 frame teranotasi**; **10.006 motor** individual | Bounding box motor + jumlah rider + penggunaan helm per-rider; mendukung *tracking* antar-frame | Publik via OSF (osf.io/4pwj8); kode di GitHub (Lin, 2020) |
| **AI City Challenge 2023 — Track 5** | Video lalu lintas (India) | **100 video latih + 100 video uji** (≈20 dtk, 10 fps, 1920×1080) | **7 kelas**: Motorbike (29.827), DHelmet (22.233), DNoHelmet (6.885), P1Helmet (97), P1NoHelmet (4.460), P2Helmet (0), P2NoHelmet (138) — membedakan *driver* (D) vs *passenger* (P1/P2) | Publik via registrasi challenge (aicitychallenge.org) |

Catatan: skema kelas AI City Track 5 secara eksplisit memisahkan pengemudi (D) dan penumpang (P1, P2) serta status helm masing-masing — berguna untuk deteksi pelanggaran per-individu. Ketidakseimbangan kelas sangat mencolok (mis. P2Helmet = 0 instance), yang menjadi tantangan pelatihan tersendiri (Naphade et al., 2023).

### 3.2 Dataset kustom dari penelitian lain

- **Konteks India:** dataset video 100 klip 1920×1080 @10 fps; serta dataset custom ±3.146 gambar berlabel dengan mAP dilaporkan tinggi (lihat §4).
- **Konteks Indonesia (relevan langsung untuk pengguna):**
  - CCTV ATCS Kota **Samarinda** → frame diolah & dianotasi 4 kelas (*helmet, no-helmet, rider, motorcycle*), dilatih *transfer learning* dengan YOLO11 (Universitas Jambi, 2025).
  - CCTV lalu lintas **Yogyakarta** (6 simpang) + dataset publik Roboflow → **9.882 gambar**, dilatih YOLOv8n (UPN "Veteran" Yogyakarta, 2024).
  - Studi perbandingan **YOLOv8 vs RT-DETR** dengan citra konteks Indonesia (300 citra latih + 60 validasi) (Kurniawan dkk., 2025) `[verifikasi lanjut: penulis]`.

> **Implikasi data:** Untuk replikasi yang bersih, gunakan dataset publik berlisensi jelas (HELMET / AI City). Dataset CCTV Indonesia bagus untuk *domain adaptation* lokal, tetapi sering tidak dirilis publik dan lisensinya tidak eksplisit — dokumentasikan asal & izin sebelum dipakai (lihat CLAUDE.md repo: *Penanganan Data*).

---

## 4. Temuan 2 — Metode & Arsitektur

### 4.1 Pola arsitektur yang dominan

1. **Single-stage detector (YOLO family)** — pilihan paling umum karena real-time. Digunakan dari YOLOv2 hingga YOLOv8/v10/v11. Cocok untuk deteksi langsung kelas *helmet/no-helmet*.
2. **Two-stage pipeline (deteksi motor → deteksi helm)** — deteksi sepeda motor lebih dulu, lalu region motor di-*crop* dan diklasifikasi/deteksi helmnya. Mengurangi *false positive* dari helm pejalan kaki/objek lain (Jia et al., 2021).
3. **One-stage akurasi tinggi (RetinaNet)** — *feature pyramid* multi-skala + *focal loss* untuk menangani ketidakseimbangan kelas; dipakai pada studi skala besar Myanmar (Siebert & Lin, 2020).
4. **Multi-task learning + tracking** — satu model untuk identifikasi & *tracking* motor antar-frame sekaligus registrasi helm per-rider; membedakan pengemudi vs penumpang (Lin et al., 2020).
5. **Pipeline penegakan hukum (helm + plat nomor)** — deteksi pelanggar → lokalisasi plat → OCR (mis. Tesseract) untuk identifikasi kendaraan; sering ditambah deteksi *triple riding* (boncengan tiga).

### 4.2 Ringkasan metrik yang dilaporkan

| Studi | Arsitektur | Dataset | Metrik utama (dilaporkan) |
|---|---|---|---|
| Siebert & Lin (2020) | RetinaNet | HELMET (Myanmar), 91k frame | Pendekatan deteksi helm skala besar; *Accident Analysis & Prevention* |
| Lin et al. (2020) | CNN multi-task learning + tracking | HELMET | **Weighted avg F-measure ≈ 67,3%**, **>8 FPS** (consumer hardware) |
| Jia et al. (2021) | Improved YOLOv5 (triplet attention + soft-NMS), two-stage | Lalu lintas urban | Deteksi helm real-time; *IET Image Processing* |
| GA-Enhanced YOLOv5 (2023) | YOLOv5 + genetic algorithm | AI City Track 5 | **mAP ≈ 0,6667**, peringkat ke-4 *public leaderboard* `[verifikasi lanjut: penulis]` |
| Few-Shot + YOLOv8 (2023) | YOLOv8 + few-shot sampling | AI City Track 5 | Deteksi multi-kelas real-time; arXiv 2304.08256 |
| "Legends" — YOLOv8+TTA (2023) | YOLOv8 + test-time augmentation | AI City Track 5 | **mAP ≈ 0,5861**, **95 FPS**, peringkat ke-7 |
| MDPI Algorithms (2024) | YOLOv8 + DCGAN (augmentasi sintetis) | Custom | Mengatasi kelas minoritas via citra sintetis; mAP tinggi |
| UPN Yogyakarta (2024) | YOLOv8n (transfer learning) | CCTV Yogyakarta, 9.882 img | **Akurasi 79,49%, presisi 94,79%, recall 76,47%** |
| Top team CTCAI (2023) | (ensembel) | AI City Track 5 | **Skor 0,8340** (juara *public leaderboard*) |

*Pola umum:* presisi sering tinggi (>90%) sementara *recall* lebih rendah pada kondisi nyata (oklusi, gerak cepat, pencahayaan buruk, sudut pandang lebar). Ketidakseimbangan kelas (sedikit contoh "tanpa helm" / penumpang) adalah hambatan berulang, yang ditangani lewat augmentasi, *focal loss*, GAN, atau *few-shot sampling*.

---

## 5. Temuan 3 — Sintesis & Gap

- **Konvergen:** YOLO real-time = standar de-facto; akurasi benchmark naik tiap generasi YOLO; *transfer learning* dari bobot COCO adalah praktik baku.
- **Divergen:** ada *trade-off* antara deteksi langsung (cepat, sederhana) vs pipeline two-stage/multi-task (lebih akurat untuk per-rider & tracking, lebih kompleks).
- **Gap penelitian:**
  1. Generalisasi lintas-domain (model terlatih di satu kota/negara turun performanya di tempat lain) — relevan untuk konteks Indonesia.
  2. Kelas minoritas (penumpang tanpa helm, boncengan tiga) masih lemah.
  3. Pelaporan tidak seragam (mAP@0.5 vs mAP@[.5:.95], definisi "akurasi" berbeda) menyulitkan perbandingan adil.
  4. Sedikit dataset Indonesia berlisensi terbuka — peluang kontribusi.

---

## 6. Rekomendasi Replikasi (untuk re-implementasi dari awal)

### 🥇 Jalur A — Baseline cepat & paling reproducible: **YOLOv8 single-stage pada AI City Track 5**
- **Mengapa:** dataset publik dengan anotasi 7-kelas siap pakai; banyak metode terbuka (arXiv 2304.08256 few-shot YOLOv8; arXiv 2304.09248 GA-YOLOv5; repo GitHub `vnptai/AI-City-Challenge-2023`) sebagai pembanding; YOLOv8 punya implementasi resmi (Ultralytics) yang matang.
- **Yang dibutuhkan:**
  - *Data:* registrasi AI City Challenge → unduh Track 5 (100+100 video) → ekstrak frame.
  - *Framework:* Python + PyTorch + Ultralytics YOLOv8.
  - *Hyperparameter awal:* imgsz=1280–1920 (objek kecil), epochs 50–100, batch sesuai VRAM, augmentasi mosaic/HSV, transfer learning dari `yolov8s/m.pt`.
  - *Hardware:* 1 GPU (≥8–12 GB VRAM, mis. RTX 3060/3090) cukup untuk model s/m.
  - *Metrik:* mAP@0.5 dan mAP@[.5:.95] per kelas; laporkan FPS.
  - *Tantangan:* ketidakseimbangan kelas ekstrem → terapkan *few-shot sampling*/augmentasi/oversampling.

### 🥈 Jalur B — Reproduksi penuh dengan tracking: **Multi-task learning pada HELMET dataset**
- **Mengapa:** dataset **dan kode** tersedia (OSF + GitHub `LinHanhe/Helmet_use_detection`); mereproduksi metrik benchmark (F-measure 67,3%, >8 FPS) dan kemampuan membedakan pengemudi/penumpang + *tracking* — fitur yang tidak ada di baseline sederhana.
- **Yang dibutuhkan:** unduh HELMET dari OSF; ikuti repo (deep learning, berbasis CNN multi-task); siapkan pipeline *tracking*; GPU setara Jalur A. *Risiko:* kode lebih lama → mungkin perlu penyesuaian dependency/versi framework.

> **Saran alur:** Mulai **Jalur A** untuk mendapatkan baseline YOLOv8 yang berjalan dan terukur dalam waktu singkat, lalu lanjut **Jalur B** bila butuh *tracking* & atribusi per-rider. Untuk konteks lokal, lakukan *fine-tuning* pada data CCTV Indonesia setelah baseline stabil (ubah **satu variabel per run**, sesuai konvensi eksperimen repo).

---

## 7. Keterbatasan

- Beberapa metrik berasal dari abstrak/ringkasan, bukan pembacaan penuh teks; perbedaan protokol evaluasi antar-studi membatasi perbandingan langsung.
- Sebagian sumber Indonesia adalah tesis/jurnal mahasiswa (tier lebih rendah) — relevan untuk konteks lokal namun perlu dibaca kritis.
- Detail bertanda `[verifikasi lanjut]` (nama penulis preprint, nomor halaman) perlu konfirmasi dari sumber primer sebelum dikutip dalam paper formal.

## 8. Pernyataan Penggunaan AI

Laporan ini disusun dengan bantuan alat riset berbasis AI (Claude Code + skill Academic Research Skills/deep-research). Seluruh sumber diverifikasi keberadaannya melalui pencarian web; pengguna tetap bertanggung jawab memverifikasi sumber primer sebelum publikasi.

---

## Referensi (APA 7.0)

> Status: keberadaan tiap entri diverifikasi via judul + venue + URL. `[verifikasi lanjut]` = detail tertentu (penulis/halaman/volume) perlu konfirmasi sumber primer.

Jia, W., et al. (2021). Real-time automatic helmet detection of motorcyclists in urban traffic using improved YOLOv5 detector. *IET Image Processing, 15*(14). https://doi.org/10.1049/ipr2.12295 `[verifikasi lanjut: daftar penulis lengkap & halaman]`

Lin, H., Deng, J. D., Albers, D., & Siebert, F. W. (2020). Helmet use detection of tracked motorcycles using CNN-based multi-task learning. *IEEE Access, 8*. https://doi.org/10.1109/ACCESS.2020.3021357 `[verifikasi lanjut: nomor halaman]` — Kode: https://github.com/LinHanhe/Helmet_use_detection

Lin, H. (2020). *HELMET dataset* [Data set]. Open Science Framework. https://osf.io/4pwj8/

Naphade, M., et al. (2023). The 7th AI City Challenge. *CVPR Workshops (CVPRW)*. https://www.aicitychallenge.org/2023-challenge-tracks/ `[verifikasi lanjut: daftar penulis & sitasi resmi]`

Siebert, F. W., & Lin, H. (2020). Detecting motorcycle helmet use with deep learning. *Accident Analysis & Prevention, 134*, 105319. https://doi.org/10.1016/j.aap.2019.105319 (Preprint: arXiv:1910.13232)

Tsai, C.-Y., et al. (2023). Video analytics for detecting motorcyclist helmet rule violations. *CVPR Workshops (CVPRW)*. https://openaccess.thecvf.com/content/CVPR2023W/AICity/papers/Tsai_Video_Analytics_for_Detecting_Motorcyclist_Helmet_Rule_Violations_CVPRW_2023_paper.pdf `[verifikasi lanjut: penulis]`

*Real-time multi-class helmet violation detection using few-shot data sampling technique and YOLOv8.* (2023). arXiv:2304.08256. https://arxiv.org/abs/2304.08256 `[verifikasi lanjut: penulis]`

*Real-time helmet violation detection in AI City Challenge 2023 with genetic algorithm-enhanced YOLOv5.* (2023). arXiv:2304.09248. https://arxiv.org/abs/2304.09248 `[verifikasi lanjut: penulis]`

*Enforcing traffic safety: A deep learning approach for detecting motorcyclists' helmet violations using YOLOv8 and DCGAN-generated images.* (2024). *Algorithms, 17*(5), 202. https://doi.org/10.3390/a17050202 `[verifikasi lanjut: penulis]`

*Computer-vision based automatic rider helmet violation detection and vehicle identification in Indian smart city scenarios using NVIDIA TAO toolkit and YOLOv8.* (2025). PMC12321817. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12321817/ `[verifikasi lanjut: penulis & jurnal]`

### Sumber konteks Indonesia
Universitas Jambi. (2025). *Deteksi penggunaan helm pada pengendara sepeda motor dengan YOLO11* [Skripsi]. Repository Unja. https://repository.unja.ac.id/83381/

UPN "Veteran" Yogyakarta. (2024). *Deteksi pengendara motor tanpa menggunakan helm dengan algoritma YOLOv8n berbasis CNN* [Skripsi]. eprints UPNYK. http://eprints.upnyk.ac.id/41638/

*Deteksi helm pengendara dan plat nomor kendaraan pada CCTV lampu lalu lintas menggunakan algoritma YOLO.* (2024). ResearchGate. https://www.researchgate.net/publication/379489611

*Sistem deteksi penggunaan helm pada pengendara sepeda motor di Indonesia menggunakan perbandingan model YOLOv8 dan RT-DETR.* (2025). ResearchGate. https://www.researchgate.net/publication/398377632
