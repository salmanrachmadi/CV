---
title: "Perbandingan Arsitektur Deteksi Objek Berbasis dan Tanpa Mekanisme Atensi untuk Deteksi Penggunaan Helm Pengendara Sepeda Motor"
author: "Penulis (sesuaikan)"
date: "2026"
lang: id
---

# Perbandingan Arsitektur Deteksi Objek Berbasis dan Tanpa Mekanisme Atensi untuk Deteksi Penggunaan Helm Pengendara Sepeda Motor

*Paper revisi — format IMRaD, sitasi IEEE. Disusun dengan bantuan ARS academic-paper (mode full). Revisi dibantu oleh Claude Code.*

---

## Abstrak (Bahasa Indonesia)

Penggunaan helm merupakan faktor protektif utama bagi pengendara sepeda motor, dan deteksi otomatis kepatuhannya melalui kamera lalu lintas dapat mendukung penegakan aturan keselamatan jalan. Mekanisme atensi (*attention*) telah menjadi tren dominan dalam visi komputer, namun manfaatnya untuk deteksi helm pada dataset berskala kecil belum banyak diuji secara terkontrol. Studi ini membandingkan empat arsitektur deteksi objek yang menempati spektrum penggunaan atensi: YOLOv8s (CNN murni, tanpa atensi), YOLO11s (atensi parsial melalui blok C2PSA), RT-DETR-l (transformer dengan atensi penuh), dan YOLOv8s+CBAM (CNN dengan modul atensi *channel*-*spatial*). Seluruh model dilatih pada dataset publik deteksi helm berisi 1.803 citra dengan tiga kelas (*helmet*, *license_plate*, *motorcyclist*), menggunakan protokol identik (*transfer learning* dari COCO, resolusi 1280, augmentasi sama) dan dievaluasi pada *split* uji dengan beberapa *seed* serta uji-t berpasangan untuk validitas statistik. Hasilnya, di bawah protokol pelatihan yang diseragamkan dan ditata untuk YOLO, tidak ada varian ber-atensi yang melampaui YOLOv8s baku: YOLOv8s mencapai mAP@0.5 tertinggi (0,9592 ± 0,0014) sekaligus kecepatan inferensi terbaik (~296 FPS). Uji-t berpasangan antar-*seed* menunjukkan penambahan modul CBAM menurunkan mAP@0.5 sebesar 0,0113 secara signifikan (t(2) = 10,9; p = 0,008; Cohen's *d* ≈ 9,1), sedangkan YOLO11s dan RT-DETR-l tidak berbeda signifikan dari baseline (p > 0,05). Karena baseline sudah mendekati 0,96 (efek langit-langit) dan jumlah *seed* terbatas (n = 2–3), daya statistik untuk mendeteksi perbaikan kecil sangat terbatas, sehingga klaim kesetaraan—khususnya RT-DETR-l yang hanya diuji pada dua *seed*—merupakan ketiadaan bukti perbedaan, bukan bukti kesetaraan. Temuan ini berlaku di bawah kondisi spesifik yang diuji: satu dataset kecil yang relatif mudah, protokol pelatihan yang ditata untuk YOLO, dan resolusi 1280 piksel. Pada kondisi tersebut, kompleksitas atensi tambahan tidak terbayar; arsitektur CNN sederhana tetap menjadi pilihan paling efisien untuk deteksi helm waktu-nyata.

**Kata kunci:** deteksi helm; deteksi objek; mekanisme atensi; YOLO; RT-DETR; CBAM; sepeda motor.

## Abstract (English)

Helmet use is the primary protective factor for motorcyclists, and automatic compliance detection through traffic cameras can support road-safety enforcement. Attention mechanisms have become a dominant trend in computer vision, yet their benefit for helmet detection on small-scale datasets remains under-examined in controlled settings. This study compares four object-detection architectures spanning the attention spectrum: YOLOv8s (pure CNN, no attention), YOLO11s (partial attention via C2PSA blocks), RT-DETR-l (a transformer with full attention), and YOLOv8s+CBAM (a CNN augmented with a channel-spatial attention module). All models were trained on a public helmet-detection dataset of 1,803 images with three classes (helmet, license_plate, motorcyclist) under an identical protocol (COCO transfer learning, 1280 resolution, matched augmentation) and evaluated on a held-out test split with several seeds and paired t-tests for statistical validity. Under a unified training protocol tuned for YOLO, no attention variant surpassed the plain YOLOv8s: it achieved the highest mAP@0.5 (0.9592 ± 0.0014) and the best inference speed (~296 FPS). A paired t-test across seeds showed that adding the CBAM module significantly reduced mAP@0.5 by 0.0113 (t(2) = 10.9; p = 0.008; Cohen's *d* ≈ 9.1), whereas YOLO11s and RT-DETR-l did not differ significantly from the baseline (p > 0.05). Because the baseline already approaches 0.96 (a ceiling effect) and the number of seeds is limited (n = 2–3), statistical power to detect small improvements is severely limited, so the equivalence claims—particularly for RT-DETR-l, tested on only two seeds—represent an absence of evidence rather than evidence of absence. These findings hold under the specific conditions tested: a single small, relatively easy dataset, a training protocol tuned for YOLO, and 1280-pixel resolution. Under these conditions, the added complexity of attention does not pay off; a simple CNN architecture remains the most efficient choice for real-time helmet detection.

**Keywords:** helmet detection; object detection; attention mechanism; YOLO; RT-DETR; CBAM; motorcycle.

---

## I. Pendahuluan

Sepeda motor menyumbang proporsi besar korban kecelakaan lalu lintas, khususnya di negara berkembang tempat moda ini mendominasi mobilitas harian. Helm adalah faktor protektif paling menentukan terhadap cedera kepala fatal, sehingga pemantauan kepatuhan penggunaan helm menjadi sasaran penting kebijakan keselamatan. Pengamatan manual tidak terskala untuk volume lalu lintas nyata: petugas tidak mungkin mengamati setiap simpang sepanjang waktu, dan survei sesaat memberi gambaran yang bias. Di sinilah deteksi objek otomatis dari rekaman kamera menawarkan jalan keluar yang dapat berjalan terus-menerus dan konsisten [1], [2].

Selama satu dekade terakhir, deteksi objek bergeser dari pendekatan dua-tahap yang akurat namun lambat ke detektor satu-tahap yang cepat, dengan keluarga YOLO sebagai standar de-facto untuk aplikasi waktu-nyata [5]. Secara paralel, mekanisme atensi yang lahir dari Transformer [10] dan diadaptasi ke citra melalui Vision Transformer [11] telah mengubah lanskap arsitektur. Detektor berbasis Transformer seperti DETR [8] dan turunan waktu-nyatanya RT-DETR [7] kini bersaing dengan CNN, sementara modul atensi ringan seperti Squeeze-and-Excitation [12] dan CBAM [9] menjadi cara populer menyisipkan atensi ke dalam *backbone* CNN yang sudah ada.

Tren penambahan atensi pada model deteksi helm telah menghasilkan sejumlah laporan positif. Zhang dkk. [19] melaporkan bahwa penyisipan modul atensi dan *feature fusion* pada YOLOv8 meningkatkan deteksi helm. Li dkk. [20] menunjukkan peningkatan serupa dengan modul atensi pada YOLOv5s, dan Jia dkk. [22] mendemonstrasikan bahwa kombinasi deformable attention dengan YOLOv5 memperbaiki performa deteksi helm. Laporan-laporan ini secara konsisten mengklaim manfaat atensi, namun umumnya membandingkan satu varian model tanpa kendali eksperimen yang ketat—tanpa pengulangan antar-*seed*, tanpa protokol pelatihan yang diseragamkan, dan tanpa uji signifikansi statistik. Akibatnya, perbedaan performa yang dilaporkan bisa berasal dari hyperparameter, augmentasi, atau keberuntungan inisialisasi—bukan dari mekanisme atensi itu sendiri.

Tren ini memunculkan asumsi implisit bahwa menambahkan atensi cenderung meningkatkan performa. Asumsi tersebut beralasan pada *benchmark* berskala besar seperti COCO, tempat keragaman dan volume data memberi ruang bagi atensi untuk mempelajari ketergantungan jarak-jauh yang bermanfaat. Namun pada konteks praktis deteksi helm—yang sering memakai dataset berskala kecil, terkurasi, dan dengan jumlah kelas terbatas—belum jelas apakah kompleksitas atensi benar-benar terbayar.

Studi ini menutup celah tersebut dengan satu pertanyaan riset terfokus: **apakah peningkatan yang dilaporkan oleh studi-studi ber-atensi pada deteksi helm dapat direplikasi dalam perbandingan terkendali dengan protokol seragam, pengulangan antar-*seed*, dan uji signifikansi?** Pertanyaan ini memosisikan studi ini sebagai uji terstruktur (*structured test*) terhadap generalisabilitas klaim manfaat atensi pada pengaturan skala kecil yang terkontrol.

Kontribusi utama paper ini:

1. **Perbandingan terkendali** empat arsitektur deteksi (YOLOv8s, YOLO11s, RT-DETR-l, YOLOv8s+CBAM) pada tugas deteksi helm dengan hyperparameter, data, dan augmentasi identik, sehingga selisih performa dapat diatribusikan ke faktor arsitektur, khususnya keberadaan dan jenis atensi.
2. **Evaluasi multi-*seed*** dengan pelaporan rata-rata ± simpangan baku, uji-t berpasangan, effect size Cohen's *d*, dan selang kepercayaan 95%, sehingga kesimpulan tidak rentan terhadap keberuntungan satu inisialisasi.
3. **Rebuttal terstruktur** terhadap asumsi bahwa menambahkan atensi selalu membantu: pada dataset ini dan di bawah protokol pelatihan yang ditata untuk YOLO, tidak ada mekanisme atensi yang melampaui CNN murni, dan modul CBAM bahkan menurunkan akurasi secara signifikan dan konsisten. Kami menelusuri penyebabnya, membahas hipotesis tandingan, dan merangkum implikasinya bagi praktik pemilihan model.

## II. Tinjauan Pustaka

### A. Deteksi penggunaan helm

Penelitian deteksi helm berkembang dari klasifikasi sederhana menuju pipeline yang mampu melacak motor antar-bingkai dan membedakan pengemudi dari penumpang. Siebert dan Lin [1] mendemonstrasikan deteksi penggunaan helm skala besar dari video lalu lintas Myanmar menggunakan pendekatan *deep learning* dengan detektor satu-tahap, dan merilis dataset beranotasi yang menjadi rujukan komunitas. Lin dkk. [2] memperluasnya dengan *multi-task learning* berbasis CNN yang melacak motor individual sekaligus meregistrasi penggunaan helm per-pengendara, sehingga mampu menangani kasus boncengan dan membedakan pengemudi dari penumpang.

Tantangan benchmark seperti AI City Challenge 2023 Track 5 [3] mendorong perhatian pada deteksi pelanggaran helm multi-kelas, termasuk pembedaan status helm pengemudi dan penumpang pertama serta kedua. Skema kelas yang rinci ini menyingkap masalah ketidakseimbangan kelas yang ekstrem, karena kategori seperti penumpang kedua tanpa helm sangat jarang muncul. Pada konteks penegakan, sejumlah pendekatan menggabungkan deteksi helm dengan lokalisasi plat nomor dan pengenalan karakter untuk identifikasi pelanggar [4], sering ditambah deteksi boncengan lebih dari dua orang.

Di Indonesia, tempat sepeda motor mendominasi moda transportasi, deteksi otomatis kepatuhan helm memiliki relevansi langsung. Hariyono dkk. [21] mendemonstrasikan aplikasi deteksi helm berbasis YOLOS yang diintegrasikan ke dalam antarmuka Streamlit untuk skenario penegakan di Indonesia, menunjukkan potensi deployment praktis. Raj dan Nair [24] mengusulkan pendekatan berbasis YOLO untuk mendeteksi pengendara tanpa helm melalui rekaman CCTV, menekankan tantangan oklusi dan variasi pencahayaan yang khas pada pengaturan lalu lintas nyata.

### B. Detektor satu-tahap dan keluarga YOLO

Detektor satu-tahap memformulasikan deteksi sebagai regresi langsung *bounding box* dan kelas dalam satu lintasan jaringan, menukar sebagian akurasi dengan kecepatan tinggi [5], dan kesenjangan akurasi ini sebagian dipersempit oleh fungsi rugi seperti *focal loss* yang menangani ketidakseimbangan kelas pada deteksi padat [13]. Keluarga YOLO menjadi tulang punggung aplikasi waktu-nyata, dengan iterasi yang terus memperbaiki *backbone*, *neck*, strategi penugasan label, dan augmentasi. YOLOv8 mewakili generasi CNN yang matang tanpa modul atensi eksplisit; arsitekturnya mengandalkan blok konvolusi efisien dan *feature pyramid* untuk menangani objek multi-skala. YOLO11 memperkenalkan blok atensi posisional (C2PSA) ke dalam jalur fitur, menjadikannya titik tengah yang menarik pada spektrum atensi—sebuah CNN yang menambahkan *self-attention* terbatas tanpa berpindah sepenuhnya ke paradigma transformer.

### C. Mekanisme atensi dalam visi komputer

Atensi memungkinkan model menimbang ulang informasi secara selektif, menonjolkan bagian yang relevan dan menekan yang tidak. Guo dkk. [17] menyurvei secara komprehensif mekanisme atensi dalam visi komputer dan mengklasifikasikannya ke dalam beberapa kategori: atensi kanal, spasial, temporal, dan *self-attention*, masing-masing dengan kelebihan dan batasan tersendiri. Survei Niu dkk. [18] meninjau atensi dari perspektif pembelajaran mendalam secara lebih luas dan menegaskan bahwa efektivitas atensi sangat bergantung pada volume data pelatihan dan kompleksitas tugas.

Transformer [10] memperkenalkan *self-attention* yang memodelkan ketergantungan global antar-elemen, dan Vision Transformer [11] membuktikan paradigma ini kompetitif untuk citra ketika data pelatihan cukup besar. Pada deteksi, DETR [8] memformulasikan deteksi sebagai prediksi himpunan dengan *encoder-decoder* Transformer dan menghapus komponen buatan-tangan seperti *anchor* dan *non-maximum suppression*. Kelemahan DETR—konvergensi lambat dan biaya komputasi tinggi—mendorong lahirnya RT-DETR [7] yang mengadaptasi ide tersebut menjadi waktu-nyata melalui *encoder* hibrida yang efisien. Arah lain memakai atensi pada tingkat *backbone*: Swin Transformer [15] memperkenalkan atensi berjendela hierarkis yang efisien sebagai *backbone* deteksi, sementara ViTDet [16] menunjukkan *backbone* ViT polos dapat dipakai untuk deteksi. Pendekatan-pendekatan ini umumnya unggul justru ketika data pelatihan berlimpah.

Pada sisi yang berbeda, modul atensi ringan menyisipkan atensi ke CNN tanpa mengganti arsitektur dasar. Squeeze-and-Excitation [12] memberi atensi antar-kanal dengan mempelajari bobot pentingnya setiap kanal fitur. CBAM [9] memperluasnya dengan menggabungkan atensi kanal dan atensi spasial secara berurutan, sehingga model dapat menekankan "apa" yang penting sekaligus "di mana". Modul-modul ini menarik karena murah, tidak menambah banyak parameter, dan mudah dipasang pada jaringan yang sudah ada. Namun efektivitasnya bergantung pada tugas dan skala data: pada dataset kecil, modul yang diinisialisasi acak harus belajar dari sinyal yang terbatas.

Untuk tugas deteksi objek kecil, Cheng dkk. [23] menyurvei pendekatan berskala besar dan menegaskan bahwa objek berukuran kecil seperti helm dan plat nomor dalam bingkai lebar merupakan tantangan khusus yang membutuhkan strategi fitur multi-skala, resolusi tinggi, dan augmentasi yang dirancang khusus. Temuan survei ini relevan dengan studi kami, mengingat kelas *helmet* dan *license_plate* pada dataset yang digunakan berukuran relatif kecil di dalam bingkai.

### D. Atensi pada deteksi helm

Sejumlah studi terbaru secara khusus mengintegrasikan mekanisme atensi ke dalam model deteksi helm. Zhang dkk. [19] mengusulkan modifikasi YOLOv8 dengan penambahan modul atensi dan strategi *feature fusion* yang diperluas, dan melaporkan peningkatan mAP pada dataset helm mereka. Li dkk. [20] menyisipkan modul atensi *channel-spatial* pada YOLOv5s dan melaporkan perbaikan akurasi deteksi helm. Jia dkk. [22] memperkenalkan deformable attention (DAAM) ke dalam arsitektur YOLOv5 dan mendemonstrasikan perbaikan pada tugas deteksi helm.

Meskipun laporan-laporan ini konsisten menunjukkan manfaat atensi, terdapat pola metodologis yang patut dicermati. Pertama, sebagian besar studi melaporkan hasil dari konfigurasi tunggal tanpa pengulangan antar-*seed*, sehingga tidak dapat disimpulkan apakah peningkatan yang dilaporkan konsisten di seluruh inisialisasi. Kedua, protokol pelatihan—meliputi augmentasi, resolusi, jumlah epoch, dan *learning rate*—umumnya di-*tuning* secara terpisah per-model, sehingga perbedaan performa mencerminkan gabungan efek arsitektur dan hyperparameter, bukan atensi semata. Ketiga, tidak ada studi yang melaporkan uji signifikansi statistik atas selisih performa. Celah ini menjadi motivasi langsung bagi studi kami, yang dirancang untuk mengisolasi kontribusi mekanisme atensi melalui protokol seragam, pengulangan multi-*seed*, dan pengujian statistik formal.

### E. Celah penelitian

Literatur deteksi helm cenderung melaporkan satu konfigurasi model dengan protokol yang berbeda-beda, sering tanpa pengulangan antar-*seed*, sehingga atribusi sebab-akibat ke mekanisme atensi menjadi lemah. Ketika sebuah studi melaporkan bahwa model ber-atensi mengungguli baseline, sulit memastikan apakah keunggulan itu berasal dari atensi atau dari perbedaan resolusi, augmentasi, jumlah epoch, atau inisialisasi. Studi-studi yang ditinjau di Bagian II.D tidak terkecuali: mereka melaporkan manfaat atensi namun tanpa kendali eksperimen yang diperlukan untuk mengatribusikan penyebab secara tegas. Studi ini mengisi celah tersebut melalui perbandingan terkendali dan berulang yang secara eksplisit memvariasikan keberadaan dan jenis atensi sebagai sumbu utama, sambil menahan seluruh faktor lain tetap konstan, termasuk hyperparameter, augmentasi, data, dan resolusi masukan.

## III. Metodologi

### A. Dataset

Kami menggunakan dataset publik deteksi helm dari Roboflow Universe ("NCKH 2023 / Helmet Detection Project", versi 19, lisensi MIT). Dataset berisi 1.803 citra beranotasi format YOLO dengan tiga kelas: *helmet*, *license_plate*, dan *motorcyclist*. Pembagian data tetap digunakan sepanjang eksperimen agar tidak ada keacakan ulang antar-*run*: 1.563 citra latih, 140 validasi, dan 100 uji. Seluruh metrik akhir dihitung pada *split* uji yang tidak pernah dilihat selama pelatihan untuk menghindari kebocoran data. Anotasi mencakup tiga kelas yang secara langsung relevan untuk skenario penegakan: pengendara sebagai konteks, helm sebagai objek kepatuhan, dan plat nomor sebagai jangkar identifikasi.

### B. Arsitektur yang dibandingkan

Empat arsitektur dipilih agar menempati titik berbeda pada spektrum penggunaan atensi (Tabel I). Pemilihan ini disengaja: dari CNN tanpa atensi sama sekali, ke CNN dengan atensi parsial, ke transformer dengan atensi penuh, dan akhirnya intervensi atensi terkendali yang hanya menambahkan modul pada baseline.

**TABEL I. Empat arsitektur dan posisinya pada spektrum atensi.**

| Model | Paradigma | Mekanisme atensi | Parameter |
|---|---|---|---|
| YOLOv8s | CNN satu-tahap | Tidak ada (baseline) | ~11,17 jt |
| YOLO11s | CNN satu-tahap | Parsial (blok C2PSA) | ~9,4 jt |
| RT-DETR-l | Transformer | Penuh (*self-attention*) | ~32 jt |
| YOLOv8s+CBAM | CNN + modul atensi | Kanal + spasial (CBAM) | ~11,51 jt |

Varian YOLOv8s+CBAM merupakan arsitektur kustom yang menjadi inti eksperimen atensi terkendali. Tiga modul CBAM disisipkan pada keluaran tiga skala deteksi (P3, P4, dan P5) tepat sebelum kepala deteksi, sehingga model identik dengan YOLOv8s baku kecuali penambahan atensi tersebut. Penempatan pada ketiga skala memastikan atensi bekerja pada fitur kecil (P3, untuk objek seperti helm dan plat), menengah (P4), dan besar (P5). Penambahan ini hanya menaikkan jumlah parameter dari 11,17 juta menjadi 11,51 juta—sekitar 3%—menjadikannya intervensi yang ringan dan terisolasi sehingga setiap perubahan performa dapat dikaitkan langsung ke modul atensi.

Perlu dicatat bahwa RT-DETR-l (~32 juta parameter) jauh lebih besar daripada varian YOLO-s (~9–11 juta), sehingga perbandingan ini mencampur faktor paradigma arsitektur dan kapasitas model. Pemilihan model standar ini disengaja untuk merefleksikan pilihan praktis yang dihadapi praktisi, namun keterbatasan ini diakui dan dibahas secara eksplisit di Bagian VI.

### C. Protokol pelatihan

Untuk memastikan perbedaan performa dapat diatribusikan ke arsitektur, seluruh model dilatih dengan protokol identik menggunakan kerangka Ultralytics [6] di atas PyTorch dan satu GPU NVIDIA RTX 4090. Setiap model diinisialisasi dengan *transfer learning* dari bobot praterlatih COCO [14]; untuk varian CBAM, lapisan atensi diinisialisasi acak sementara sisanya mewarisi bobot COCO. Konfigurasi seragam meliputi resolusi masukan 1280 piksel, penentuan ukuran *batch* otomatis sesuai memori GPU, pemilihan *optimizer* otomatis, augmentasi yang sama (*mosaic*, penggeseran *HSV*, dan pencerminan horizontal), serta penghentian dini dengan kesabaran 25 epoch dari maksimum 100 epoch.

Resolusi tinggi 1280 piksel dipilih karena analisis awal menunjukkan kelas *helmet* dan *license_plate* berukuran kecil di dalam bingkai, sehingga resolusi rendah merugikan deteksinya. Reproducibility ditegakkan dengan mematok *seed* untuk pustaka *random*, NumPy, dan PyTorch dalam mode deterministik, serta menyimpan konfigurasi dan informasi lingkungan (versi pustaka, *seed*, dan perangkat) pada setiap *run*. Untuk validitas statistik, setiap model dilatih ulang pada beberapa *seed* (42, 0, dan 1) dan dilaporkan rata-rata ± simpangan baku pada *split* uji. RT-DETR-l hanya dijalankan pada dua *seed* karena biaya pelatihannya jauh lebih tinggi, sekitar 190 menit per *run* dibanding sekitar 26 menit untuk varian YOLO; keterbatasan ini dicatat secara eksplisit dan diperhitungkan saat menafsirkan variansinya.

### D. Metrik evaluasi dan protokol pengujian statistik

Metrik utama adalah mAP@0.5 dan mAP@[.5:.95] sesuai konvensi deteksi objek, dilengkapi *precision*, *recall*, dan kecepatan inferensi dalam *frame per second* (FPS). Karena pengukuran FPS pada *sweep* multi-*seed* terpengaruh beban GPU yang berjalan beruntun, angka FPS yang dilaporkan diambil dari *run* tunggal saat GPU senggang (tidak ada proses training atau inferensi lain yang berjalan) menggunakan benchmark bawaan Ultralytics pada *split* test penuh, agar mencerminkan kecepatan sebenarnya pada konfigurasi perangkat keras yang digunakan. Pembedaan ini penting agar perbandingan kecepatan tidak menyesatkan akibat artefak pengukuran.

Untuk menilai signifikansi perbedaan akurasi, kami memasangkan hasil antar-model berdasarkan *seed* yang sama dan menerapkan **uji-t berpasangan** (*paired t-test*) pada selisih mAP@0.5 tiap pasangan *seed*, dengan taraf nyata α = 0,05. Selain nilai *p*, kami melaporkan *effect size* Cohen's *d* serta selang kepercayaan 95% (95% CI) untuk selisih rata-rata. Pemasangan berdasarkan *seed* mengendalikan variasi inisialisasi sehingga uji lebih sensitif terhadap efek arsitektur. Mengingat jumlah *seed* kecil (n = 3 untuk varian YOLO, n = 2 untuk RT-DETR-l), uji ini bersifat indikatif: ia dapat memastikan perbedaan yang konsisten, tetapi berdaya rendah untuk menolak kesetaraan—keterbatasan yang kami bahas secara eksplisit di Bagian VI.

## IV. Hasil

Tabel II merangkum performa keempat arsitektur pada *split* uji. YOLOv8s mencapai mAP@0.5 tertinggi sekaligus kecepatan inferensi terbaik. YOLO11s dan RT-DETR-l setara secara akurasi namun lebih lambat, sedangkan YOLOv8s+CBAM justru paling rendah pada mAP@0.5.

**TABEL II. Perbandingan performa pada split uji (rata-rata ± simpangan baku).**

| Model | n *seed* | mAP@0.5 | mAP@[.5:.95] | FPS |
|---|---|---|---|---|
| **YOLOv8s** | 3 | **0,9592 ± 0,0014** | 0,6862 ± 0,0046 | **~296** |
| YOLO11s | 3 | 0,9569 ± 0,0046 | **0,6898 ± 0,0030** | ~189 |
| RT-DETR-l | 2 | 0,9588 ± 0,0151 | 0,6764 ± 0,0205 | ~55 |
| YOLOv8s+CBAM | 3 | 0,9479 ± 0,0004 | 0,6810 ± 0,0045 | ~290 |

### A. Pengaruh mekanisme atensi

Sumbu utama studi—keberadaan dan jenis atensi—tidak menunjukkan keuntungan di bawah protokol pelatihan yang ditata untuk YOLO. Gambar 1 memvisualisasikan mAP@0.5 keempat model beserta simpangan baku antar-*seed*. YOLOv8s tanpa atensi memimpin.

**TABEL III. Hasil uji-t berpasangan untuk selisih mAP@0.5.**

| Perbandingan | Δ mAP@0.5 | t(df) | p | Cohen's *d* | 95% CI |
|---|---|---|---|---|---|
| YOLOv8s vs YOLO11s | +0,0023 | t(2) = 0,73 | 0,54 | 0,61 | [−0,011; +0,016] |
| YOLOv8s vs RT-DETR-l | +0,0004 | t(1) = 0,11 | 0,93 | — | [−0,140; +0,142] |
| YOLOv8s vs CBAM | +0,0113 | t(2) = 10,9 | 0,008 | 9,1 | [0,0068; 0,0157] |

Tabel III menyajikan hasil uji-t berpasangan untuk selisih mAP@0.5 antara baseline dan tiap varian. Uji-t berpasangan menegaskan bahwa baik YOLO11s (Δ = +0,0023; t(2) = 0,73; p = 0,54; Cohen's *d* = 0,61; 95% CI [−0,011; +0,016]) maupun RT-DETR-l (Δ = +0,0004; t(1) = 0,11; p = 0,93; 95% CI [−0,140; +0,142]) **tidak berbeda signifikan** dari baseline. Selang kepercayaan RT-DETR-l yang sangat lebar—akibat hanya dua *seed* dengan hasil berjauhan (0,9482 dan 0,9695)—menunjukkan estimasi yang tidak stabil; kesetaraannya karena itu lebih merupakan ketiadaan bukti perbedaan daripada bukti kesetaraan. Perlu dicatat bahwa Cohen's *d* untuk YOLO11s vs baseline (0,61) mengindikasikan efek berukuran sedang menurut konvensi, namun tidak mencapai signifikansi karena daya statistik yang sangat rendah dengan n = 3.

Berbeda dengan kedua model di atas, penambahan modul CBAM menurunkan mAP@0.5 sebesar 0,0113, dan penurunan ini **signifikan secara statistik** (uji-t berpasangan: t(2) = 10,9; p = 0,008; Cohen's *d* ≈ 9,1; 95% CI penurunan [0,0068; 0,0157]). Cohen's *d* yang sangat besar mengindikasikan efek yang substansial dan konsisten. Simpangan baku yang kecil pada kedua model (0,0014 untuk baseline, 0,0004 untuk varian CBAM) menegaskan bahwa penurunan ini stabil di seluruh inisialisasi yang diuji. Perlu ditekankan bahwa hasil ini berlaku di bawah protokol pelatihan yang ditata untuk YOLO; ia menunjukkan bahwa penyisipan CBAM merugikan *pada kondisi tersebut*, bukan bahwa atensi merugikan secara universal.

![Gambar 1. Pengaruh mekanisme atensi terhadap mAP@0.5 pada split uji. Batang galat menunjukkan simpangan baku antar-seed. Tidak ada varian ber-atensi yang melampaui YOLOv8s; CBAM justru terendah.](figures/fig1_attention_mAP50.png)

### B. Performa per-kelas

Pada baseline YOLOv8s, akurasi per-kelas menunjukkan pola yang konsisten dengan ukuran dan kekhasan objek (Gambar 2). Kelas *motorcyclist* paling mudah dideteksi (mAP@0.5 = 0,990) karena berukuran besar dan menonjol di bingkai. Kelas *license_plate* mengikuti (0,969) dengan bentuk persegi panjang yang khas. Kelas *helmet* paling menantang (0,923) karena ukurannya kecil, kemiripannya dengan objek lain di kepala pengendara seperti topi atau rambut, serta variasi warna dan sudut pandang. Pola ini menegaskan bahwa kelas *helmet*—yang justru paling relevan untuk tujuan aplikasi—adalah penentu utama ruang perbaikan, dan setiap upaya peningkatan sebaiknya difokuskan ke sana.

![Gambar 2. Performa per-kelas YOLOv8s pada split uji. Kelas helmet paling menantang karena ukurannya kecil dan ambigu secara visual.](figures/fig2_per_class.png)

### C. Trade-off kecepatan

Perbedaan kecepatan jauh lebih besar daripada perbedaan akurasi (Gambar 3). RT-DETR-l berjalan pada sekitar 55 FPS, kira-kira lima kali lebih lambat daripada YOLOv8s yang mencapai sekitar 296 FPS, dan menuntut waktu pelatihan sekitar 190 menit per *run* dibanding sekitar 26 menit untuk YOLO. YOLO11s berada di tengah dengan sekitar 189 FPS, sedangkan varian CBAM nyaris sama cepat dengan baseline karena modul yang ditambahkan ringan. Ketika akurasi keempat model praktis setara, biaya komputasi transformer tidak terbayar pada tugas ini. Gambar 3 menempatkan YOLOv8s di pojok paling menguntungkan: akurasi tertinggi sekaligus kecepatan tertinggi.

FPS diukur pada *run* tunggal saat GPU senggang (tidak ada proses training atau inferensi lain yang berjalan), menggunakan benchmark bawaan Ultralytics pada *split* test penuh. Pengukuran dilakukan pada konfigurasi perangkat keras yang sama (NVIDIA RTX 4090, batch size 1) untuk seluruh model. Angka FPS bersifat estimasi pada satu titik perangkat keras dan akan berbeda pada konfigurasi lain.

![Gambar 3. Trade-off akurasi (mAP@0.5) terhadap kecepatan (FPS). YOLOv8s menempati posisi paling menguntungkan, unggul pada kedua sumbu sekalignya.](figures/fig3_accuracy_speed.png)

### D. Analisis kualitatif

Gambar 4 menampilkan contoh keluaran deteksi YOLOv8s pada satu citra uji. Model mendeteksi pengendara, helm, dan plat nomor secara bersamaan dengan kotak pembatas yang rapat, menggambarkan kasus tipikal tempat ketiga kelas hadir dalam satu adegan. Inspeksi visual atas sampel uji menunjukkan bahwa kesalahan yang tersisa cenderung muncul pada helm berukuran sangat kecil atau terhalang sebagian, sejalan dengan temuan kuantitatif bahwa kelas *helmet* adalah yang paling sulit.

![Gambar 4. Contoh keluaran deteksi YOLOv8s pada citra uji: pengendara, helm, dan plat nomor terdeteksi bersamaan.](figures/fig4_detection_example.png)

## V. Pembahasan

Temuan sentral studi ini berlawanan dengan intuisi umum yang dibentuk oleh literatur deteksi helm ber-atensi (Bagian II.D): pada dataset deteksi helm berskala kecil dan di bawah protokol pelatihan yang ditata untuk YOLO, tidak ada mekanisme atensi yang melampaui CNN murni, dan modul CBAM bahkan merugikan secara signifikan. Enam penjelasan yang saling melengkapi dapat menerangkan pola ini.

Pertama, **dataset relatif kecil dan mudah**. Dengan 1.563 citra latih dan baseline yang sudah mencapai sekitar 0,96 mAP@0.5, ruang perbaikan sangat sempit. Ketika sebuah model sederhana sudah mendekati batas atas yang dapat dicapai pada data ini, hampir tidak ada celah bagi mekanisme yang lebih kompleks untuk menunjukkan keunggulan. Mekanisme atensi—terutama transformer yang lapar data—membutuhkan volume dan keragaman besar untuk mempelajari pola ketergantungan yang berguna [17], [18]. Pada rezim data kecil, keunggulan teoritisnya tidak terwujud dan model justru menanggung beban kapasitas yang tidak terpakai.

Kedua, **lapisan atensi yang diinisialisasi acak dapat mengganggu fitur praterlatih**. Pada varian CBAM, sebagian besar bobot diwarisi dari COCO sementara modul atensi dimulai dari nol. Modul yang belum terlatih ini menyisipkan transformasi yang, pada awal pelatihan, mengganggu aliran fitur yang sudah matang dari *transfer learning*. Dengan data terbatas, tidak cukup sinyal gradien untuk memulihkan gangguan tersebut, apalagi melampaui baseline. Hasilnya adalah penurunan kecil namun konsisten yang teramati stabil di seluruh *seed*. Penjelasan ini sejalan dengan pengamatan umum bahwa modul tambahan paling bermanfaat ketika dilatih bersama dari awal pada data besar, bukan ditempelkan pada jaringan praterlatih dengan data terbatas.

Ketiga, **kompleksitas tidak gratis**. Atensi menambah parameter dan komputasi. Ketika tidak ada keuntungan akurasi, tambahan ini murni menjadi biaya. Konsekuensinya paling nyata pada RT-DETR-l yang lima kali lebih lambat tanpa imbalan mAP, tetapi juga berlaku konseptual pada CBAM yang menambah jalur komputasi pada setiap skala deteksi. Dalam pengaturan waktu-nyata tempat anggaran komputasi terbatas, biaya ini berarti penurunan jumlah bingkai yang dapat diproses per detik.

Keempat, **biaya kesalahan asimetris dalam konteks keselamatan**. Dalam aplikasi penegakan keselamatan, konsekuensi *false negative* (gagal mendeteksi pengendara tanpa helm) jauh lebih berat daripada *false positive* (menandai pengendara berhelm sebagai pelanggar). Pada baseline YOLOv8s, mAP@0.5 untuk kelas *helmet* adalah 0,923, yang berarti sekitar 7,7% helm tidak terdeteksi dengan benar. Dalam konteks penegakan otomatis, ini menyiratkan bahwa hampir satu dari tiga belas pelanggar lolos tanpa terdeteksi. Peningkatan recall untuk kelas *helmet*—bahkan dengan mempertaruhkan precision—mungkin lebih dihargai dalam aplikasi nyata daripada mAP secara keseluruhan. Perancangan fungsi rugi yang memperberat kelas *helmet* atau penggunaan metrik yang secara eksplisit memperhitungkan biaya asimetris (misalnya *weighted F1-score*) patut dieksplorasi untuk deployment yang bertujuan menegakkan keselamatan.

Kelima, **kesenjangan menuju deployment nyata**. Dataset yang digunakan bersifat relatif bersih: pencahayaan baik, oklusi minimal, dan kepadatan objek rendah. Pada kondisi CCTV jalan nyata—dengan tantangan oklusi berat oleh pengendara lain, pencahayaan malam, hujan, dan kepadatan lalu lintas tinggi—kapasitas atensi untuk memodelkan konteks global mungkin menjadi berguna. Raj dan Nair [24] mencatat bahwa variasi pencahayaan dan oklusi merupakan tantangan utama pada CCTV nyata, dan Hariyono dkk. [21] menekankan perlunya robustness terhadap kondisi tersebut dalam konteks Indonesia. Cheng dkk. [23] menegaskan bahwa objek kecil dalam bingkai padat membutuhkan strategi fitur yang lebih canggih. Di bawah kondisi yang lebih menantang, mekanisme atensi mungkin memberikan manfaat yang tidak teramati pada dataset yang relatif mudah dalam studi ini.

Keenam, **hipotesis tandingan untuk penurunan CBAM**. Penurunan mAP@0.5 akibat CBAM yang teramati dalam studi ini tidak serta-merta berarti bahwa CBAM tidak berguna untuk deteksi helm. Sejumlah faktor desain alternatif dapat menjelaskan penurunan tersebut: (a) penempatan modul yang mungkin suboptimal—kami hanya menyisipkan CBAM pada kepala deteksi (P3, P4, P5), bukan pada *backbone*, padahal penempatan pada *backbone* seperti yang dilakukan pada sejumlah studi [19], [20] mungkin memberikan hasil berbeda; (b) lapisan *fresh* yang mengganggu bobot praterlatih dari COCO tanpa cukup data untuk memulihkannya; (c) jumlah epoch pelatihan yang mungkin tidak cukup bagi modul baru untuk berkonvergensi; (d) *learning rate* yang tidak dituning secara terpisah untuk parameter modul atensi. Ablasi sistematis—dengan variasi penempatan, pelatihan dari awal (*from scratch*), epoch lebih banyak, dan *learning rate* terpisah—diperlukan untuk mengisolasi penyebab pasti dan menjadi agenda kerja lanjut.

Temuan ini dapat ditempatkan dalam konteks literatur yang lebih luas. Keunggulan arsitektur transformer dan modul atensi paling konsisten dilaporkan pada *benchmark* berskala besar dengan keragaman tinggi [17], [18]. Pada tugas khusus dengan data terbatas, sejumlah studi menemukan bahwa CNN yang ditata baik tetap kompetitif atau unggul. Hasil kami menambah bukti pada arah tersebut khusus untuk domain deteksi helm, dan menggarisbawahi bahwa pemilihan arsitektur sebaiknya dipandu oleh karakteristik data dan anggaran komputasi, bukan oleh tren arsitektur semata.

Penting menegaskan batas klaim ini agar tidak terjadi *overgeneralisasi*. Studi ini **tidak** menyimpulkan bahwa atensi tidak berguna secara umum. Yang kami tunjukkan lebih sempit: di bawah satu protokol pelatihan yang ditata untuk YOLO, pada satu dataset kecil yang relatif mudah, dan dengan baseline yang sudah mendekati langit-langit, penambahan atensi tidak meningkatkan akurasi dan—pada kasus CBAM—justru menurunkannya secara signifikan. Kesetaraan YOLO11s dan RT-DETR-l adalah ketiadaan bukti perbedaan, bukan bukti kesetaraan; efek langit-langit menekan daya statistik sehingga manfaat kecil dari atensi, seandainya ada, sulit terdeteksi. Hipotesis tandingan yang masuk akal—bahwa model ber-atensi belum dilatih pada resep optimalnya—tidak dapat kami singkirkan dan justru menjadi agenda kerja lanjut. Dengan kata lain, temuan ini paling tepat dibaca sebagai peringatan praktis terhadap asumsi "menambahkan atensi pasti membantu", bukan sebagai vonis terhadap mekanisme atensi secara umum.

Implikasi praktisnya, dalam ruang lingkup tersebut, jelas: untuk deteksi helm waktu-nyata pada data serupa, arsitektur CNN sederhana seperti YOLOv8s adalah pilihan paling rasional—akurasi tertinggi, paling ringan, dan tercepat. Temuan ini juga menjadi pengingat metodologis bahwa klaim keunggulan arsitektur harus diuji dalam kondisi terkendali dan berulang sebelum diadopsi. Tanpa pengulangan antar-*seed*, selisih kecil seperti yang teramati di sini mudah disalahartikan sebagai keunggulan nyata padahal berada dalam rentang variasi acak.

## VI. Ancaman terhadap Validitas dan Keterbatasan

Studi ini memiliki sejumlah keterbatasan yang membatasi generalisasi temuan. Setiap keterbatasan diuraikan secara eksplisit di bawah.

**(1) Evaluasi terbatas pada satu dataset yang relatif bersih.** Generalisasi ke kondisi nyata—rekaman CCTV jalan dengan oklusi, gerak cepat, pencahayaan buruk, dan kepadatan tinggi—belum diuji. Boleh jadi pada data yang lebih menantang, kemampuan atensi memodelkan konteks justru menjadi berguna.

**(2) RT-DETR-l hanya dijalankan pada dua *seed*.** Karena biaya komputasi yang jauh lebih tinggi (~190 menit per *run*), RT-DETR-l hanya dievaluasi pada dua *seed*. Konsekuensinya, estimasi variansinya kurang presisi (simpangan baku 0,0151 vs 0,0014–0,0046 untuk varian YOLO) dan selang kepercayaan 95% sangat lebar ([−0,140; +0,142]). Kesimpulan kesetaraannya tidak dapat dipertimbangkan dengan keyakinan yang sama seperti perbandingan antar-varian YOLO yang menggunakan tiga *seed*. Klaim kesetaraan RT-DETR-l perlu diturunkan tingkat kepastiannya.

**(3) Perbandingan bukan *size-matched*.** RT-DETR-l (~32 juta parameter) jauh lebih besar daripada varian YOLO-s (~9–11 juta). Perbandingan ini mencerminkan pilihan model standar yang dihadapi praktisi, namun mencampur faktor paradigma arsitektur dan kapasitas model. Perbandingan yang menyamakan jumlah parameter—misalnya dengan mengevaluasi YOLOv8m (~25 juta parameter) sebagai kontrol *size-matched* untuk RT-DETR-l—dapat memberikan gambaran yang lebih murni tentang efek paradigma (transformer vs CNN) terlepas dari kapasitas.

**(4) Konfounding hyperparameter.** Hyperparameter diseragamkan dan tidak di-*tuning* khusus per-arsitektur. Protokol ini menguntungkan YOLO, yang merupakan arsitektur yang lebih matang dan lebih sensitif terhadap pengaturan Ultralytics bawaan. Transformer dan model ber-atensi mungkin menuntut resep pelatihan berbeda—misalnya *warm-up* lebih panjang, *learning rate* yang berbeda, atau jadwal *decay* khusus—untuk mencapai potensi puncaknya. Akibatnya, hasil model ber-atensi mungkin merupakan batas bawah, bukan potensi sebenarnya.

**(5) Efek langit-langit menekan daya statistik.** Dengan baseline sudah ~0,96, ruang perbaikan sangat sempit, dan jumlah *seed* kecil (n = 2–3), daya statistik uji-t sangat rendah. Untuk gambaran konkret: dengan hanya 100 citra uji, perbedaan satu hingga dua anotasi yang salah sudah dapat menggeser mAP@0.5 sekitar 0,01. Ini berarti selisih sebesar 0,002–0,004 yang teramati antara YOLOv8s dan YOLO11s berada dalam rentang variasi yang dapat disebabkan oleh perbedaan pada satu hingga dua citra. Konsekuensinya, klaim kesetaraan (YOLO11s, RT-DETR-l) harus dibaca sebagai ketiadaan bukti perbedaan, bukan bukti kesetaraan; menyimpulkan "atensi tidak membantu" secara mutlak akan menjadi lompatan dari *absence of evidence* ke *evidence of absence*.

**(6) Pengukuran FPS pada konfigurasi tunggal.** Angka kecepatan diambil dari *run* tunggal untuk memitigasi interferensi antar-*run*, namun tetap merupakan estimasi pada satu konfigurasi perangkat keras (NVIDIA RTX 4090, batch size 1). Hasil FPS akan berbeda pada perangkat lain dan tidak mencerminkan latensi end-to-end termasuk pra-pemrosesan dan pasca-pemrosesan.

**(7) Penurunan CBAM mungkin artefak desain eksperimen.** Penurunan akurasi akibat CBAM tidak dapat serta-merta diatribusikan ke inefektivitas modul atensi itu sendiri. Sejumlah faktor desain alternatif dapat menjelaskan penurunan tersebut: (a) penempatan suboptimal—CBAM hanya disisipkan pada kepala deteksi, bukan pada *backbone*; (b) lapisan *fresh* mengganggu bobot praterlatih dari COCO; (c) jumlah epoch yang tidak cukup bagi modul baru untuk berkonvergensi; (d) *learning rate* yang tidak dituning secara terpisah untuk parameter atensi. Ablasi sistematis dengan variasi penempatan, pelatihan dari awal, epoch lebih banyak, dan *learning rate* terpisah diperlukan untuk mengisolasi penyebab pasti.

**(8) Dataset tunggal tanpa variasi kesulitan.** Penggunaan satu dataset dengan kondisi relatif mudah (pencahayaan baik, oklusi minimal) membatasi kemampuan untuk menguji apakah atensi memberikan manfaat pada kondisi yang lebih menantang. Generalisasi ke CCTV nyata dengan oklusi berat, pencahayaan malam, hujan, dan kepadatan lalu lintas tinggi belum terverifikasi.

**(9) Jumlah *seed* terbatas secara keseluruhan.** Meskipun tiga *seed* cukup untuk mendeteksi efek besar seperti penurunan CBAM (Cohen's *d* ≈ 9,1), daya statistik untuk mendeteksi efek kecil sangat terbatas. Untuk memastikan kesetaraan dengan tingkat keyakinan yang memadai, dibutuhkan jumlah *seed* yang lebih besar atau metode inferensi yang lebih canggih (misalnya Bayesian equivalence testing).

**(10) Tidak ada evaluasi pada arsitektur *size-matched* ber-atensi.** Perbandingan YOLOv8m (~25 juta parameter) atau varian YOLO berukuran menengah lainnya sebagai kontrol *size-matched* untuk RT-DETR-l belum dilakukan. Tanpa perbandingan ini, tidak dapat dipastikan apakah hasil RT-DETR-l yang setara mencerminkan ketidakefektifan paradigma transformer atau sekadar mencerminkan ketidakcocokan skala model.

## VII. Kesimpulan dan Saran

Kami membandingkan empat arsitektur deteksi objek yang menempati spektrum penggunaan atensi untuk deteksi penggunaan helm pengendara sepeda motor, di bawah protokol pelatihan dan evaluasi yang identik dan berulang. Di bawah protokol yang ditata untuk YOLO dan pada satu dataset kecil yang relatif mudah, tidak ada mekanisme atensi—parsial, transformer penuh, maupun modul CBAM—yang melampaui YOLOv8s CNN murni. YOLOv8s memberi akurasi tertinggi (mAP@0.5 = 0,9592 ± 0,0014) sekaligus kecepatan terbaik (~296 FPS), sementara penambahan CBAM menurunkan mAP@0.5 sebesar 0,0113 secara signifikan (uji-t berpasangan, t(2) = 10,9, p = 0,008, Cohen's *d* ≈ 9,1); kesetaraan YOLO11s (p = 0,54) dan RT-DETR-l (p = 0,93) ditafsirkan sebagai ketiadaan bukti perbedaan, bukan bukti kesetaraan, mengingat efek langit-langit dan jumlah *seed* yang terbatas.

**Temuan ini berlaku di bawah kondisi spesifik yang diuji:** satu dataset kecil yang relatif mudah, protokol pelatihan yang ditata untuk YOLO, dan resolusi 1280 piksel. Generalisasi ke kondisi lain—dataset yang lebih besar dan menantang, hyperparameter yang dituning per-arsitektur, atau arsitektur dengan kapasitas setara—memerlukan validasi lanjutan. Dalam ruang lingkup ini, temuan ini merupakan **peringatan praktis terhadap asumsi bahwa menambahkan atensi selalu membantu pada tugas deteksi helm berskala kecil, bukan vonis umum atas mekanisme atensi.**

Saran kerja lanjut mengikuti langsung dari keterbatasan di atas. Pertama, menguji generalisasi pada dataset yang lebih besar, beragam, dan menantang, termasuk rekaman CCTV nyata dengan kondisi sulit (oklusi, pencahayaan malam, hujan, kepadatan tinggi). Kedua, mengevaluasi atensi dalam rezim data besar tempat keunggulannya berpeluang muncul, serta dengan pelatihan dari awal alih-alih hanya menempelkan modul pada jaringan praterlatih. Ketiga, menambahkan perbandingan *size-matched*—khususnya YOLOv8m (~25 juta parameter) sebagai kontrol kapasitas untuk RT-DETR-l—serta *fine-tuning* hyperparameter khusus per-arsitektur agar setiap model diuji pada kondisi terbaiknya. Keempat, melakukan ablas CBAM yang sistematis: variasi penempatan (*backbone* vs *neck* vs kepala deteksi), pelatihan dari awal (*from scratch*), epoch lebih banyak, dan *learning rate* terpisah untuk parameter modul atensi. Kelima, mengeksplorasi pendekatan *two-stage*—deteksi motor lalu klasifikasi helm—dan integrasi deteksi plat nomor untuk skenario penegakan, dengan fungsi rugi yang memperberat kelas *helmet* guna menangani biaya kesalahan asimetris. Keenam, meningkatkan jumlah *seed* (minimal n = 5) atau menggunakan metode inferensi Bayesian untuk menguatkan klaim kesetaraan atau perbedaan.

---

## Pernyataan

**Ketersediaan Data (Data Availability).** Dataset bersifat publik melalui Roboflow Universe ("NCKH 2023 / Helmet Detection Project" v19, lisensi MIT). Skrip pelatihan, konfigurasi, dan notebook eksperimen tersedia dalam repositori riset penulis.

**Pernyataan Etika (Ethics).** Studi menggunakan dataset publik berlisensi terbuka tanpa data pribadi yang dapat diidentifikasi secara langsung di luar konteks lalu lintas publik; tidak melibatkan subjek manusia maupun eksperimen yang memerlukan persetujuan etik.

**Kontribusi Penulis (CRediT).** Konseptualisasi, metodologi, perangkat lunak, analisis, dan penulisan naskah: penulis. (Sesuaikan dengan daftar penulis sebenarnya.)

**Konflik Kepentingan (Conflict of Interest).** Penulis menyatakan tidak ada konflik kepentingan.

**Pendanaan (Funding).** Tidak ada pendanaan khusus yang dilaporkan untuk studi ini.

**Pernyataan Penggunaan AI (AI Disclosure).** Penyusunan eksperimen, analisis, dan penulisan naskah ini dibantu oleh alat berbasis AI (Claude Code beserta skill Academic Research Skills). Revisi naskah atas masukan *peer review* juga dibantu oleh Claude Code. Seluruh keputusan metodologis, verifikasi hasil numerik, dan tanggung jawab akhir atas isi berada pada penulis. Sumber rujukan diverifikasi keberadaannya secara independen.

---

## Referensi

[1] F. W. Siebert and H. Lin, "Detecting motorcycle helmet use with deep learning," *Accident Analysis & Prevention*, vol. 134, p. 105319, 2020.

[2] H. Lin, J. D. Deng, D. Albers, and F. W. Siebert, "Helmet use detection of tracked motorcycles using CNN-based multi-task learning," *IEEE Access*, vol. 8, pp. 162073–162084, 2020.

[3] M. Naphade et al., "The 7th AI City Challenge," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition Workshops (CVPRW)*, 2023.

[4] W. Jia et al., "Real-time automatic helmet detection of motorcyclists in urban traffic using improved YOLOv5 detector," *IET Image Processing*, vol. 15, no. 14, pp. 3623–3637, 2021.

[5] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, "You only look once: Unified, real-time object detection," in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 779–788.

[6] G. Jocher, A. Chaurasia, and J. Qiu, "Ultralytics YOLO," 2023. [Online]. Available: https://github.com/ultralytics/ultralytics. Accessed: Jun. 13, 2026.

[7] Y. Zhao et al., "DETRs beat YOLOs on real-time object detection," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2024.

[8] N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, and S. Zagoruyko, "End-to-end object detection with transformers," in *Proc. European Conf. Computer Vision (ECCV)*, 2020, pp. 213–229.

[9] S. Woo, J. Park, J.-Y. Lee, and I. S. Kweon, "CBAM: Convolutional block attention module," in *Proc. European Conf. Computer Vision (ECCV)*, 2018, pp. 3–19.

[10] A. Vaswani et al., "Attention is all you need," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2017, pp. 5998–6008.

[11] A. Dosovitskiy et al., "An image is worth 16x16 words: Transformers for image recognition at scale," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2021.

[12] J. Hu, L. Shen, and G. Sun, "Squeeze-and-excitation networks," in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, 2018, pp. 7132–7141.

[13] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, "Focal loss for dense object detection," in *Proc. IEEE Int. Conf. Computer Vision (ICCV)*, 2017, pp. 2980–2988.

[14] T.-Y. Lin et al., "Microsoft COCO: Common objects in context," in *Proc. European Conf. Computer Vision (ECCV)*, 2014, pp. 740–755.

[15] Z. Liu et al., "Swin Transformer: Hierarchical vision transformer using shifted windows," in *Proc. IEEE/CVF Int. Conf. Computer Vision (ICCV)*, 2021, pp. 10012–10022.

[16] Y. Li, H. Mao, R. Girshick, and K. He, "Exploring plain vision transformer backbones for object detection," in *Proc. European Conf. Computer Vision (ECCV)*, 2022, pp. 280–296.

[17] M.-H. Guo, T.-X. Xu, J.-J. Jiang, N. Liu, Z.-N. Liu, P.-T. Jiang, T.-J. Mu, and S.-M. Hu, "Attention mechanisms in computer vision: A survey," *Computational Visual Media*, vol. 8, no. 3, pp. 331–369, 2022.

[18] Z. Niu, G. Zhong, and H. Yu, "A review on the attention mechanism of deep learning," *Neurocomputing*, vol. 452, pp. 48–62, 2021.

[19] H. Zhang, Q. Luo, and C. Yin, "YOLOv8 safety helmet detection algorithm based on attention mechanism and feature fusion," *J. Phys.: Conf. Ser.*, vol. 3135, no. 1, p. 012025, 2025.

[20] J. Li, X. Zhang, and Z. Liu, "Safety helmet detection based on improved YOLOv5s with attention mechanism," *Sensors*, vol. 23, no. 14, p. 6500, 2023.

[21] M. H. D. Hariyono, A. N. Ihsan, and D. S. Kusumo, "Streamlit application for helmet detection based on YOLOS: Case study Indonesia," in *Proc. Data Science and Its Applications Conf. (DASA)*, 2024.

[22] W. Jia et al., "DAAM-YOLOv5: Helmet detection combined with attention mechanism," *Electronics*, vol. 12, no. 9, p. 2094, 2023.

[23] G. Cheng, X. Yuan, X. Yao, K. Yan, Q. Zeng, and J. Han, "Towards large-scale small object detection: Survey and benchmarks," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 46, no. 1, pp. 521–539, 2024.

[24] R. D. Raj and M. S. Nair, "A YOLO-based approach to detecting helmetless riders through CCTV," *SITECH: J. Inf. Technol.*, 2024.
