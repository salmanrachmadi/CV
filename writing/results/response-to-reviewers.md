# Respons terhadap Reviewer

**Naskah:** Perbandingan Arsitektur Deteksi Objek Berbasis dan Tanpa Mekanisme Atensi untuk Deteksi Penggunaan Helm Pengendara Sepeda Motor

**Tanggal revisi:** 2026-06-13

---

Kepada Editor dan Reviewer yang terhormat,

Terima kasih atas masukan yang mendalam dan konstruktif terhadap naskah kami. Kami telah merevisi naskah secara menyeluruh berdasarkan seluruh komentar dari kelima reviewer. Berikut adalah respons titik-per-titik terhadap setiap isu yang disampaikan.

Perubahan dalam naskah revisi ditandai secara implisit melalui penguatan argumen, penambahan referensi, dan perluasan bagian Limitasi.

---

## Respons terhadap Reviewer EIC (Editor-in-Chief)

### EIC-1: Klaim "atensi tidak membantu" terlalu umum untuk satu dataset

**Tipe:** MAJOR · **Status:** RESOLVED

**Respons:** Kami telah mempersempit seluruh klaim sentral di seluruh naskah. Abstrak (BA dan EN) sekarang secara eksplisit menyatakan bahwa temuan "berlaku di bawah kondisi spesifik yang diuji: satu dataset kecil yang relatif mudah, protokol pelatihan yang ditata untuk YOLO, dan resolusi 1280 piksel." Bagian Kesimpulan (§VII) menambahkan pernyataan serupa dan menyebut temuan ini sebagai "peringatan praktis terhadap asumsi bahwa menambahkan atensi selalu membantu pada tugas deteksi helm berskala kecil, bukan vonis umum atas mekanisme atensi." Bagian Pembahasan (§V) juga diperkuat dengan kalimat: "Studi ini **tidak** menyimpulkan bahwa atensi tidak berguna secara umum."

> Lihat: Abstrak (paragraf terakhir), §V (paragraf ke-7), §VII (paragraf ke-2).

### EIC-2: Kontribusi inkremental

**Tipe:** MAJOR · **Status:** RESOLVED

**Respons:** Kami telah memperkuat framing kontribusi (§I, kontribusi 3) sebagai "rebuttal terstruktur terhadap asumsi bahwa menambahkan atensi selalu membantu." Pertanyaan riset telah diperkuat menjadi "apakah peningkatan yang dilaporkan oleh studi-studi ber-atensi pada deteksi helm dapat direplikasi dalam perbandingan terkendali?" — memosisikan paper sebagai pengujian generalisabilitas klaim positif yang ada di literatur, bukan sekadar perbandingan rutin. Subbagian baru §II.D ("Atensi pada deteksi helm") secara eksplisit meninjau studi yang melaporkan manfaat atensi [19], [20], [22] dan mengidentifikasi celah metodologis mereka, membangun kasus untuk studi kami sebagai kontribusi yang mengisi celah tersebut.

> Lihat: §I (kontribusi 3, pertanyaan riset), §II.D (subbagian baru).

### EIC-3: Kebaruan metodologis terbatas

**Tipe:** MAJOR · **Status:** DELIBERATE_LIMITATION

**Respons:** Kami mengakui bahwa kebaruan metodologis memang terbatas — kekuatan paper terletak pada kebaruan temuan (hasil negatif yang terkendali) dan kebaruan framing (rebuttal terstruktur), bukan pada kebaruan metode. Hal ini telah kami refleksikan dalam pembahasan tentang posisi paper sebagai "peringatan praktis" (§V, §VII). Pemilihan venue yang tepat (workshop/konferensi terapan) sejalan dengan karakter ini.

---

## Respons terhadap Reviewer 1 — Metodologi

### R1-1: Konfounding hyperparameter (MAJOR)

**Tipe:** MAJOR · **Status:** DELIBERATE_LIMITATION

**Respons:** Kami mengakui bahwa protokol hyperparameter seragam menguntungkan YOLO dan mungkin belum memaksimalkan potensi model ber-atensi. Keterbatasan ini sekarang diakui secara eksplisit sebagai **Limitasi (4)** di §VI: "Hyperparameter diseragamkan dan tidak di-*tuning* khusus per-arsitektur. Protokol ini menguntungkan YOLO... Transformer dan model ber-atensi mungkin menuntut resep pelatihan berbeda... hasil model ber-atensi mungkin merupakan batas bawah, bukan potensi sebenarnya." Hipotesis tandingan ini juga disebut di §V: "Hipotesis tandingan yang masuk akal—bahwa model ber-atensi belum dilatih pada resep optimalnya—tidak dapat kami singkirkan dan justru menjadi agenda kerja lanjut." Fine-tuning per-arsitektur ditambahkan di §VII Future Work.

> Lihat: §VI(4), §V (paragraf ke-7), §VII (saran ke-3).

### R1-2: Uji signifikansi lemah (MAJOR)

**Tipe:** MAJOR · **Status:** RESOLVED

**Respons:** Kami telah mengganti heuristik dengan uji statistik formal. §III.D sekarang mendeskripsikan protokol eksplisit: "uji-t berpasangan (*paired t-test*) pada selisih mAP@0.5 tiap pasangan *seed*, dengan taraf nyata α = 0,05." **Tabel III** baru ditambahkan di §IV.A yang merangkum seluruh perbandingan berpasangan dengan nilai t(df), p, Cohen's *d*, dan selang kepercayaan 95%. Cohen's *d* untuk YOLO11s vs baseline (0,61) didiskusikan sebagai efek berukuran sedang yang tidak mencapai signifikansi karena daya rendah.

> Lihat: §III.D (paragraf baru), §IV.A (Tabel III dan pembahasan effect size).

### R1-3: RT-DETR n=2 (MAJOR)

**Tipe:** MAJOR · **Status:** DELIBERATE_LIMITATION

**Respons:** Keterbatasan n=2 untuk RT-DETR-l sekarang diperluas sebagai **Limitasi (2)** di §VI dengan pembahasan konsekuensi konkret: "estimasi variansinya kurang presisi (simpangan baku 0,0151 vs 0,0014–0,0046) dan selang kepercayaan 95% sangat lebar ([−0,140; +0,142])." Klaim kesetaraan RT-DETR-l secara konsisten dinyatakan sebagai "ketiadaan bukti perbedaan" di seluruh naskah. Penyelesaian seed ke-3 ditambahkan di §VII Future Work.

> Lihat: §VI(2), §IV.A (Tabel III), §VII (saran ke-6).

### R1-4: Bukan size-matched (MAJOR)

**Tipe:** MAJOR · **Status:** DELIBERATE_LIMITATION

**Respons:** Ketidakcocokan kapasitas sekarang diakui di tiga tempat: (a) §III.B mencatat secara eksplisit bahwa "RT-DETR-l (~32 juta parameter) jauh lebih besar" dan menjelaskan alasan pemilihan model standar; (b) **Limitasi (3)** di §VI membahas implikasi dan merekomendasikan "YOLOv8m (~25 juta parameter) sebagai kontrol *size-matched*"; (c) **Limitasi (10)** di §VI menegaskan perlunya evaluasi arsitektur size-matched. Perbandingan YOLOv8m ditambahkan di §VII Future Work.

> Lihat: §III.B (paragraf ke-3), §VI(3), §VI(10), §VII (saran ke-3).

### R1-5: Efek langit-langit / daya statistik rendah (MAJOR)

**Tipe:** MAJOR → CRITICAL · **Status:** RESOLVED

**Respons:** Efek langit-langit sekarang didiskusikan secara mendalam di **Limitasi (5)** §VI dengan kuantifikasi konkret: "dengan hanya 100 citra uji, perbedaan satu hingga dua anotasi yang salah sudah dapat menggeser mAP@0.5 sekitar 0,01." Seluruh klaim kesetaraan secara konsisten dinyatakan sebagai "ketiadaan bukti perbedaan, bukan bukti kesetaraan." Pembahasan di §V menegaskan: "menyimpulkan 'atensi tidak membantu' secara mutlak akan menjadi lompatan dari *absence of evidence* ke *evidence of absence*." Limitasi (9) juga menambahkan bahwa tiga seed cukup untuk efek besar (CBAM) tapi tidak untuk efek kecil.

> Lihat: §VI(5), §VI(9), §V (paragraf ke-7), Abstrak.

### R1-6: CBAM penurunan mungkin artefak desain (MAJOR)

**Tipe:** MAJOR · **Status:** RESOLVED

**Respons:** Kami telah menambahkan penjelasan keenam di §V yang secara eksplisit membahas empat hipotesis tandingan untuk penurunan CBAM: (a) penempatan suboptimal pada neck bukan backbone; (b) lapisan fresh mengganggu bobot COCO; (c) epoch tidak cukup; (d) learning rate tidak dituning terpisah. **Limitasi (7)** di §VI mengulangi poin ini dan menyatakan bahwa "penurunan akurasi akibat CBAM tidak dapat serta-merta diatribusikan ke inefektivitas modul atensi itu sendiri." Ablasi CBAM sistematis (variasi penempatan, from-scratch, epoch lebih banyak, LR terpisah) ditambahkan di §VII Future Work (saran ke-4).

> Lihat: §V (penjelasan ke-6), §VI(7), §VII (saran ke-4).

### R1-7: FPS protokol tidak konsisten (MINOR)

**Tipe:** MINOR · **Status:** RESOLVED

**Respons:** Protokol FPS sekarang dideskripsikan secara eksplisit di dua tempat: (a) §III.D menyatakan bahwa "angka FPS yang dilaporkan diambil dari *run* tunggal saat GPU senggang... menggunakan benchmark bawaan Ultralytics pada *split* test penuh"; (b) §IV.C menambahkan paragraf terpisah: "FPS diukur pada *run* tunggal saat GPU senggang... Pengukuran dilakukan pada konfigurasi perangkat keras yang sama (NVIDIA RTX 4090, batch size 1)." **Limitasi (6)** di §VI mengakui bahwa angka FPS bersifat estimasi pada satu titik perangkat keras.

> Lihat: §III.D, §IV.C (paragraf baru), §VI(6).

---

## Respons terhadap Reviewer 2 — Domain

### R2-1: Cakupan literatur tipis (MAJOR)

**Tipe:** MAJOR · **Status:** RESOLVED

**Respons:** Kami telah menambahkan 8 referensi baru ([17]–[24]), meningkatkan total dari 16 menjadi 24 referensi. Penambahan mencakup:
- Survei atensi: Guo dkk. [17], Niu dkk. [18]
- Survei deteksi objek kecil: Cheng dkk. [23]
- Deteksi helm berbasis atensi: Zhang dkk. [19], Li dkk. [20], Jia dkk. [22]
- Konteks Indonesia: Hariyono dkk. [21], Raj dan Nair [24]
- Subbagian baru **§II.D "Atensi pada deteksi helm"** meninjau laporan positif dan mengidentifikasi celah metodologis mereka.
- §II.A diperluas dengan konteks Indonesia [21], [24].
- §II.C diperluas dengan survei atensi [17], [18] dan survei objek kecil [23].

> Lihat: §II.A, §II.C, §II.D (subbagian baru), Daftar Referensi.

### R2-2: Framing kontribusi (MAJOR)

**Tipe:** MAJOR · **Status:** RESOLVED

**Respons:** Kontribusi sekarang secara eksplisit diposisikan sebagai "rebuttal terstruktur" (§I, kontribusi 3). Pertanyaan riset telah diperkuat menjadi "apakah peningkatan yang dilaporkan oleh studi-studi ber-atensi... dapat direplikasi dalam perbandingan terkendali?" Subbagian §II.D membangun kasus bahwa studi-studi positif yang ada tidak memiliki kendali eksperimen yang memadai, memosisikan paper kami sebagai pengisi celah metodologis tersebut. §VII menyebut temuan ini sebagai "peringatan praktis terhadap asumsi bahwa menambahkan atensi selalu membantu pada tugas deteksi helm berskala kecil."

> Lihat: §I (pertanyaan riset, kontribusi 3), §II.D, §VII.

### R2-3: Posisi pada peta arsitektur (MINOR)

**Tipe:** MINOR · **Status:** RESOLVED

**Respons:** §II.C sekarang merujuk secara eksplisit pada Swin Transformer [15] dan ViTDet [16] sebagai titik pada spektrum atensi yang tidak terwakili dalam eksperimen kami (backbone ViT murni dan atensi berjendela hierarkis). §II.D meninjau studi yang menggunakan atensi pada deteksi helm. Limitasi bahwa empat titik tidak mencakup seluruh spektrum telah disebutkan di §VI dan §VII Future Work.

> Lihat: §II.C (paragraf ke-2), §II.D.

---

## Respons terhadap Reviewer 3 — Perspektif / Dampak

### R3-1: Kesenjangan ke deployment nyata (MAJOR)

**Tipe:** MAJOR · **Status:** RESOLVED

**Respons:** Penjelasan kelima di §V ("Kesenjangan menuju deployment nyata") secara eksplisit membahas bahwa pada kondisi CCTV jalan nyata (oklusi, malam, hujan, kepadatan), atensi mungkin memberikan manfaat yang tidak teramati pada dataset bersih kami. Kami merujuk Raj dan Nair [24] tentang tantangan CCTV nyata, Hariyono dkk. [21] tentang konteks Indonesia, dan Cheng dkk. [23] tentang objek kecil dalam bingkai padat. **Limitasi (8)** di §VI juga mengakui ketiadaan validasi lintas-dataset. §VII Future Work (saran ke-1) secara eksplisit menyebut pengujian pada CCTV nyata.

> Lihat: §V (penjelasan ke-5), §VI(8), §VII (saran ke-1).

### R3-2: "So what" perlu diperluas (MINOR)

**Tipe:** MINOR · **Status:** RESOLVED

**Respons:** Implikasi telah diperluas di §V melalui penjelasan keempat tentang biaya kesalahan asimetris (~7,7% helm terlewat), penjelasan kelima tentang deployment nyata, dan penjelasan keenam tentang hipotesis tandingan CBAM. §VII Future Work (saran ke-5) sekarang menyebut integrasi deteksi plat nomor, pendekatan two-stage, dan fungsi rugi yang memperberat kelas *helmet*.

> Lihat: §V (penjelasan ke-4 dan ke-5), §VII (saran ke-5).

### R3-3: Kelas minoritas & keselamatan (MINOR)

**Tipe:** MINOR · **Status:** RESOLVED

**Respons:** Penjelasan keempat di §V ("Biaya kesalahan asimetris dalam konteks keselamatan") secara eksplisit membahas bahwa *false negative* pada deteksi "tanpa helm" lebih merugikan, dan mengusulkan fungsi rugi berbobot dan metrik seperti *weighted F1-score* untuk deployment yang bertujuan menegakkan keselamatan.

> Lihat: §V (penjelasan ke-4).

---

## Respons terhadap Devil's Advocate

### DA-CRITICAL-1: Overgeneralisasi klaim inti

**Tipe:** CRITICAL · **Status:** RESOLVED

**Respons:** Seluruh klaim telah dipersempit dengan qualifier scope yang konsisten. Lihat respons EIC-1 di atas untuk detail lengkap perubahan di Abstrak, §V, dan §VII. Judul dipertahankan karena bersifat deskriptif ("Perbandingan... Berbasis dan Tanpa Mekanisme Atensi"), bukan interpretatif.

### DA-CRITICAL-2: Efek langit-langit melemahkan seluruh inferensi

**Tipe:** CRITICAL · **Status:** RESOLVED

**Respons:** Lihat respons R1-5 di atas. Efek langit-langit kini diakui secara kuantitatif (§VI(5): 100 citra uji → 1-2 anotasi menggeser mAP ~0,01) dan seluruh klaim dinyatakan sebagai "ketiadaan bukti" bukan "bukti ketiadaan."

### DA-MAJOR-1: Atribusi kausal "atensi" pada perbandingan yang tak terisolasi

**Tipe:** MAJOR · **Status:** DELIBERATE_LIMITATION

**Respons:** Kami mengakui bahwa RT-DETR ≠ "YOLO + atensi" dan perbandingan mencampur faktor paradigma dan kapasitas. §III.B (paragraf ke-3) menyatakan ini secara eksplisit. §VI(3) dan §VI(10) membahas implikasi. §VII merekomendasikan YOLOv8m sebagai kontrol size-matched. Frasa "perbandingan terkendali" di kontribusi 1 sekarang diikuti penjelasan bahwa "selisih performa dapat diatribusikan ke faktor arsitektur, khususnya keberadaan dan jenis atensi" — dengan catatan di §III.B bahwa kapasitas juga berbeda.

### DA-MAJOR-2: Inkonsistensi protokol pengukuran akurasi vs FPS

**Tipe:** MAJOR · **Status:** RESOLVED

**Respons:** Lihat respons R1-7 di atas. Protokol FPS sekarang dideskripsikan secara eksplisit dan konsisten di §III.D, §IV.C, dan §VI(6).

### DA-MINOR: Potensi cherry-pick naratif FPS

**Tipe:** MINOR · **Status:** RESOLVED

**Respons:** §IV.C sekarang menyatakan bahwa "Angka FPS bersifat estimasi pada satu titik perangkat keras dan akan berbeda pada konfigurasi lain." §VI(6) mengakui bahwa FPS diukur pada satu konfigurasi dan tidak mencerminkan latensi end-to-end. Kami tidak memilih angka "paling menguntungkan" — angka berasal dari satu-satunya run tunggal yang tersedia saat GPU senggang.

---

## Ringkasan Perubahan

| Metrik | Nilai |
|--------|-------|
| Total komentar yang ditangani | 17 |
| Resolved | 12 |
| Deliberate Limitation | 5 |
| Reviewer Disagree | 0 |
| Referensi baru ditambahkan | 8 (dari 16 → 24) |
| Subbagian baru | 1 (§II.D "Atensi pada deteksi helm") |
| Tabel baru | 1 (Tabel III: Hasil uji-t berpasangan) |
| Limitasi ditambahkan | 4 (dari 6 → 10) |
| Penjelasan di §V ditambahkan | 3 (dari 3 → 6) |
| Perubahan estimasi word count | +25% |

Kami percaya bahwa revisi ini telah secara substansial memperkuat naskah, khususnya melalui pensempitan klaim, penguatan metodologi statistik, perluasan literatur, dan pengakuan eksplisit atas seluruh keterbatasan yang disampaikan oleh reviewer.

Hormat kami,

Penulis
