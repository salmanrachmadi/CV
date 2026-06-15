# Laporan Peer-Review (Simulasi Panel 5 Reviewer)

**Naskah:** Perbandingan Arsitektur Deteksi Objek Berbasis dan Tanpa Mekanisme Atensi untuk Deteksi Penggunaan Helm Pengendara Sepeda Motor
**Bidang:** Computer Vision / Deep Learning · **Target:** Conference/Journal IEEE
**Tanggal review:** 2026-06-13

> Catatan: laporan ini bersifat *read-only* terhadap naskah. Skor pada skala 0–10. Bobot dimensi: Orisinalitas 20%, Rigor Metodologis 25%, Kecukupan Bukti 25%, Koherensi Argumen 15%, Kualitas Tulisan 15%.

---

## Phase 0 — Analisis Bidang & Konfigurasi Reviewer

- **Disiplin utama:** Computer vision (deteksi objek), aplikasi keselamatan transportasi.
- **Paradigma:** Empiris-eksperimental (perbandingan arsitektur terkendali).
- **Kematangan:** Draf solid, terstruktur baik; klaim sentral berisiko *overclaim*.
- **Tier target realistis:** Workshop/konferensi terapan atau jurnal terapan tingkat menengah.

**Lima reviewer:**
1. **EIC** — editor jurnal CV terapan; menilai kesesuaian, orisinalitas, signifikansi.
2. **R1 Metodologi** — ahli evaluasi model deteksi & statistik eksperimen.
3. **R2 Domain** — peneliti deteksi helm / ITS & atensi pada deteksi.
4. **R3 Perspektif** — praktisi deployment ITS/edge & dampak nyata.
5. **Devil's Advocate** — penantang klaim inti & sesat-pikir.

---

## Phase 1 — Lima Laporan Review Independen

### Reviewer EIC (Editor-in-Chief)

**Ringkasan.** Naskah menangani masalah relevan (kepatuhan helm) dengan desain perbandingan yang rapi dan—menyegarkan—melaporkan *hasil negatif*. Penulisan jelas, reproducibility ditekankan, figur memadai. Namun signifikansi dibatasi oleh validitas eksternal (satu dataset "mudah", efek langit-langit) dan klaim sentral yang melampaui bukti.

**Kekuatan:** (1) pertanyaan riset fokus; (2) protokol terkendali + multi-*seed*; (3) keterbukaan melaporkan hasil negatif & keterbatasan.
**Kelemahan utama:** (1) klaim "atensi tidak membantu" terlalu umum untuk satu dataset; (2) kontribusi inkremental; (3) kebaruan metodologis terbatas.

**Skor:** Orisinalitas 6 · Rigor 5 · Bukti 5 · Koherensi 6 · Tulisan 8.
**Rekomendasi:** **Major Revision.**

### Reviewer 1 — Metodologi

**Temuan kritis (penghambat publikasi):**
1. **Konfounding hyperparameter (MAJOR).** Semua arsitektur memakai hyperparameter identik (`optimizer=auto`, lr sama, augmentasi YOLO, 1280 px). Transformer seperti RT-DETR umumnya menuntut jadwal pelatihan, *learning rate*, dan augmentasi berbeda. Maka klaim "atensi tidak membantu" rancu dengan "atensi tidak di-*tune* dengan benar". Resep yang adil untuk satu keluarga belum tentu adil untuk yang lain.
2. **Uji signifikansi lemah (MAJOR).** Verdict "nyata vs noise" memakai heuristik |Δ| terhadap rata-rata simpangan baku, bukan uji statistik formal. Dengan n=3, estimasi std sendiri tidak stabil. Gunakan minimal uji-t berpasangan antar-*seed* atau selang kepercayaan; sebutkan *effect size*.
3. **RT-DETR n=2 (MAJOR).** Dua *seed* tidak cukup untuk std bermakna; std 0,0151 menandakan ketidakstabilan. Klaim kesetaraan RT-DETR rapuh. Lengkapi ke n=3 atau turunkan kekuatan klaim.
4. **Bukan *size-matched* (MAJOR).** RT-DETR-l (~32 jt) vs YOLO-s (~9–11 jt) mencampur faktor kapasitas dan arsitektur. Sertakan pembanding setara (mis. YOLOv8m ~25 jt) atau RT-DETR varian lebih kecil.
5. **Efek langit-langit / daya statistik rendah (MAJOR).** Baseline ~0,96 mAP menyisakan sedikit ruang; eksperimen mungkin tidak mampu mendeteksi manfaat atensi seandainya ada. Buat *headroom* (subset lebih kecil/sulit, atau metrik lebih ketat).
6. **CBAM: penurunan mungkin artefak desain (MAJOR).** Penyisipan CBAM kustom + lapisan fresh di atas bobot praterlatih bisa menyebabkan penurunan karena penempatan/inisialisasi, bukan karena "atensi merugikan". Perlu ablasi: penempatan berbeda, latih-dari-awal, atau pelatihan lebih panjang.
7. **FPS protokol tidak konsisten (MINOR).** Akurasi multi-*seed* tetapi FPS dari *run* tunggal "senggang" — protokol pengukuran berbeda. Laporkan distribusi latensi pada protokol seragam.

**Skor:** Orisinalitas 5 · Rigor 4 · Bukti 4 · Koherensi 6 · Tulisan 8.
**Rekomendasi:** **Major Revision.**

### Reviewer 2 — Domain

1. **Cakupan literatur tipis (MAJOR).** Hanya 14 rujukan. Hilang: survei atensi pada deteksi, Swin Transformer & ViTDet (padahal dibahas konseptual), karya deteksi-helm berbasis atensi terbaru, dan studi konteks Indonesia (disebut di materi riset tetapi tak disitir). Tambah perbandingan dengan temuan literatur tentang atensi pada data kecil.
2. **Framing kontribusi (MAJOR).** Hasil negatif bisa menjadi kontribusi kuat, tetapi harus diposisikan eksplisit terhadap klaim literatur yang ia tantang. Saat ini terasa sebagai "uji rutin" alih-alih bantahan terstruktur.
3. **Posisi pada peta arsitektur (MINOR).** "Spektrum atensi" menarik tetapi belum dirumuskan formal; perjelas mengapa empat titik ini representatif dan apa yang absen (mis. backbone ViT murni).

**Skor:** Orisinalitas 6 · Rigor 5 · Bukti 5 · Koherensi 6 · Tulisan 8.
**Rekomendasi:** **Major Revision.**

### Reviewer 3 — Perspektif / Dampak

1. **Kesenjangan ke deployment nyata (MAJOR).** Klaim praktis "pakai YOLOv8s" berpijak pada dataset bersih. Justru di CCTV nyata (oklusi, malam, kepadatan) atensi/konteks berpeluang berguna. Tanpa uji lintas-domain, rekomendasi praktis kurang berpijak.
2. **"So what" perlu diperluas (MINOR).** Implikasi saat ini sempit. Kaitkan ke anggaran *edge*, biaya energi, dan skenario penegakan (helm + plat) untuk memperbesar relevansi.
3. **Kelas minoritas & keselamatan (MINOR).** Diskusikan biaya kesalahan asimetris (gagal mendeteksi "tanpa helm" lebih merugikan) — relevan untuk aplikasi keselamatan.

**Skor:** Orisinalitas 6 · Rigor 5 · Bukti 5 · Koherensi 7 · Tulisan 8.
**Rekomendasi:** **Minor–Major Revision.**

### Devil's Advocate

**Counter-argument terkuat (≈250 kata).**
Klaim sentral—"mekanisme atensi tidak membantu deteksi helm"—adalah *overgeneralisasi* dari satu eksperimen yang terkonfounding. Hasil yang sama persis konsisten dengan hipotesis tandingan yang sama sekali berbeda: "kami tidak mengoptimalkan model ber-atensi dengan benar". Tiga fakta menopang tandingan ini. Pertama, hyperparameter dikunci ke resep yang menguntungkan YOLO; RT-DETR dan CBAM tidak pernah diberi kesempatan pada resepnya sendiri. Kedua, baseline sudah ~0,96 sehingga ada efek langit-langit—eksperimen nyaris tidak punya daya untuk mendeteksi perbaikan walau perbaikan itu ada. Ketiga, penurunan CBAM bersandar pada std yang sangat kecil dan heuristik, bukan uji nyata; dengan n=3, ini rapuh. Maka naskah melakukan lompatan dari "absence of evidence" ke "evidence of absence"—sebuah sesat-pikir klasik. Judul pun menjanjikan isolasi "berbasis vs tanpa atensi", padahal RT-DETR berbeda dalam banyak hal selain atensi, sehingga atribusi ke atensi tidak valid untuk titik itu.

**Daftar isu:**
- **CRITICAL** — Overgeneralisasi klaim inti (Abstrak, §V, §VII): ubah dari "atensi tidak membantu" menjadi klaim ber-ruang-lingkup ("di bawah hyperparameter ter-tuning-YOLO pada dataset kecil yang mudah, penambahan atensi tidak meningkatkan akurasi").
- **CRITICAL** — Efek langit-langit melemahkan seluruh inferensi (§IV–V): perlu *headroom* eksperimental atau pengakuan eksplisit atas daya statistik rendah.
- **MAJOR** — Atribusi kausal "atensi" pada perbandingan yang tak terisolasi (RT-DETR ≠ "YOLO + atensi").
- **MAJOR** — Inkonsistensi protokol pengukuran akurasi (multi-*seed*) vs FPS (*run* tunggal).
- **MINOR** — Potensi *cherry-pick* naratif: memilih angka FPS paling menguntungkan untuk interpretasi.

**Penjelasan alternatif yang diabaikan:** undertraining model ber-atensi; penempatan CBAM suboptimal; dataset terlalu mudah; kapasitas berlebih pada data kecil.

**Uji "So what":** lolos sebagian — temuan berguna sebagai peringatan praktis, tetapi hanya jika klaim dipersempit.

---

## Phase 2 — Keputusan Editorial & Peta Revisi

### Konsensus panel
- **Disepakati 5/5:** penulisan jelas; desain terkendali + multi-*seed* adalah kekuatan; klaim sentral **melampaui bukti**; validitas eksternal lemah (dataset tunggal).
- **Disepakati 4/5 (MAJOR):** konfounding hyperparameter; tidak *size-matched*; uji statistik lemah; literatur tipis.
- **Devil's Advocate CRITICAL:** overgeneralisasi + efek langit-langit → **menurut aturan panel, keputusan tidak boleh Accept.**

### Skor akhir (rata-rata tertimbang)

| Dimensi | Bobot | Skor | Kontribusi |
|---|---|---|---|
| Orisinalitas | 20% | 5,8 | 1,16 |
| Rigor Metodologis | 25% | 4,4 | 1,10 |
| Kecukupan Bukti | 25% | 4,8 | 1,20 |
| Koherensi Argumen | 15% | 6,2 | 0,93 |
| Kualitas Tulisan | 15% | 8,0 | 1,20 |
| **Total** | | | **≈ 5,6 / 10** |

### ⚖️ Verdict: **MAJOR REVISION**

Naskah memiliki fondasi kuat (eksperimen rapi, hasil negatif yang jujur, tulisan baik), tetapi klaim sentralnya belum didukung secara memadai karena konfounding dan keterbatasan daya statistik. Bukan penolakan—isu-isunya dapat diperbaiki—tetapi memerlukan revisi besar pada klaim dan/atau eksperimen.

### Peta Revisi (berprioritas)

**P1 — Wajib (menentukan diterima/tidak):**
1. **Persempit klaim inti** di Abstrak, §V, §VII: dari "atensi tidak membantu" → "di bawah protokol ter-tuning-YOLO pada dataset kecil yang mudah, penambahan atensi tidak meningkatkan akurasi dan menambah biaya". (Menjawab DA-CRITICAL.)
2. **Atasi efek langit-langit / daya statistik:** tambahkan kondisi ber-*headroom* (subset lebih kecil/sulit atau dataset kedua yang lebih menantang) **atau** akui eksplisit keterbatasan daya. (DA-CRITICAL.)
3. **Uji statistik yang benar:** ganti heuristik dengan uji-t berpasangan/selang kepercayaan + *effect size*.

**P2 — Sangat disarankan:**
4. **Perbandingan *size-matched*** (mis. YOLOv8m vs RT-DETR-l) untuk memisahkan kapasitas dari arsitektur.
5. **Lengkapi RT-DETR ke n=3** atau turunkan klaim kesetaraannya.
6. **Ablasi CBAM** (penempatan, latih-dari-awal, durasi) untuk menguji apakah penurunan bersifat intrinsik atau artefak.

**P3 — Penguat:**
7. **Perluas literatur** (survei atensi-pada-deteksi, Swin/ViTDet, karya helm berbasis atensi, studi Indonesia).
8. **Validasi lintas-dataset** untuk validitas eksternal.
9. **Protokol FPS seragam** + laporan latensi.
10. **Pertegas framing kontribусi** sebagai bantahan terstruktur terhadap asumsi "atensi selalu membantu".

> Catatan: jika revisi eksperimental (P2/P3) di luar anggaran, naskah masih dapat naik kelas dengan **mempersempit klaim secara tegas (P1)** sehingga kesimpulan sepadan dengan bukti—jalur tercepat menuju layak-publikasi.
