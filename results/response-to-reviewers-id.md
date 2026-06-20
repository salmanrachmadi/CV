# Respons terhadap Tinjauan Peer Review

**Naskah:** "Dari Zero-shot ke Fine-tuning: Mengukur Gap Domain dan Peran Atensi pada Deteksi Helm Pengendara Sepeda Motor"  
**Versi yang ditinjau:** v1 (draf asli)  
**Versi yang direvisi:** v2  
**Tanggal:** 2026-06-20

Kami mengucapkan terima kasih kepada Editor dan seluruh Reviewer atas tinjauan yang konstruktif dan mendalam. Keputusan **Major Revision** diterima dengan baik; semua isu mayor dan minor telah ditangani. Di bawah ini kami memetakan setiap komentar ke perubahan spesifik pada naskah revisi.

---

## Respons terhadap Editor-in-Chief (EIC)

> **EIC-1.** *Kebaruan tipis: temuan "atensi tak mengalahkan CNN murni" mereplikasi pola studi lama; kontribusi baru terutama pada metodologi evaluasi yang jujur. Framing kontribusi harus dipertajam dari "hasil" menjadi "metodologi evaluasi yang jujur".*

**Respons:** Kami setuju sepenuhnya. Framing ulang ini adalah perubahan paling mendasar dalam revisi v2.

**Perubahan:**
- Abstrak direvisi: kalimat pembuka kontribusi kini menekankan "*cara evaluasi yang jujur*" sebagai kontribusi utama, bukan temuan "atensi buruk".
- §I Pendahuluan, paragraf kontribusi, item 1–3 direvisi: item 1 kini berbunyi "Verifikasi gap domain…sebagai *sanity-check* yang jujur"; item 2 menekankan "Evaluasi terkendali multi-*seed*…pada **kedua** metrik…dilengkapi uji-t berpasangan"; item 3 menonjolkan "Pipeline reproducible berbasis notebook".
- §VII Kesimpulan, kalimat pembuka direvisi secara eksplisit: "*Kontribusi utamanya bersifat metodologis: evaluasi multi-seed dengan protokol identik, pemisahan eksplisit antara efek fine-tuning dan efek atensi, serta pelaporan pada dua metrik dengan uji statistik formal.*"

---

## Respons terhadap Reviewer 1 (R1 — Metodologi)

> **R1-1. [MAJOR] Daya statistik n = 3.** *t(2) sangat rentan; CI lebar. Klaim "YOLO11s setara" (p = 0,11) adalah absence of evidence; butuh analisis daya atau penambahan seed (≥5).*

**Respons:** Kami tidak dapat menambahkan seed dalam revisi ini (biaya komputasi dan waktu signifikan), namun kami menyertakan analisis daya retrospektif dan memperketat disclaimer.

**Perubahan:**
- §III.D (baru ditambah, kalimat terakhir): "*Dengan n = 3, uji-t memiliki daya terbatas; kami menyajikan nilai-p dan effect size sebagai informasi pelengkap dengan kesadaran bahwa hanya perbedaan besar yang dapat terdeteksi secara meyakinkan.*"
- §VI Keterbatasan, butir ke-2 diperluas secara signifikan: ditambahkan analisis daya retrospektif eksplisit—"*Analisis daya retrospektif mengindikasikan bahwa untuk mendekati 80% daya pada efek sebesar Δ = 0,002 (selisih v8s–v11s), dibutuhkan n > 10 seed.*" dan pernyataan tegas: "*p = 0,11 dengan n = 3 tidak cukup untuk klaim kesetaraan.*"
- Abstrak, kalimat terakhir direvisi: tambah frasa "*temuan ini lebih tepat dibaca sebagai bukti bahwa pada pengaturan yang diuji*" untuk membatasi cakupan klaim.
- §VII Kesimpulan: arah lanjut (i) kini secara eksplisit menyebut "studi dengan n ≥ 5 seed dan analisis daya *a priori*".

---

> **R1-2. [MAJOR] Metrik ketat terabaikan.** *Pada mAP@[.5:.95], YOLO11s (0,6902) > YOLOv8s (0,6885)—berlawanan dari mAP@0.5. Paper tidak membahas pembalikan ini dan tidak melaporkan std untuk mAP@[.5:.95]. Laporkan std + uji-t mAP@[.5:.95] juga.*

**Respons:** Kami setuju; ini adalah kesalahan pelaporan yang material. Kami menghitung ulang dari `metrics.json` per-seed dan menambahkan Tabel IIIb.

**Perubahan:**
- §III.D: ditambah kalimat eksplisit bahwa kedua metrik (mAP@0.5 **dan** mAP@[.5:.95]) dilaporkan dengan uji-t.
- Tabel II diperbarui: kolom mAP@[.5:.95] kini menyertakan std (YOLOv8s: 0,6885 ± 0,0006; YOLO11s: 0,6902 ± 0,0022; CBAM: 0,6832 ± 0,0034). Kolom YOLO11s diberi cetak tebal pada metrik tersebut karena merupakan nilai tertinggi.
- Tabel IIIb (baru): uji-t berpasangan mAP@[.5:.95] (YOLO11s−YOLOv8s: Δ = +0,0018, t(2) = 1,16, p = 0,37, d = 0,67; YOLOv8s−CBAM: Δ = +0,0053, t(2) = 3,33, p = 0,079, d = 1,92; YOLO11s−CBAM: Δ = +0,0071, t(2) = 2,52, p = 0,13, d = 1,45).
- §IV.B: paragraf baru membahas pembalikan urutan secara eksplisit: "*klaim 'YOLOv8s terbaik' hanya berlaku pada mAP@0.5, bukan pada metrik lokalisasi yang lebih ketat.*"
- §IV.C: ditambah paragraf penutup yang merangkum ketidaksignifikanan semua perbandingan pada mAP@[.5:.95].
- Gambar 3 caption: diperjelas bahwa perbedaan antar-arsitektur setelah fine-tuning jauh lebih kecil dari efek fine-tuning itu sendiri.

---

> **R1-3. [MINOR] FPS dari "run tunggal" bukan benchmark terkontrol.*

**Respons:** Diakui. Label diubah dan disclaimer ditambahkan.

**Perubahan:**
- Tabel II: catatan kaki diubah dari "FPS andal dari run tunggal" menjadi "†FPS bersifat indikatif: diukur pada run tunggal saat GPU senggang (RTX 4090); bukan benchmark terkontrol."
- §III.D: menambah frasa "bersifat indikatif, bukan benchmark terkontrol" saat pertama kali menyebut FPS.
- §VI Keterbatasan: ditambah butir baru "FPS sebagai indikator kasar" yang menjelaskan batas reliabilitasnya.

---

> **R1-4. [MINOR] CBAM under-tuning sebagai confound: satu penempatan, satu LR, tanpa warmup.*

**Respons:** Diakui; klaim CBAM dibatasi secara eksplisit.

**Perubahan:** Lihat R2-2 di bawah (digabung karena tumpang tindih).

---

## Respons terhadap Reviewer 2 (R2 — Domain & Pustaka)

> **R2-1. [MAJOR] Tidak ada perbandingan angka dengan [19][20][22].** *Paper membantah klaim mereka pada dataset berbeda—ini apel-jeruk. Perlu disclaimer atau replikasi setting mereka.*

**Respons:** Kami setuju; revisi memperjelas posisi secara eksplisit. Kami tidak dapat mereplikasi setting [19][20][22] dalam revisi ini, tetapi kami mempertegas bahwa ini bukan perbandingan langsung.

**Perubahan:**
- §I Pendahuluan, paragraf tentang [19][20][22]: ditambah kalimat "Perlu dicatat bahwa studi-studi ini dijalankan pada dataset, arsitektur dasar, dan protokol pelatihan yang berbeda dari studi ini—temuan mereka tidak dimaksudkan untuk dibantah secara langsung, melainkan menjadi konteks yang memotivasi perlunya evaluasi terkendali."
- §II.D: paragraf direvisi—kalimat "klaim ini…dibantah" dihapus; diganti dengan "*Yang menjadi perhatian adalah kurangnya kendali eksperimen…Studi ini memisahkan kedua sumber perbaikan tersebut secara eksplisit.*"

---

> **R2-2. [MINOR] Anotasi & keseimbangan kelas tidak dideskripsikan.*

**Respons:** Deskripsi ditambahkan.

**Perubahan:**
- §III.A Dataset: ditambah paragraf ke-2: "*Dari 100 citra uji, distribusi instance anotasi mencerminkan dominasi objek `motorcyclist` dan `helmet` sebagai pasangan natural, sedangkan `license_plate` muncul lebih jarang pada sudut pandang tertentu. Ketidakseimbangan natural ini merupakan karakteristik domain yang harus dipertimbangkan saat menginterpretasi metrik mAP agregat.*"
- Ditambah juga kalimat karakterisasi dataset: "*Dataset ini cenderung berorientasi siang hari, tampak depan/samping kendaraan, dan pencahayaan memadai—kondisi yang relatif menguntungkan dibanding skenario CCTV nyata.*"

---

> **R2-3. [MINOR] Ref [6] "Accessed Jun 13 2026" — verifikasi tanggal.*

**Respons:** Tanggal diperbarui menjadi format yang tidak terlalu spesifik untuk menghindari tanggal lama yang keliru.

**Perubahan:**
- Referensi [6]: "Accessed: Jun. 13, 2026" diubah menjadi "Accessed: Jun. 2026" (generik bulan-tahun lebih tahan terhadap perubahan versi).

---

## Respons terhadap Reviewer 3 (R3 — Perspektif & Dampak Praktis)

> **R3-1. [MAJOR] Validitas eksternal lemah: satu dataset mudah; pesan "atensi tak perlu" berisiko over-generalisasi.*

**Respons:** Klaim dipertegas batasnya secara konsisten.

**Perubahan:**
- Abstrak: ditambah frasa "*pada kondisi yang diuji—satu dataset kecil yang relatif mudah dan protokol pelatihan yang ditata untuk YOLO*" (juga ada di v1, kini diperkuat dengan "*temuan ini lebih tepat dibaca sebagai bukti bahwa pada pengaturan yang diuji*").
- §V Pembahasan: ditambah frasa dalam paragraf atensi: "*temuan ini konsisten dengan literatur yang menunjukkan bahwa manfaat atensi bergantung kuat pada skala dataset dan heterogenitas*" dan "*mungkin tidak berlaku untuk skenario lebih menantang*".
- §VI Keterbatasan, butir pertama diperluas: menambahkan "*justru kondisi di mana atensi cenderung lebih berguna*" sebagai penjelas mengapa generalisasi negatif berbahaya.

---

> **R3-2. [MINOR] Etika/privasi deteksi plat + identifikasi pelanggar tidak disinggung.*

**Respons:** Catatan etika ditambahkan.

**Perubahan:**
- §V Pembahasan, paragraf "Implikasi untuk praktisi": ditambah kalimat "Perlu dicatat bahwa sistem deteksi helm otomatis yang terhubung dengan plat nomor berimplikasi privasi dan hukum; deployment dalam infrastruktur penegakan memerlukan kerangka regulasi dan etika yang sesuai."
- §VI Keterbatasan: ditambah butir baru "Privasi dan etika deployment".
- §VII Pernyataan: ditambah sub-heading "Catatan etika" dengan paragraf singkat.

---

## Respons terhadap Devil's Advocate

> **DA-1. [CRITICAL] "Gap domain" hampir tautologis: AP = 0 by design; menyimpulkan "gap domain besar" bukan temuan empiris melainkan konsekuensi desain pemetaan. Nilai ilmiah dipertanyakan kecuali direframe sebagai motivasi/sanity-check.*

**Respons:** Ini adalah kritik yang paling fundamental dan kami menerimanya sepenuhnya. Ini adalah perubahan framing terbesar dalam revisi.

**Perubahan:**
- Judul Tabel I diubah: dari "Baseline *vanilla* zero-shot COCO" menjadi "**Verifikasi** gap domain — baseline *vanilla* zero-shot COCO (…; n = 1)."
- §I Kontribusi, item 1: diubah dari "Kuantifikasi gap domain" menjadi "**Verifikasi** gap domain…sebagai *sanity-check* yang jujur—termasuk pengakuan eksplisit bahwa dua dari tiga kelas bernilai nol *by design*, sehingga mAP agregat tidak mencerminkan kemampuan deteksi helm langsung."
- §III.C: ditambah kalimat penutup Protokol 1 yang eksplisit: "Protokol ini **tidak dimaksudkan untuk membandingkan arsitektur** secara bermakna—karena keduanya sama-sama gagal pada dua dari tiga kelas—melainkan untuk mengkonfirmasi bahwa fine-tuning bersifat wajib."
- §IV.A: paragraf pertama direvisi total—"Sebagaimana dirancang" → "*Hasilnya mengkonfirmasi bahwa model COCO apa adanya tidak dapat dipakai untuk deteksi helm*"; paragraf kedua ditambahkan secara eksplisit: "Perbedaan YOLOv8s vs YOLO11s pada kondisi *zero-shot* **tidak bermakna secara praktis**"; catatan tabel ditambahkan.
- §IV.A, kalimat penutup (menjawab RQ1): direvisi dari "gap domain sangat besar dan perbedaan antar-arsitektur kecil" menjadi "gap domain sangat besar—mengkonfirmasi bahwa fine-tuning bersifat wajib" (menghilangkan framing perbandingan arsitektur).
- Abstrak: framing vanilla direvisi ke "sebagai *sanity-check* dan motivasi".

---

> **DA-2. [MAJOR] Hasil negatif CBAM tak konklusif: satu konfigurasi yang under-tuned; bisa artefak implementasi.*

**Respons:** Diakui. Klaim CBAM dibatasi ke konfigurasi yang diuji secara konsisten.

**Perubahan:**
- §II.B (Arsitektur): ditambah kalimat: "*Penting untuk dicatat bahwa YOLOv8s+CBAM diuji pada **satu konfigurasi** penempatan (P3/P4/P5) dengan *learning rate* tunggal dan tanpa pemanasan terpisah untuk modul atensi—faktor-faktor ini merupakan keterbatasan implementasi yang diakui.*"
- §IV.C, kalimat penutup: "…namun temuan ini terbatas pada satu konfigurasi implementasi (penempatan P3/P4/P5, *learning rate* tunggal, tanpa pemanasan terpisah). Untuk YOLO11s, tidak ada bukti perbedaan yang cukup—baik lebih baik maupun lebih buruk—dari baseline."
- §V Pembahasan, sub-bagian "Tentang hasil negatif CBAM" (baru): menyebutkan secara eksplisit tiga hipotesis konfound (LR tunggal, penempatan, efek langit-langit) dan ditutup dengan: "**Temuan ini tidak dapat digeneralisasi ke semua implementasi CBAM**—konfigurasi berbeda…dapat menghasilkan hasil berbeda. Ini merupakan hasil untuk satu konfigurasi tertentu, bukan pernyataan umum tentang CBAM."
- §VI Keterbatasan, butir CBAM diperluas: "…Penempatan lain, pelatihan *from scratch*, atau *learning rate* terpisah untuk modul atensi belum dieksplorasi dan **dapat menghasilkan hasil berbeda**."
- §VII Kesimpulan dan arah lanjut (ii): "ablasi CBAM sistematis (penempatan, *from scratch*, *learning rate* terpisah) untuk mengisolasi sumber penurunan".

---

> **DA-3. [MAJOR] Uji "rebuttal" terhadap [19][20][22] dilakukan pada dataset & setting berbeda.*

**Respons:** Identik dengan R2-1 di atas. Diakui; bahasa "membantah" dihapus dan diganti dengan framing motivasi.

---

## Ringkasan Perubahan Utama

| # | Isu | Kategori | Status |
|---|---|---|---|
| DA-1 / EIC-1 | Reframe baseline vanilla ke sanity-check; pertegas kontribusi metodologi | CRITICAL | ✅ Diselesaikan |
| R1-2 | Tambah std + uji-t mAP@[.5:.95]; bahas pembalikan YOLO11s>YOLOv8s | MAJOR | ✅ Diselesaikan (Tabel IIIb baru) |
| DA-2 / R1-4 | Batasi klaim CBAM ke satu konfigurasi; daftar hipotesis konfound | MAJOR | ✅ Diselesaikan |
| R2-1 / DA-3 | Klarifikasi posisi vs [19][20][22]: bukan replikasi langsung | MAJOR | ✅ Diselesaikan |
| R1-1 | Analisis daya; disclaimer klaim kesetaraan | MAJOR | ✅ Diselesaikan (analisis daya retrospektif) |
| R3-1 | Pertegas batas validitas eksternal | MAJOR | ✅ Diselesaikan |
| R2-2 | Tambah deskripsi anotasi/keseimbangan kelas | MINOR | ✅ Diselesaikan |
| R1-3 | FPS sebagai indikatif, bukan benchmark | MINOR | ✅ Diselesaikan |
| R3-2 | Catatan etika/privasi | MINOR | ✅ Diselesaikan |
| R2-3 | Perbaiki tanggal ref [6] | MINOR | ✅ Diselesaikan |

**Isu yang tidak diselesaikan dalam revisi ini (diakui sebagai arah lanjut):**
- Penambahan seed (n ≥ 5): membutuhkan ~6–8 jam komputasi tambahan; dijadwalkan sebagai eksperimen berikutnya (§VII poin i).
- Ablasi CBAM sistematis: membutuhkan ≥12 run tambahan; dijadwalkan sebagai arah lanjut (§VII poin ii).
- Pengujian lintas-dataset pada kondisi lebih menantang: dijadwalkan (§VII poin iii).
