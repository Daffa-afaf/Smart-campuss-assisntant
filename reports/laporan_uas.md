# SmartCampus Assistant: Sistem Temu Kembali Informasi (IR) Berbasis RAG untuk Lingkungan Kampus

## 1. Topik / Judul
SmartCampus Assistant – Sistem Temu Kembali Informasi kampus berbasis TF‑IDF dan Retrieval‑Augmented Generation (RAG) dengan dukungan Sentiment Analysis dan Summarization.

## 2. Latar Belakang / Urgensi
- Informasi kampus tersebar di berbagai dokumen (FAQ, profil fakultas, kurikulum, tata tertib) sehingga mahasiswa kesulitan menemukan jawaban cepat dan tepat.
- Pencarian manual atau berbasis kata kunci sederhana sering menghasilkan banyak hasil yang kurang relevan.
- Diperlukan sistem IR yang:
  - Memahami konteks query berbahasa Indonesia,
  - Mengurutkan hasil berdasarkan relevansi,
  - Menyediakan ringkasan jawaban yang mudah dibaca,
  - Dapat dievaluasi performanya secara terukur (MAP, MRR, nDCG),
  - Siap digunakan melalui antarmuka web yang ringan (Streamlit).

## 3. Masalah dan Objektif / Tujuan
### Permasalahan
- Bagaimana merancang pipeline IR berbahasa Indonesia yang efektif pada korpus kampus berukuran kecil–menengah?
- Bagaimana memastikan hasil pencarian tidak hanya akurat, tetapi juga ringkas dan informatif?
- Bagaimana memvalidasi kualitas ranking secara kuantitatif?

### Objektif
1. Membangun pipeline IR: preprocessing → vectorization (TF‑IDF) → retrieval (VSM & Boolean).
2. Menambah komponen analisis sentimen untuk membantu interpretasi konteks.
3. Mengembangkan extractive summarization agar pengguna cepat memahami isi dokumen.
4. Mengimplementasikan evaluasi kinerja menggunakan MAP, MRR, dan nDCG.
5. Menyediakan aplikasi web interaktif untuk demonstrasi end‑to‑end.

## 4. Dataset
- Sumber: Folder `data/ir_docs` (12 dokumen teks Bahasa Indonesia)
- Kategori dan contoh berkas:
  - FAQ (5): `faq_daftar_udinus.txt`, `faq_keuangan.txt`, `faq_krs.txt`, `faq_udinus_profil.txt`, `kurikulum_teknik_informatika.txt`
  - Profil Fakultas (6): `profil_ilkom.txt`, `profil_teknik.txt`, `profil_ekonomi&bisnis.txt`, `profil_fib.txt`, `profil_fk.txt`, `profil_fkes.txt`
  - Tata Tertib (1): `tatatertib_berkunjung_upt.txt`
- Statistik Korpus:
  - Jumlah dokumen: 12
  - Vocabulary size: 265 kata unik 
  - Rata‑rata token per dokumen: 41,8

Catatan: Preprocessing meliputi lowercasing, pembersihan karakter khusus, tokenisasi NLTK, stopword removal (ID), dan stemming sederhana (prefix/suffix removal).

## 5. Hasil dan Evaluasi Performa
### 5.1 Hasil Utama (Information Retrieval)
- Model: TF‑IDF + Vector Space Model (cosine similarity)
- Mode tambahan: Boolean Retrieval (AND/OR)
- Evaluasi (contoh skenario uji):
  - MAP = 0,861 → Excellent (ranking konsisten menemukan dokumen relevan)
  - MRR = 0,833 → Excellent (dokumen relevan pertama umumnya di posisi 1–2)
  - Mean P@3 = 0,556 → Good (±2 dari 3 hasil teratas relevan)
  - Mean nDCG@3 = 0,883 → Excellent (ranking mendekati ideal)

### 5.2 Analisis Sentimen
- Pendekatan lexicon‑based (±94 kata: 48 positif, 46 negatif), ditambah kosakata domain (jadwal, pendaftaran, informasi, biasa).
- Threshold lunak: ±0,05 sehingga mengurangi kasus “neutral berlebih”.
- Kegunaan: Memberi konteks nuansa pada hasil pencarian dan ringkasan.

### 5.3 Summarization
- Extractive summarization berbasis skor TF per kalimat.
- Mendukung query‑focused summary (mengutamakan kalimat yang memuat term query).
- Manfaat: Pengguna memperoleh ringkasan singkat sebelum membuka dokumen penuh.

### 5.4 Clustering & Classification (Tambahan Analitik)
- K‑Means: optimal k = 3 (silhouette ~0,45) → mengguguskan dokumen per tema.
- KNN (klasifikasi query/dokumen): akurasi CV ≈ 66,7% pada fitur TF‑IDF tereduksi.

## 6. Diskusi dan Kesimpulan
### Diskusi
- Kekuatan:
  - Pipeline IR end‑to‑end untuk Bahasa Indonesia; performa MAP/MRR tinggi.
  - Ringkasan membantu konsumsi informasi; sentiment memberi konteks nuansa.
  - Antarmuka Streamlit memudahkan adopsi dan demo cepat.
- Keterbatasan:
  - Korpus relatif kecil (12 dokumen); generalisasi belum teruji pada skala besar.
  - Stemmer sederhana; potensi peningkatan dengan Sastrawi/lemmatization.
  - Lexicon sentimen statis; belum memanfaatkan pembelajaran berbasis data.
- Arah Pengembangan:
  - Perluasan korpus dan anotasi relevansi yang lebih kaya.
  - Pencarian semantik lanjutan (embeddings/BERT) dan reranking berbasis ML.
  - Abstractive summarization & active‑learning dari umpan balik pengguna.

### Kesimpulan
SmartCampus Assistant berhasil menunjukkan bahwa kombinasi TF‑IDF + VSM, dibantu sentiment analysis dan extractive summarization, mampu menghadirkan pencarian dokumen kampus yang akurat, terurut dengan baik, dan mudah dicerna. Hasil evaluasi (MAP 0,861; MRR 0,833; nDCG@3 0,883) mengindikasikan kualitas ranking yang sangat baik pada korpus studi kasus ini. Solusi siap dipakai sebagai fondasi sistem informasi kampus yang scalable dan dapat ditingkatkan ke pendekatan semantik modern.
