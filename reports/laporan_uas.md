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
  - K-NN from scratch dengan cosine similarity manual memberikan kontrol penuh dan interpretability.
  - Feature selection (Chi-square) meningkatkan macro-F1 hingga 19% dengan reduksi dimensi 62%.
  - Clustering K-Means menghasilkan 3 kelompok meaningful: FAQ/Panduan, Profil Institusional, dan Regulasi.
  
- Keterbatasan:
  - Korpus relatif kecil (12 dokumen); generalisasi belum teruji pada skala besar.
  - Stemmer sederhana; potensi peningkatan dengan Sastrawi/lemmatization.
  - Lexicon sentimen statis; belum memanfaatkan pembelajaran berbasis data.
  - K-NN sensitif terhadap class imbalance (Cluster Regulasi hanya 1 dokumen).
  - Error analysis menunjukkan vocabulary overlap tinggi antara FAQ dan Profil menyebabkan misklasifikasi.
  
- Arah Pengembangan:
  - Perluasan korpus dan anotasi relevansi yang lebih kaya (minimal 20 dokumen per kategori).
  - Pencarian semantik lanjutan (embeddings/BERT) dan reranking berbasis ML.
  - Abstractive summarization & active‑learning dari umpan balik pengguna.
  - Implementasi weighted voting untuk handle class imbalance.
  - Augmentasi data untuk balance cluster dan kategori classification.

### Etika dan Keterbatasan Sentiment Analysis

#### A. Keterbatasan Teknis
1. **Lexicon-Based Approach Limitations**:
   - Lexicon statis (94 kata) tidak dapat menangkap konteks dinamis kampus
   - Threshold ±0.05 bersifat arbitrary dan tidak adaptif terhadap domain spesifik
   - Tidak dapat mendeteksi sarkasme, ironi, atau sentiment implisit
   - Contoh kasus: "Biaya kuliah cukup terjangkau" → positif, tapi konteks "cukup" bisa ambigu

2. **Bias dalam Lexicon**:
   - Lexicon dibangun manual dengan bias subjektif peneliti
   - Domain-specific words ("jadwal", "pendaftaran") diberi skor netral padahal bisa bermuatan sentiment dalam konteks tertentu
   - Tidak ada validasi lexicon terhadap corpus kampus sebenarnya

3. **Context Insensitivity**:
   - "Proses pendaftaran tidak rumit" → sistem deteksi "tidak" + "rumit" (negatif) padahal maksudnya positif
   - Negasi handling belum diimplementasikan (akan meningkatkan false negative)

#### B. Bias dan Fairness Issues

1. **Representation Bias**:
   - Corpus 12 dokumen hanya merepresentasikan perspektif institusi (official documents)
   - **Missing**: Suara mahasiswa, keluhan/feedback aktual, opini minoritas
   - Berpotensi menghasilkan "echo chamber" → hanya sentiment positif/netral yang terdeteksi

2. **Language Bias**:
   - Fokus pada Bahasa Indonesia formal/baku
   - **Tidak menangkap**: Bahasa gaul mahasiswa, campur kode (Inggris-Indonesia), dialek lokal
   - Mahasiswa non-native speaker atau dari daerah mungkin ter-diskriminasi dalam sentiment detection

3. **Demographic Bias**:
   - Sistem tidak mempertimbangkan variasi sentiment berdasarkan:
     - Fakultas (teknik vs sosial mungkin punya sentiment berbeda terhadap "teknologi")
     - Angkatan (mahasiswa baru vs senior)
     - Latar belakang ekonomi (sentiment tentang "biaya kuliah")

#### C. Ethical Concerns dalam Konteks Kampus

1. **Privacy & Consent**:
   - **Risiko**: Jika diterapkan pada feedback mahasiswa tanpa informed consent
   - Mahasiswa mungkin tidak tahu sentiment mereka dianalisis → pelanggaran privasi
   - **Solusi**: Transparency notice + opt-in mechanism

2. **Surveillance & Power Imbalance**:
   - Institusi menggunakan sentiment analysis untuk "monitor" mahasiswa → chilling effect
   - Mahasiswa mungkin self-censor jika tahu opini mereka dianalisis
   - **Contoh bahaya**: Identifikasi mahasiswa dengan sentiment negatif tinggi → penalized atau di-blacklist

3. **Misuse of Results**:
   - **Skenario buruk**: 
     - Sistem digunakan untuk filtering beasiswa → mahasiswa dengan sentiment negatif ditolak
     - Ranking fakultas berdasarkan sentiment → fakultas dengan feedback jujur dihukum
     - Automated decision tanpa human review → lack of accountability

4. **Amplification of Institutional Power**:
   - Sentiment analysis hanya memvalidasi perspektif institusi (karena corpus bias)
   - Kritik legitimate dari mahasiswa mungkin di-dismiss sebagai "negative sentiment"
   - **Mengabaikan**: Structural issues yang perlu diperbaiki

#### D. Rekomendasi Ethical Guidelines

1. **Transparency & Explainability**:
   - ✅ Jelaskan ke stakeholder bagaimana sentiment dihitung (lexicon-based, threshold, etc.)
   - ✅ Provide sentiment score breakdown (positive words count, negative words count)
   - ✅ Allow users to see WHY their text classified as positive/negative/neutral

2. **Human-in-the-Loop**:
   - ❌ JANGAN gunakan sentiment analysis untuk automated decision-making
   - ✅ Gunakan sebagai **assisting tool** untuk human reviewers
   - ✅ Final decision harus melibatkan human judgment

3. **Fairness & Representation**:
   - ✅ Diversify training data: include mahasiswa feedback dari berbagai fakultas, angkatan, latar belakang
   - ✅ Regular audit untuk bias: test performance across demographic groups
   - ✅ Multi-stakeholder lexicon validation (mahasiswa, dosen, admin)

4. **Purpose Limitation**:
   - ✅ **Boleh digunakan untuk**:
     - Aggregate analysis (overall sentiment trend fakultas/program)
     - Prioritization of feedback for review (bukan filtering)
     - Identifying systemic issues (many students complain about same thing)
   - ❌ **JANGAN digunakan untuk**:
     - Individual student profiling
     - Admission/scholarship decisions
     - Penalizing negative feedback
     - Marketing manipulation

5. **Consent & Opt-Out**:
   - ✅ Inform users sebelum sentiment analysis dilakukan
   - ✅ Provide opt-out option untuk mahasiswa yang tidak mau feedback-nya dianalisis
   - ✅ Anonymization jika hasil dipublikasikan

6. **Regular Evaluation & Accountability**:
   - ✅ Periodic review of lexicon relevance (update setiap semester)
   - ✅ Accuracy audit: manual check sample of classifications
   - ✅ Establish complaint mechanism untuk mahasiswa yang merasa misrepresented
   - ✅ Publicly report limitations dan bias sistem

#### E. Kesimpulan Etika

**Sentiment Analysis di lingkungan kampus adalah "double-edged sword":**
- ✅ **Potensi positif**: Membantu institusi memahami kebutuhan mahasiswa, improve services, identify issues early
- ⚠️ **Risiko besar**: Privacy invasion, surveillance, bias amplification, power abuse

**Prinsip Kunci:**
1. **"Do No Harm"**: Prioritize student welfare over institutional efficiency
2. **"Transparency over Accuracy"**: Better to be honest about limitations than to oversell capabilities
3. **"Human Dignity"**: Students are not data points; respect their autonomy and voice
4. **"Continuous Improvement"**: Ethics is not one-time checklist; requires ongoing reflection

**Untuk SmartCampus Assistant:**
- Sistem saat ini **HANYA** untuk demo akademik, bukan production use
- Jika akan di-deploy, **WAJIB** conduct ethical review oleh komite independen
- **Rekomendasi**: Use sentiment as **exploratory tool** for understanding corpus, bukan decision-making

### Kesimpulan
SmartCampus Assistant berhasil menunjukkan bahwa kombinasi TF‑IDF + VSM, dibantu sentiment analysis dan extractive summarization, mampu menghadirkan pencarian dokumen kampus yang akurat, terurut dengan baik, dan mudah dicerna. Hasil evaluasi (MAP 0,861; MRR 0,833; nDCG@3 0,883) mengindikasikan kualitas ranking yang sangat baik pada korpus studi kasus ini.

Implementasi K-NN from scratch dengan cosine similarity manual memberikan kontrol penuh terhadap proses klasifikasi dan memungkinkan analisis mendalam terhadap neighbors yang mempengaruhi prediksi. Error analysis mengidentifikasi 5 penyebab utama misklasifikasi: dataset kecil, class imbalance, feature reduction aggressive, vocabulary overlap, dan k value terlalu kecil. 

Feature selection menggunakan Chi-square terbukti paling efektif, meningkatkan macro-F1 sebesar 19% sambil mengurangi dimensi 62% (dari 265 ke 100 features). Clustering K-Means menghasilkan 3 kelompok meaningful dengan silhouette score 0.452: FAQ/Panduan Akademik, Profil Institusional, dan Regulasi/Tata Tertib.

Dari perspektif etika, sentiment analysis di lingkungan kampus memiliki potensi positif namun juga risiko signifikan terkait privacy, bias, dan power imbalance. Rekomendasi ethical guidelines mencakup transparency, human-in-the-loop, fairness audit, purpose limitation, consent mechanism, dan regular evaluation. Sistem ini siap dipakai sebagai fondasi penelitian dan pembelajaran STKI, namun memerlukan ethical review komprehensif sebelum deployment production.

Solusi siap dipakai sebagai fondasi sistem informasi kampus yang scalable dan dapat ditingkatkan ke pendekatan semantik modern, dengan catatan penting bahwa aspek etika dan fairness harus menjadi pertimbangan utama dalam setiap tahap pengembangan dan deployment.
