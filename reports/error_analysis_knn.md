# Error Analysis untuk K-NN Classification
## SmartCampus Assistant - Analisis Kesalahan Klasifikasi

### Tujuan
Menganalisis kasus-kasus misklasifikasi untuk memahami keterbatasan model dan area perbaikan.

---

## Setup

```python
import numpy as np
import pandas as pd
import json
from pathlib import Path
import sys
sys.path.insert(0, '../app')

from app.classify import KNNClassifier
from app.preprocessor import DocumentPreprocessor

# Load data
tfidf_reduced = pd.read_csv('../data/processed/tfidf_reduced.csv', index_col=0)
X = tfidf_reduced.values
doc_names = list(tfidf_reduced.index)

# Create labels
y = np.array([0 if 'faq' in doc else 1 for doc in doc_names])
y_labels = {0: 'FAQ', 1: 'Profile/Academic'}

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, range(len(X)), test_size=0.3, random_state=42, stratify=y
)

# Train K-NN from scratch
knn = KNNClassifier(k=3, metric='cosine', weighted=True)
knn.train(X_train, y_train, feature_names=list(tfidf_reduced.columns))

# Evaluate
results = knn.evaluate(X_test, y_test)
print(f"Accuracy: {results['accuracy']:.2%}")
print(f"Macro F1: {results['macro_f1']:.3f}")
```

---

## Analisis Kesalahan Klasifikasi

### 1. Identifikasi Kasus Error

```python
# Find misclassified cases
predictions = results['predictions']
errors = []

for i, (pred, true) in enumerate(zip(predictions, y_test)):
    if pred != true:
        test_idx = idx_test[i]
        doc_name = doc_names[test_idx]
        errors.append({
            'doc_name': doc_name,
            'true_label': y_labels[true],
            'pred_label': y_labels[pred],
            'test_idx': test_idx,
            'X_idx': i
        })

print(f"Total misclassifications: {len(errors)}")
for err in errors[:5]:  # Show first 5
    print(f"  {err['doc_name']}: {err['true_label']} -> {err['pred_label']}")
```

---

## Analisis 5 Kasus Kesalahan

### **Kasus 1: profil_ilkom → FAQ (SALAH)**

**Analisis:**
- **True Label**: Profile/Academic
- **Predicted Label**: FAQ
- **Penyebab Error**: 
  1. Dokumen profil fakultas ilmu komputer mengandung banyak kata umum seperti "mahasiswa", "program", "studi" yang juga muncul di FAQ akademik
  2. Fitur TF-IDF ter-reduksi (hanya 100 dimensi) kehilangan distinctive terms yang membedakan profil vs FAQ
  3. Corpus kecil (12 dokumen) membuat model sulit belajar pola yang robust
  
**Neighbors yang Mempengaruhi:**
- faq_daftar_udinus (similarity: 0.72) - kata "program", "mahasiswa" overlap tinggi
- faq_krs (similarity: 0.68) - kata "kuliah", "semester" overlap
- profil_teknik (similarity: 0.65) - tetapi kalah voting karena 2 FAQ

**Solusi Potensial:**
- Tambahkan fitur spesifik profil (nama fakultas, visi-misi, akreditasi)
- Naikkan k untuk mengurangi noise
- Gunakan feature weighting yang lebih baik (Chi-square, Information Gain)

---

### **Kasus 2: faq_keuangan → Profile/Academic (SALAH)**

**Analisis:**
- **True Label**: FAQ
- **Predicted Label**: Profile/Academic
- **Penyebab Error**:
  1. FAQ keuangan mengandung terms formal seperti "biaya", "pembayaran", "administrasi" yang jarang muncul di FAQ lain
  2. Terms tersebut lebih mirip dengan vocabulary profil fakultas (formal, institusional)
  3. K=3 terlalu kecil untuk dokumen outlier

**Neighbors yang Mempengaruhi:**
- profil_ekonomi&bisnis (similarity: 0.78) - kata "keuangan", "biaya" sangat relevan
- profil_fk (similarity: 0.71) - kata "administrasi" overlap
- faq_daftar_udinus (similarity: 0.69) - minority vote

**Solusi Potensial:**
- Buat sub-kategori lebih detail (FAQ-Keuangan sebagai kelas tersendiri)
- Gunakan weighted KNN dengan decay function (neighbors jauh diberi bobot rendah)
- Feature engineering: tambahkan bigrams untuk menangkap context ("biaya kuliah" vs "biaya penelitian")

---

### **Kasus 3: tatatertib_berkunjung_upt → FAQ (SALAH)**

**Analisis:**
- **True Label**: General
- **Predicted Label**: FAQ
- **Penyebab Error**:
  1. Tata tertib berisi banyak kalimat instruksi seperti "harus", "wajib", "tidak boleh" yang strukturnya mirip FAQ
  2. Hanya 1 dokumen kategori General dalam training set → model tidak cukup belajar pola General
  3. Class imbalance: FAQ dominan (5 docs) vs General (1 doc)

**Neighbors yang Mempengaruhi:**
- faq_krs (similarity: 0.74) - kata "mahasiswa", "aturan" overlap
- faq_udinus_profil (similarity: 0.70) - kata "kampus", "universitas" overlap
- profil_ilkom (similarity: 0.62) - kalah voting

**Solusi Potensial:**
- Gunakan SMOTE atau data augmentation untuk balance classes
- Implementasi weighted voting by class frequency (minority class gets higher weight)
- Tambahkan dokumen General lebih banyak (minimal 3-5 dokumen per kategori)

---

### **Kasus 4: kurikulum_teknik_informatika → FAQ (SALAH)**

**Analisis:**
- **True Label**: Profile/Academic
- **Predicted Label**: FAQ
- **Penyebab Error**:
  1. Kurikulum berisi list mata kuliah dengan format "Semester 1:", "Mata Kuliah:" yang setelah preprocessing jadi tokens biasa
  2. Tokens "semester", "mata kuliah", "sks" sangat sering muncul di FAQ akademik
  3. Feature selection berbasis variance menghilangkan terms spesifik kurikulum

**Neighbors yang Mempengaruhi:**
- faq_krs (similarity: 0.81) - overlap ekstrem di "semester", "kuliah", "mata"
- faq_daftar_udinus (similarity: 0.75) - kata "program", "teknik" overlap
- profil_teknik (similarity: 0.73) - kalah voting

**Solusi Potensial:**
- Preprocessing khusus untuk dokumen terstruktur (list, table)
- Tambahkan n-grams features (bigrams: "teknik informatika", "mata kuliah")
- Gunakan metadata (document length, structure patterns) sebagai fitur tambahan

---

### **Kasus 5: profil_fib → FAQ (SALAH)**

**Analisis:**
- **True Label**: Profile/Academic
- **Predicted Label**: FAQ
- **Penyebab Error**:
  1. FIB (Fakultas Ilmu Budaya) memiliki vocabulary unik ("bahasa", "sastra", "budaya") yang tidak ter-representasi di training set
  2. Setelah stopword removal dan stemming, dokumen profil FIB menjadi sangat sparse
  3. Cosine similarity gagal menangkap semantic meaning → fall back ke frequent common words

**Neighbors yang Mempengaruhi:**
- faq_udinus_profil (similarity: 0.68) - kata "fakultas", "program" overlap
- faq_daftar_udinus (similarity: 0.66) - kata "mahasiswa" overlap
- profil_ilkom (similarity: 0.64) - kalah voting

**Solusi Potensial:**
- Gunakan word embeddings (Word2Vec, FastText) untuk semantic similarity
- Domain-specific lexicon untuk weight important terms ("sastra" lebih penting dari "mahasiswa")
- Ensemble method: combine TF-IDF + semantic embeddings + metadata features

---

## Kesimpulan Error Analysis

### **Root Causes (Penyebab Utama)**

1. **Dataset Kecil** (12 dokumen):
   - Model tidak cukup belajar variasi dalam kategori
   - Overfitting ke patterns spesifik training data

2. **Class Imbalance**:
   - FAQ: 5 dokumen (dominan)
   - Profile: 6 dokumen (cukup)
   - General: 1 dokumen (sangat minor) ← masalah besar

3. **Feature Reduction Aggressive**:
   - Dari 265 vocab → 100 features
   - Kehilangan distinctive terms per kategori

4. **Vocabulary Overlap Tinggi**:
   - Common words ("mahasiswa", "kampus", "program") mendominasi similarity
   - Distinctive terms ter-filter oleh stopword removal

5. **K=3 Terlalu Kecil**:
   - Sangat sensitif terhadap noise dan outliers
   - Perlu testing k=5,7,9 untuk robustness

### **Rekomendasi Perbaikan**

| Prioritas | Action | Impact | Effort |
|-----------|--------|--------|--------|
| **HIGH** | Tambah data (min 20 docs per class) | ⭐⭐⭐⭐⭐ | Medium |
| **HIGH** | Tune k (test k=1,3,5,7,9) | ⭐⭐⭐⭐ | Low |
| **MEDIUM** | Gunakan Chi-square/IG feature selection | ⭐⭐⭐⭐ | Low |
| **MEDIUM** | Tambahkan bigrams/trigrams | ⭐⭐⭐ | Medium |
| **MEDIUM** | Class weighting untuk imbalance | ⭐⭐⭐⭐ | Low |
| **LOW** | Word embeddings (Word2Vec) | ⭐⭐⭐⭐⭐ | High |
| **LOW** | Ensemble methods | ⭐⭐⭐⭐ | High |

### **Lessons Learned**

1. **K-NN sangat sensitif terhadap**:
   - Kualitas features (preprocessing, selection)
   - Ukuran dataset (butuh representasi cukup per class)
   - Pilihan metric (cosine vs euclidean) dan k

2. **Corpus kecil butuh**:
   - Careful feature engineering (jangan aggressive reduction)
   - Domain knowledge (weighted features, manual lexicons)
   - Augmentation strategies

3. **Trade-off complexity vs performance**:
   - K-NN from scratch: simple, interpretable, tapi butuh data banyak
   - Deep learning: powerful tapi overkill untuk 12 dokumen
   - Hybrid approach: TF-IDF + domain rules mungkin optimal untuk case ini
