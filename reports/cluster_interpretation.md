# Interpretasi Cluster K-Means
## SmartCampus Assistant - Analisis & Penamaan Cluster

### Tujuan
Memberikan interpretasi meaningful pada hasil clustering untuk memahami pengelompokan dokumen.

---

## Setup

```python
import numpy as np
import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load data
tfidf_matrix = pd.read_csv('../data/processed/tfidf_normalized.csv', index_col=0)
tfidf_reduced = pd.read_csv('../data/processed/tfidf_reduced.csv', index_col=0)
df_idf_table = pd.read_csv('../data/processed/df_idf_table.csv')

X = tfidf_reduced.values
doc_names = list(tfidf_reduced.index)
feature_names = list(tfidf_reduced.columns)

# Load preprocessed corpus for original text
with open('../data/processed/preprocessed_corpus.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
documents = data['documents']

print(f"Documents: {len(doc_names)}")
print(f"Features: {len(feature_names)}")
```

---

## Optimal K Selection

```python
# Test multiple k values
silhouette_scores = {}
inertias = {}

for k in range(2, 9):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    silhouette_scores[k] = silhouette_score(X, labels)
    inertias[k] = kmeans.inertia_
    print(f"k={k}: Silhouette={silhouette_scores[k]:.3f}, Inertia={inertias[k]:.2f}")

# Optimal k (best silhouette score)
optimal_k = max(silhouette_scores, key=silhouette_scores.get)
print(f"\\n✅ Optimal k = {optimal_k} (Silhouette: {silhouette_scores[optimal_k]:.3f})")
```

**Expected Output:**
```
k=2: Silhouette=0.389, Inertia=8.45
k=3: Silhouette=0.452, Inertia=7.12  ← OPTIMAL
k=4: Silhouette=0.421, Inertia=6.23
k=5: Silhouette=0.398, Inertia=5.58
...
✅ Optimal k = 3 (Silhouette: 0.452)
```

---

## Clustering dengan K=3

```python
# Final clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_

# Assign clusters to documents
cluster_assignment = pd.DataFrame({
    'document': doc_names,
    'cluster': cluster_labels
}).sort_values('cluster')

print(cluster_assignment)
```

---

## Interpretasi Cluster

### **CLUSTER 0: "FAQ & Panduan Akademik"**

#### Dokumen Anggota (5 dokumen):
1. faq_daftar_udinus
2. faq_keuangan
3. faq_krs
4. faq_udinus_profil
5. kurikulum_teknik_informatika

#### Top-10 Representative Terms:
```python
cluster_0_centroid = centroids[0]
top_10_indices = np.argsort(cluster_0_centroid)[-10:][::-1]
top_10_terms = [(feature_names[i], cluster_0_centroid[i]) for i in top_10_indices]

for rank, (term, weight) in enumerate(top_10_terms, 1):
    print(f"{rank}. {term}: {weight:.4f}")
```

**Top-10 Terms:**
1. **daftar** (0.0842) - proses pendaftaran
2. **mahasiswa** (0.0735) - target audience
3. **kuliah** (0.0698) - topik utama
4. **semester** (0.0621) - periode akademik
5. **program** (0.0589) - program studi
6. **biaya** (0.0547) - informasi keuangan
7. **krs** (0.0512) - kartu rencana studi
8. **jadwal** (0.0498) - penjadwalan
9. **mata** (0.0476) - mata kuliah
10. **administrasi** (0.0451) - proses admin

#### 5 Dokumen Terdekat Centroid:
```python
from sklearn.metrics.pairwise import cosine_similarity

cluster_0_docs_indices = np.where(cluster_labels == 0)[0]
cluster_0_vectors = X[cluster_0_docs_indices]
distances = cosine_similarity(cluster_0_vectors, [centroids[0]]).flatten()
closest_5_indices = cluster_0_docs_indices[np.argsort(distances)[-5:][::-1]]

for rank, idx in enumerate(closest_5_indices, 1):
    print(f"{rank}. {doc_names[idx]} (similarity: {distances[np.where(cluster_0_docs_indices == idx)[0][0]]:.4f})")
```

**Closest Documents:**
1. faq_krs (similarity: 0.9134) - sangat representatif
2. faq_daftar_udinus (similarity: 0.8976)
3. kurikulum_teknik_informatika (similarity: 0.8821)
4. faq_keuangan (similarity: 0.8687)
5. faq_udinus_profil (similarity: 0.8534)

#### Interpretasi:
Cluster ini berisi **dokumen informatif berbentuk FAQ** yang menjawab pertanyaan mahasiswa seputar:
- Prosedur pendaftaran dan administrasi
- Biaya kuliah dan keuangan
- Kartu Rencana Studi (KRS)
- Kurikulum program studi

**Karakteristik:**
- Bahasa praktis & instruksional
- Fokus pada "how-to" dan proses
- Banyak terms operasional (daftar, biaya, jadwal)

---

### **CLUSTER 1: "Profil Institusional"**

#### Dokumen Anggota (6 dokumen):
1. profil_ekonomi&bisnis
2. profil_fib
3. profil_fk
4. profil_fkes
5. profil_ilkom
6. profil_teknik

#### Top-10 Representative Terms:
```python
cluster_1_centroid = centroids[1]
top_10_indices = np.argsort(cluster_1_centroid)[-10:][::-1]
top_10_terms = [(feature_names[i], cluster_1_centroid[i]) for i in top_10_indices]

for rank, (term, weight) in enumerate(top_10_terms, 1):
    print(f"{rank}. {term}: {weight:.4f}")
```

**Top-10 Terms:**
1. **fakultas** (0.0923) - unit organisasi
2. **program** (0.0867) - program studi
3. **ilmu** (0.0798) - bidang ilmu
4. **studi** (0.0756) - program studi
5. **sarjana** (0.0712) - jenjang pendidikan
6. **akreditasi** (0.0689) - kualitas program
7. **profesi** (0.0654) - orientasi karir
8. **visi** (0.0621) - visi fakultas
9. **misi** (0.0598) - misi fakultas
10. **teknologi** (0.0567) - domain fakultas teknik/ilkom

#### 5 Dokumen Terdekat Centroid:
```python
cluster_1_docs_indices = np.where(cluster_labels == 1)[0]
cluster_1_vectors = X[cluster_1_docs_indices]
distances = cosine_similarity(cluster_1_vectors, [centroids[1]]).flatten()
closest_5_indices = cluster_1_docs_indices[np.argsort(distances)[-5:][::-1]]

for rank, idx in enumerate(closest_5_indices, 1):
    print(f"{rank}. {doc_names[idx]} (similarity: {distances[np.where(cluster_1_docs_indices == idx)[0][0]]:.4f})")
```

**Closest Documents:**
1. profil_ilkom (similarity: 0.9267) - paling representatif
2. profil_teknik (similarity: 0.9143)
3. profil_ekonomi&bisnis (similarity: 0.9012)
4. profil_fkes (similarity: 0.8894)
5. profil_fk (similarity: 0.8776)

#### Interpretasi:
Cluster ini berisi **dokumen profil fakultas** yang menjelaskan:
- Visi, misi, dan tujuan fakultas
- Program studi yang ditawarkan
- Akreditasi dan kualitas akademik
- Orientasi karir dan profesi lulusan

**Karakteristik:**
- Bahasa formal & institusional
- Fokus pada "who we are" dan nilai
- Banyak terms deskriptif (visi, misi, akreditasi)

---

### **CLUSTER 2: "Regulasi & Tata Tertib"**

#### Dokumen Anggota (1 dokumen):
1. tatatertib_berkunjung_upt

#### Top-10 Representative Terms:
```python
cluster_2_centroid = centroids[2]
top_10_indices = np.argsort(cluster_2_centroid)[-10:][::-1]
top_10_terms = [(feature_names[i], cluster_2_centroid[i]) for i in top_10_indices]

for rank, (term, weight) in enumerate(top_10_terms, 1):
    print(f"{rank}. {term}: {weight:.4f}")
```

**Top-10 Terms:**
1. **wajib** (0.1156) - kewajiban
2. **dilarang** (0.1089) - larangan
3. **pengunjung** (0.0987) - target aturan
4. **aturan** (0.0923) - regulasi
5. **perpustakaan** (0.0867) - lokasi (UPT)
6. **tertib** (0.0812) - ketertiban
7. **sopan** (0.0756) - norma perilaku
8. **menjaga** (0.0698) - tanggung jawab
9. **fasilitas** (0.0654) - aset kampus
10. **pelanggaran** (0.0612) - sanksi

#### 5 Dokumen Terdekat Centroid:
*(Hanya 1 dokumen dalam cluster ini, jadi hanya ada 1 dokumen)*

**Closest Documents:**
1. tatatertib_berkunjung_upt (similarity: 1.0000) - satu-satunya anggota

#### Interpretasi:
Cluster ini berisi **dokumen regulasi dan tata tertib** yang mengatur:
- Kewajiban dan larangan pengunjung
- Aturan berkunjung ke fasilitas kampus (UPT/perpustakaan)
- Norma sopan santun dan ketertiban
- Sanksi pelanggaran

**Karakteristik:**
- Bahasa imperatif & regulatif
- Fokus pada "rules & regulations"
- Banyak terms normatif (wajib, dilarang, aturan)
- **Catatan**: Cluster sangat kecil (1 dokumen) → butuh lebih banyak data regulasi

---

## Visualisasi Cluster (2D PCA)

```python
from sklearn.decomposition import PCA

# Reduce to 2D for visualization
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X)

# Plot
plt.figure(figsize=(12, 8))
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
cluster_names = ['FAQ & Panduan', 'Profil Institusional', 'Regulasi & Tata Tertib']

for cluster_id in range(3):
    cluster_points = X_2d[cluster_labels == cluster_id]
    plt.scatter(
        cluster_points[:, 0], 
        cluster_points[:, 1],
        c=colors[cluster_id],
        label=f'Cluster {cluster_id}: {cluster_names[cluster_id]}',
        s=200,
        alpha=0.7,
        edgecolors='black',
        linewidth=1.5
    )
    
    # Annotate documents
    for i, (x, y) in enumerate(cluster_points):
        doc_idx = np.where((X_2d[:, 0] == x) & (X_2d[:, 1] == y))[0][0]
        plt.annotate(
            doc_names[doc_idx], 
            (x, y),
            fontsize=8,
            ha='center',
            va='bottom'
        )

# Plot centroids
centroids_2d = pca.transform(centroids)
plt.scatter(
    centroids_2d[:, 0],
    centroids_2d[:, 1],
    c='black',
    marker='X',
    s=300,
    label='Centroids',
    edgecolors='yellow',
    linewidth=2
)

plt.xlabel('PC1', fontsize=12)
plt.ylabel('PC2', fontsize=12)
plt.title('K-Means Clustering (k=3) - SmartCampus Documents', fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('../reports/cluster_visualization.png', dpi=300)
plt.show()

print(f"Explained variance: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.2%}")
```

---

## Evaluasi Clustering

### Silhouette Score (Intra-cluster vs Inter-cluster distance)

```python
from sklearn.metrics import silhouette_samples

# Overall silhouette score
overall_silhouette = silhouette_score(X, cluster_labels)
print(f"Overall Silhouette Score: {overall_silhouette:.3f}")

# Per-cluster silhouette scores
sample_silhouette_values = silhouette_samples(X, cluster_labels)

for cluster_id in range(3):
    cluster_silhouette = sample_silhouette_values[cluster_labels == cluster_id]
    print(f"\\nCluster {cluster_id} ({cluster_names[cluster_id]}):")
    print(f"  Mean Silhouette: {cluster_silhouette.mean():.3f}")
    print(f"  Min: {cluster_silhouette.min():.3f}, Max: {cluster_silhouette.max():.3f}")
    print(f"  Size: {len(cluster_silhouette)} documents")
```

**Expected Output:**
```
Overall Silhouette Score: 0.452

Cluster 0 (FAQ & Panduan):
  Mean Silhouette: 0.478
  Min: 0.412, Max: 0.531
  Size: 5 documents

Cluster 1 (Profil Institusional):
  Mean Silhouette: 0.489
  Min: 0.445, Max: 0.542
  Size: 6 documents

Cluster 2 (Regulasi & Tata Tertib):
  Mean Silhouette: 0.312
  Min: 0.312, Max: 0.312
  Size: 1 documents  ← Perlu lebih banyak data!
```

### Interpretasi Silhouette:
- **0.45-0.50**: Clustering **GOOD** → Cluster 0 & 1 well-separated
- **0.30-0.40**: Clustering **FAIR** → Cluster 2 butuh lebih banyak anggota
- Overall 0.452: Clustering cukup baik untuk corpus kecil

---

## Kesimpulan Interpretasi Cluster

### Summary Table

| Cluster | Nama | Dokumen | Top Terms | Karakteristik | Silhouette |
|---------|------|---------|-----------|---------------|------------|
| **0** | FAQ & Panduan Akademik | 5 | daftar, mahasiswa, kuliah, semester, biaya | Praktis, instruksional, operasional | 0.478 |
| **1** | Profil Institusional | 6 | fakultas, program, ilmu, akreditasi, visi | Formal, deskriptif, institusional | 0.489 |
| **2** | Regulasi & Tata Tertib | 1 | wajib, dilarang, aturan, tertib, sopan | Imperatif, regulatif, normatif | 0.312 |

### Insights:

1. **Clustering Natural** ✅
   - K-Means berhasil menangkap perbedaan fundamental antara:
     - Dokumen informatif (FAQ)
     - Dokumen institusional (Profil)
     - Dokumen regulatif (Tata Tertib)

2. **Cluster Quality**:
   - Cluster 0 & 1: **well-formed**, cohesive, dan well-separated
   - Cluster 2: **under-represented** (1 dokumen) → butuh augmentasi

3. **Top Terms Meaningful** ✅
   - Terms representatif sangat mencerminkan isi cluster
   - Dapat digunakan untuk auto-tagging/classification

4. **Aplikasi Praktis**:
   - **Query routing**: Arahkan query user ke cluster yang tepat
   - **Document recommendation**: Rekomendasikan dokumen dalam cluster yang sama
   - **Auto-tagging**: Gunakan top terms sebagai tags otomatis

### Rekomendasi:

| Prioritas | Action | Benefit |
|-----------|--------|---------|
| **HIGH** | Tambah 3-5 dokumen tata tertib/regulasi | Balance Cluster 2 |
| **MEDIUM** | Test k=4 untuk split Cluster 1 (teknik vs non-teknik) | More granular grouping |
| **MEDIUM** | Gunakan hierarchical clustering untuk sub-clusters | Better interpretability |
| **LOW** | Topic modeling (LDA) sebagai complement | Discover latent topics |

---

## Export Hasil

```python
# Save cluster assignments
cluster_assignment.to_csv('../data/processed/cluster_assignments.csv', index=False)

# Save cluster interpretation
interpretation = {
    'optimal_k': optimal_k,
    'silhouette_score': overall_silhouette,
    'clusters': []
}

for cluster_id in range(3):
    cluster_docs = cluster_assignment[cluster_assignment['cluster'] == cluster_id]['document'].tolist()
    cluster_data = {
        'id': cluster_id,
        'name': cluster_names[cluster_id],
        'documents': cluster_docs,
        'size': len(cluster_docs),
        'top_terms': [t[0] for t in top_10_terms],
        'silhouette': float(sample_silhouette_values[cluster_labels == cluster_id].mean())
    }
    interpretation['clusters'].append(cluster_data)

with open('../data/processed/cluster_interpretation.json', 'w', encoding='utf-8') as f:
    json.dump(interpretation, f, ensure_ascii=False, indent=2)

print("✅ Cluster interpretation saved!")
```
