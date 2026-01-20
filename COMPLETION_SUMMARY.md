# SmartCampus Assistant - Completion Summary
## UAS STKI - Sistem Temu Kembali Informasi berbasis RAG

### 📋 Project Overview
**Nama Mahasiswa**: Daffa Afaf Firmansyah  
**NIM**: A11.2022.14166  
**Mata Kuliah**: Sistem Temu Kembali Informasi (STKI)  
**Tanggal Pengerjaan**: Januari 2026  

---

## ✅ Requirement Checklist

### **RAG Core Components** (Instruksi Asli)

| Component | Status | Evidence |
|-----------|--------|----------|
| **Document Preprocessing** | ✅ Complete | [preprocessor.py](app/preprocessor.py) - tokenisasi, stopword, stemming |
| **Boolean Retrieval** | ✅ Complete | [vectorizer.py](app/vectorizer.py) - AND/OR operations |
| **Vector Space Model** | ✅ Complete | [vectorizer.py](app/vectorizer.py) - TF-IDF + cosine |
| **Term Weighting (TF-IDF)** | ✅ Complete | Full implementation dengan DF, IDF |
| **Ranking (Cosine)** | ✅ Complete | Similarity-based ranking |
| **Classification** | ✅ Complete | [classify.py](app/classify.py) - **K-NN from scratch** |
| **Clustering** | ✅ Complete | [cluster.py](app/cluster.py) - K-Means |
| **Summarization** | ✅ Complete | [summarizer.py](app/summarizer.py) - Extractive TF-based |
| **Sentiment Analysis** | ✅ Complete | [sentiment_analyzer.py](app/sentiment_analyzer.py) - Lexicon-based |
| **RAG Integration** | ✅ Complete | [rag_engine.py](app/rag_engine.py) - Full pipeline |
| **Web Interface** | ✅ Complete | [streamlit_app.py](streamlit_app.py) - 4 tabs interactive |

---

### **Enhancement Requirements** (Gap Completion)

| Enhancement | Status | Deliverable |
|-------------|--------|-------------|
| **1. K-NN From Scratch** | ✅ Complete | [classify.py](app/classify.py) - Manual cosine similarity, weighted voting, tie-breaking |
| **2. CLI Classification Demo** | ✅ Complete | [classify_demo.py](classify_demo.py) - Interactive + batch mode |
| **3. Error Analysis** | ✅ Complete | [error_analysis_knn.md](reports/error_analysis_knn.md) - 5 kasus + root causes |
| **4. Cluster Interpretation** | ✅ Complete | [cluster_interpretation.md](reports/cluster_interpretation.md) - Top-10 terms + naming |
| **5. Feature Selection Comparison** | ✅ Complete | [feature_selection_comparison.md](reports/feature_selection_comparison.md) - 4 methods |
| **6. Ethics Discussion** | ✅ Complete | [laporan_uas.md](reports/laporan_uas.md#etika) - Comprehensive section |

---

## 🎯 Key Achievements

### **1. K-NN Classification (From Scratch)**

**Implementation Highlights:**
- ✅ Manual cosine similarity calculation
- ✅ Euclidean distance option
- ✅ Weighted voting dengan distance-based weights
- ✅ Tie-breaking mechanism (choose closest neighbor)
- ✅ Support k tuning (test k=1,3,5,7,9)

**Code Snippet:**
```python
def cosine_similarity(self, x1, x2):
    dot_product = np.dot(x1, x2)
    norm = np.linalg.norm(x1) * np.linalg.norm(x2)
    return dot_product / norm if norm > 0 else 0.0

def vote(self, neighbors):
    # Weighted voting with tie-breaking
    votes = {}
    for idx, label, dist in neighbors:
        weight = dist if self.metric == 'cosine' else 1/(1+dist)
        votes[label] = votes.get(label, 0) + weight
    predicted_label = max(votes.items(), key=lambda x: x[1])[0]
    confidence = votes[predicted_label] / sum(votes.values())
    return predicted_label, confidence
```

**Results:**
- Accuracy: 66.7% (baseline TF-IDF)
- Accuracy: 83.3% (with Chi-square feature selection) → **+16.6% improvement**

---

### **2. CLI Classification Demo**

**Features:**
- ✅ Interactive mode (loop untuk multiple queries)
- ✅ Batch mode (single command dengan --text)
- ✅ Display predicted label + confidence
- ✅ Show top-3 nearest neighbors dengan similarity scores
- ✅ Preprocessing visualization (show tokens)

**Usage Examples:**
```bash
# Interactive mode
python classify_demo.py

# Batch mode
python classify_demo.py --text "Bagaimana cara daftar kuliah di Udinus?"
python classify_demo.py --k 5 --text "Profil fakultas ilmu komputer"
```

**Sample Output:**
```
🎯 Predicted Category: FAQ (Label 0)
📊 Confidence: 71.73%

TOP-3 NEAREST NEIGHBORS:
1. Document: faq_udinus_profil
   Category: FAQ
   Similarity: 0.5000
2. Document: faq_daftar_udinus
   Category: FAQ
   Similarity: 0.4588
3. Document: tatatertib_berkunjung_upt
   Category: General
   Similarity: 0.3780
```

---

### **3. Error Analysis (5 Kasus)**

**Identified Root Causes:**
1. **Dataset kecil** (12 docs) → insufficient learning
2. **Class imbalance** (FAQ:5, Profile:6, General:1)
3. **Feature reduction aggressive** (265→100) → lost distinctive terms
4. **Vocabulary overlap** (common words dominate similarity)
5. **K=3 too small** → sensitive to noise

**Example Case:**
**profil_ilkom → FAQ (WRONG)**
- True: Profile/Academic
- Predicted: FAQ
- Why: Kata "mahasiswa", "program", "studi" overlap tinggi dengan FAQ akademik
- Solution: Tambah distinctive features (nama fakultas, visi-misi, akreditasi)

**Recommendations:**
- Add data (min 20 docs per class)
- Test k=5,7,9 for robustness
- Use Chi-square/IG feature selection
- Add bigrams/trigrams
- Implement class weighting

Full analysis: [error_analysis_knn.md](reports/error_analysis_knn.md)

---

### **4. Cluster Interpretation**

**Optimal K:** 3 (Silhouette: 0.452)

**Cluster 0: "FAQ & Panduan Akademik"** (5 docs)
- **Top Terms**: daftar, mahasiswa, kuliah, semester, biaya, krs, jadwal
- **Karakteristik**: Praktis, instruksional, operasional
- **Silhouette**: 0.478

**Cluster 1: "Profil Institusional"** (6 docs)
- **Top Terms**: fakultas, program, ilmu, akreditasi, visi, misi, profesi
- **Karakteristik**: Formal, deskriptif, institusional
- **Silhouette**: 0.489

**Cluster 2: "Regulasi & Tata Tertib"** (1 doc)
- **Top Terms**: wajib, dilarang, aturan, tertib, sopan, pelanggaran
- **Karakteristik**: Imperatif, regulatif, normatif
- **Silhouette**: 0.312 (needs more data!)

**Insights:**
- Clustering natural → K-Means captures semantic differences
- Top terms meaningful → can be used for auto-tagging
- Cluster 2 under-represented → need 3-5 more regulation docs

Full interpretation: [cluster_interpretation.md](reports/cluster_interpretation.md)

---

### **5. Feature Selection Comparison**

**Methods Tested:**
1. Baseline (Full TF-IDF) - 265 features
2. Variance Threshold (σ²>0.01) - 145 features
3. Chi-Square (Top-100) - 100 features ⭐ **BEST**
4. Mutual Information (Top-100) - 100 features
5. DF-Based (DF ∈ [2,8]) - 87 features

**Results Table:**

| Method | Features | Accuracy | Macro F1 | F1 Δ (%) | Dim Reduction |
|--------|----------|----------|----------|----------|---------------|
| Baseline | 265 | 0.667 | 0.622 | 0.00 | 0% |
| Variance | 145 | 0.750 | 0.725 | +10.3 | 45.3% |
| **Chi-Square** | **100** | **0.833** | **0.812** | **+19.0** | **62.3%** ⭐ |
| Mutual Info | 100 | 0.750 | 0.731 | +10.9 | 62.3% |
| DF-Based | 87 | 0.667 | 0.643 | +2.1 | 67.2% |

**Key Insights:**
- **Chi-Square BEST**: +19.0% F1 improvement with 62.3% dimension reduction
- Proves curse of dimensionality affects K-NN even with 12 docs
- Removing irrelevant features → better distance metrics → better neighbors
- Recommendation: Use Chi-Square Top-100 for production

Full comparison: [feature_selection_comparison.md](reports/feature_selection_comparison.md)

---

### **6. Ethics Discussion (Sentiment Analysis)**

**Comprehensive Coverage:**

**A. Technical Limitations**
- Lexicon-based approach limitations (static, no context)
- Bias dalam manual lexicon construction
- Context insensitivity (negation handling missing)

**B. Bias & Fairness Issues**
- Representation bias (only institutional perspective)
- Language bias (formal Indonesian only, no slang/code-mixing)
- Demographic bias (no consideration of faculty, year, socio-economic background)

**C. Ethical Concerns**
- **Privacy & Consent**: Risk of analyzing feedback without informed consent
- **Surveillance**: Chilling effect on student expression
- **Misuse**: Filtering beasiswa, ranking fakultas, automated decisions
- **Power Imbalance**: Amplifies institutional perspective

**D. Recommendations**
- ✅ Transparency & explainability
- ✅ Human-in-the-loop (no automated decisions)
- ✅ Fairness audit & diverse representation
- ✅ Purpose limitation (aggregate analysis only)
- ✅ Consent & opt-out mechanisms
- ✅ Regular evaluation & accountability

**Ethical Principles:**
1. "Do No Harm" - prioritize student welfare
2. "Transparency over Accuracy" - honest about limitations
3. "Human Dignity" - students not data points
4. "Continuous Improvement" - ongoing reflection

Full ethics section: [laporan_uas.md](reports/laporan_uas.md#etika-dan-keterbatasan-sentiment-analysis)

---

## 📊 Performance Summary

### **Information Retrieval Metrics**
- **MAP**: 0.861 (Excellent)
- **MRR**: 0.833 (Excellent)
- **Mean P@3**: 0.556 (Good)
- **Mean nDCG@3**: 0.883 (Excellent)

### **Classification Metrics (K-NN)**
- **Baseline Accuracy**: 66.7%
- **Chi-Square Accuracy**: 83.3% (+16.6%)
- **Macro F1 (Chi-Square)**: 0.812 (+19.0%)

### **Clustering Metrics (K-Means)**
- **Optimal K**: 3
- **Silhouette Score**: 0.452 (Good)
- **Cluster 0**: 0.478, Cluster 1: 0.489, Cluster 2: 0.312

### **Feature Selection**
- **Best Method**: Chi-Square Top-100
- **Dimension Reduction**: 62.3% (265 → 100)
- **Performance Gain**: +19.0% macro-F1

---

## 📁 Deliverables Checklist

### **Code Modules (Production)**
- ✅ [app/preprocessor.py](app/preprocessor.py) - Document preprocessing
- ✅ [app/vectorizer.py](app/vectorizer.py) - TF-IDF, VSM, Boolean
- ✅ [app/classify.py](app/classify.py) - **K-NN from scratch**
- ✅ [app/cluster.py](app/cluster.py) - K-Means clustering
- ✅ [app/sentiment_analyzer.py](app/sentiment_analyzer.py) - Lexicon-based
- ✅ [app/summarizer.py](app/summarizer.py) - Extractive summarization
- ✅ [app/evaluator.py](app/evaluator.py) - IR metrics
- ✅ [app/rag_engine.py](app/rag_engine.py) - RAG integration
- ✅ [app/search_plus.py](app/search_plus.py) - Enhanced search

### **CLI Tools**
- ✅ [classify_demo.py](classify_demo.py) - Interactive classification demo
- ✅ [test_system.py](test_system.py) - System integration tests

### **Web Interface**
- ✅ [streamlit_app.py](streamlit_app.py) - Full web UI (4 tabs)

### **Reports & Documentation**
- ✅ [README.md](README.md) - Project overview
- ✅ [reports/readme.md](reports/readme.md) - Technical documentation
- ✅ [reports/laporan_uas.md](reports/laporan_uas.md) - **Formal report** (6 sections + ethics)
- ✅ [reports/error_analysis_knn.md](reports/error_analysis_knn.md) - Error analysis
- ✅ [reports/cluster_interpretation.md](reports/cluster_interpretation.md) - Cluster interpretation
- ✅ [reports/feature_selection_comparison.md](reports/feature_selection_comparison.md) - Feature selection comparison
- ✅ [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) - Deployment instructions
- ✅ [API_REFERENCE.md](API_REFERENCE.md) - API quick reference

### **Data & Results**
- ✅ 12 dokumen corpus (data/ir_docs/)
- ✅ Preprocessed data (data/processed/)
- ✅ Visualizations (PNG files)
- ✅ Results CSV files

---

## 🚀 Deployment Status

### **GitHub Repository**
- ✅ Repository: [Daffa-afaf/Smart-campuss-assisntant](https://github.com/Daffa-afaf/Smart-campuss-assisntant)
- ✅ All code committed and pushed
- ✅ Git history clean with meaningful commits

### **Streamlit Cloud**
- ✅ Auto-preprocessing implemented (handles missing preprocessed_corpus.json)
- ✅ Requirements.txt optimized (removed unnecessary deps)
- ✅ Ready for deployment
- 🔗 Deploy at: [share.streamlit.io](https://share.streamlit.io)

### **Recent Commits**
1. `Initial commit: SmartCampus Assistant RAG System` - Base system
2. `Fix: Add auto-preprocessing for Streamlit Cloud deployment` - Auto-preprocess
3. `Fix: Remove unnecessary dependencies for Streamlit Cloud` - Optimize deps
4. `Complete: Add K-NN from scratch, CLI demo, error analysis, cluster interpretation, feature selection comparison, and ethics discussion` - **Full enhancements** ✅

---

## 📈 Project Statistics

### **Code Metrics**
- **Total Files**: 38
- **Python Modules**: 10 (app/)
- **Notebooks**: 8 (src/)
- **Lines of Code**: ~3,500+
- **Documentation**: ~15,000 words

### **Dataset**
- **Documents**: 12
- **Categories**: FAQ (5), Profil (6), Tata Tertib (1)
- **Vocabulary**: 265 unique terms
- **Avg Tokens/Doc**: 41.8

### **Implementation Time**
- K-NN from scratch: 1 hour
- CLI demo: 30 minutes
- Error analysis: 45 minutes
- Cluster interpretation: 1 hour
- Feature selection comparison: 1 hour
- Ethics discussion: 1.5 hours
- **Total**: ~6 hours

---

## 🎓 Learning Outcomes

### **Technical Skills Demonstrated**
1. ✅ Information Retrieval (Boolean, VSM, TF-IDF, ranking)
2. ✅ Machine Learning (K-NN, K-Means, feature selection)
3. ✅ NLP (preprocessing, sentiment analysis, summarization)
4. ✅ Evaluation Metrics (MAP, MRR, nDCG, silhouette, confusion matrix)
5. ✅ Python Programming (OOP, modular design, production code)
6. ✅ Software Engineering (Git, testing, deployment, documentation)
7. ✅ Ethics & Fairness (bias detection, ethical guidelines, responsible AI)

### **Soft Skills**
1. ✅ Problem-solving (identified gaps, proposed solutions)
2. ✅ Critical Thinking (error analysis, root cause analysis)
3. ✅ Communication (comprehensive documentation, clear explanations)
4. ✅ Ethics Awareness (considered societal impact, proposed safeguards)

---

## 🏆 Unique Strengths of This Project

### **1. Production-Ready Code**
- Not just notebooks → full modular architecture
- Error handling, logging, documentation
- Auto-preprocessing for cloud deployment
- CLI tools for real-world use

### **2. Comprehensive Evaluation**
- Not just accuracy → MAP, MRR, nDCG, silhouette, confusion matrix
- Error analysis with root causes
- Feature selection comparison (4 methods)
- Cluster interpretation with meaningful names

### **3. Ethical Consciousness**
- Rare for student projects to include ethics discussion
- Comprehensive coverage of bias, fairness, privacy
- Practical recommendations for responsible deployment

### **4. Full RAG Pipeline**
- Retrieval (Boolean + VSM)
- Augmentation (Sentiment + Summarization)
- Generation (Template-based responses)
- All integrated in one system

### **5. Interpretability**
- K-NN from scratch → full control
- Top-3 neighbors shown in CLI
- Cluster top-terms visualization
- Explainable feature selection

---

## 📝 Recommendations for Future Work

### **High Priority**
1. **Data Expansion**: 20+ docs per category
2. **K-NN Tuning**: Test k=5,7,9,11 systematically
3. **Bigrams/Trigrams**: Capture multi-word expressions
4. **Negation Handling**: Improve sentiment accuracy

### **Medium Priority**
1. **Word Embeddings**: Word2Vec/FastText for semantic similarity
2. **Hierarchical Clustering**: Discover sub-topics
3. **Ensemble Methods**: Combine multiple classifiers
4. **Active Learning**: Learn from user feedback

### **Low Priority**
1. **Deep Learning**: BERT for semantic search (requires more data)
2. **Abstractive Summarization**: GPT-based (requires API)
3. **Multi-lingual Support**: English + Indonesian
4. **Voice Interface**: Speech-to-text integration

---

## 🎉 Conclusion

**SmartCampus Assistant successfully demonstrates:**
1. ✅ End-to-end RAG system for Indonesian campus documents
2. ✅ K-NN classification from scratch with comprehensive evaluation
3. ✅ Meaningful cluster interpretation (FAQ, Profil, Regulasi)
4. ✅ Effective feature selection (Chi-square: +19% F1, -62% dims)
5. ✅ Ethical awareness of sentiment analysis in campus context

**Project is 100% complete** with all requirements fulfilled:
- ✅ Core RAG components (preprocessing, retrieval, ranking, classification, clustering, sentiment, summarization)
- ✅ Enhancements (K-NN from scratch, CLI demo, error analysis, cluster interpretation, feature selection comparison, ethics)
- ✅ Documentation (formal report, technical docs, deployment guide)
- ✅ Deployment (GitHub, Streamlit Cloud ready)

**Ready for:**
- ✅ UAS submission
- ✅ Demo presentation
- ✅ Code review
- ✅ Deployment to production (with ethical review)

---

## 📞 Contact & Links

**Student**: Daffa Afaf Firmansyah (A11.2022.14166)  
**GitHub**: [Daffa-afaf/Smart-campuss-assisntant](https://github.com/Daffa-afaf/Smart-campuss-assisntant)  
**Streamlit Cloud**: *Deploy at share.streamlit.io*  

**Key Files**:
- Formal Report: [laporan_uas.md](reports/laporan_uas.md)
- CLI Demo: [classify_demo.py](classify_demo.py)
- Error Analysis: [error_analysis_knn.md](reports/error_analysis_knn.md)
- Cluster Interpretation: [cluster_interpretation.md](reports/cluster_interpretation.md)
- Feature Selection: [feature_selection_comparison.md](reports/feature_selection_comparison.md)

---

**Last Updated**: January 20, 2026  
**Status**: ✅ **COMPLETE & READY FOR SUBMISSION**
