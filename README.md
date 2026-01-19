# SmartCampus Assistant - RAG System 🎓

Sistem Retrieval-Augmented Generation (RAG) untuk SmartCampus Assistant menggunakan TF-IDF, sentiment analysis, dan summarization untuk pencarian dokumen cerdas.

## 📋 Features

- **Information Retrieval**: TF-IDF Vector Space Model & Boolean Retrieval
- **Sentiment Analysis**: Lexicon-based Indonesian sentiment detection
- **Text Summarization**: Extractive & query-focused summarization
- **Document Clustering**: K-Means clustering untuk grouping dokumen
- **Query Classification**: KNN classifier untuk kategorisasi query
- **IR Metrics**: Precision@k, Recall@k, MAP, MRR, nDCG
- **Web Interface**: Interactive Streamlit application

## 🏗️ Project Structure

```
stki-uas-A11.2022.14166-DaffaAfafFirmansyah/
├── app/                          # Production modules
│   ├── __init__.py              # Package initialization
│   ├── preprocessor.py          # Text preprocessing
│   ├── vectorizer.py            # TF-IDF & VSM
│   ├── sentiment_analyzer.py   # Sentiment analysis
│   ├── summarizer.py            # Text summarization
│   ├── classify.py              # KNN classifier
│   ├── cluster.py               # K-Means clustering
│   ├── evaluator.py             # IR metrics
│   ├── rag_engine.py            # Main RAG engine
│   └── search_plus.py           # Enhanced search
├── data/
│   ├── ir_docs/                 # Document corpus (12 documents)
│   └── processed/               # Preprocessed data & models
├── src/                          # Jupyter notebooks (research)
│   ├── preprocess.ipynb
│   ├── vectorize.ipynb
│   ├── sentiment.ipynb
│   ├── summarizer.ipynb
│   ├── feature_selection.ipynb
│   ├── kmeans_cluster.ipynb
│   ├── knn_clasifier.ipynb
│   └── eval_metrics.ipynb
├── streamlit_app.py             # Web interface
├── requirements.txt             # Python dependencies
└── README.md

```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <your-repo-url>
cd stki-uas-A11.2022.14166-DaffaAfafFirmansyah

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Notebooks (Optional - Data Already Preprocessed)

Execute notebooks in order:
```bash
1. src/preprocess.ipynb          # Preprocessing
2. src/vectorize.ipynb           # TF-IDF vectorization
3. src/feature_selection.ipynb  # Feature selection
4. src/kmeans_cluster.ipynb     # Clustering
5. src/knn_clasifier.ipynb      # Classification
6. src/sentiment.ipynb          # Sentiment analysis
7. src/summarizer.ipynb         # Summarization
8. src/eval_metrics.ipynb       # Evaluation
```

### 3. Run Streamlit App

```bash
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`

## 💻 Usage

### Using RAG Engine (Python)

```python
from app.rag_engine import RAGEngine

# Initialize
rag = RAGEngine()
rag.initialize()

# Search
results = rag.search("jadwal pendaftaran mahasiswa", top_k=5)

# Get document info
doc_info = rag.get_document_info("faq_keuangan")

# Summarize
summary = rag.get_document_summary("profil_ilkom", ratio=0.3)

# Analyze sentiment
sentiment = rag.analyze_sentiment("bagus dan berkualitas")
```

### Using SmartCampusSearch (Simplified)

```python
from app.search_plus import SmartCampusSearch

search = SmartCampusSearch()

# Semantic search
results = search.search("beasiswa", mode='semantic', top_k=5)

# Boolean search
results = search.search("mahasiswa pendaftaran", mode='boolean_and')

# Ask questions
answer = search.ask("Bagaimana cara mendaftar beasiswa?")
```

## 📊 Performance Metrics

Evaluation results on test queries:

| Metric | Score |
|--------|-------|
| MAP    | 0.861 |
| MRR    | 0.833 |
| P@3    | 0.556 |
| nDCG@3 | 0.883 |

**Interpretation**: Excellent retrieval performance with high precision and ranking quality.

## 🎯 API Reference

### RAGEngine

Main integration engine combining all RAG components.

**Methods:**
- `initialize()` - Load and initialize all components
- `search(query, top_k, use_sentiment, return_summaries)` - Search documents
- `boolean_search(query, mode)` - Boolean retrieval
- `get_document_summary(doc_name, ratio)` - Get document summary
- `analyze_sentiment(text)` - Analyze text sentiment
- `evaluate(test_queries, ground_truth)` - Evaluate retrieval quality

### SmartCampusSearch

Simplified search interface.

**Methods:**
- `search(query, mode, top_k, include_sentiment, include_summary)` - Enhanced search
- `ask(question, context_size)` - Question answering
- `get_document(doc_name)` - Get document information
- `analyze_query(query)` - Analyze query characteristics
- `compare_documents(doc1, doc2)` - Compare two documents
- `get_statistics()` - Get corpus statistics

## 🌐 Deployment

### Deploy to Streamlit Cloud

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Set main file: `streamlit_app.py`
5. Deploy!

### Deploy to Heroku

```bash
# Create Procfile
echo "web: streamlit run streamlit_app.py --server.port $PORT" > Procfile

# Deploy
heroku create your-app-name
git push heroku main
```

## 📚 Document Corpus

12 documents covering:
- FAQ: Pendaftaran, Keuangan, KRS, Profil Udinus
- Profil: Fakultas (Ilkom, Teknik, Ekonomi, FIB, FK, FKes)
- Kurikulum: Teknik Informatika
- Tata Tertib: Perpustakaan UPT

## 🔧 Technologies

- **Python 3.12**
- **scikit-learn**: ML models & TF-IDF
- **NLTK**: Indonesian NLP
- **NumPy & Pandas**: Data processing
- **Streamlit**: Web interface
- **Matplotlib & Seaborn**: Visualizations

## 📖 Notebooks Documentation

Each notebook includes:
- ✅ Markdown explanations
- ✅ Code implementation
- ✅ Visualizations
- ✅ Saved outputs (CSV, JSON, PNG)
- ✅ Performance metrics

## 🤝 Contributing

This is a UAS project for STKI course. Feel free to use as reference.

## 👤 Author

**Daffa Afaf Firmansyah**
- NIM: A11.2022.14166
- Course: Sistem Temu Kembali Informasi (STKI)
- Institution: Universitas Dian Nuswantoro

## 📄 License

Academic project - Educational use only.

---

**Note**: Ensure `data/processed/preprocessed_corpus.json` exists before running the application. Run preprocessing notebooks if needed.
