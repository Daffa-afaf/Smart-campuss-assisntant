#!/usr/bin/env python
"""
CLI Demo for K-NN Document Classification
SmartCampus Assistant - Interactive Classification Tool

Usage:
    python classify_demo.py
    python classify_demo.py --text "Bagaimana cara daftar kuliah di Udinus?"
    python classify_demo.py --k 5
"""

import sys
import argparse
from pathlib import Path
import json
import pandas as pd
import numpy as np
from collections import Counter

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent / 'app'))

from app.classify import KNNClassifier
from app.preprocessor import DocumentPreprocessor
from app.vectorizer import VectorSpaceModel


class ClassificationDemo:
    """Interactive classification demo."""
    
    def __init__(self, k: int = 3):
        """Initialize demo.
        
        Args:
            k: Number of neighbors for KNN
        """
        self.k = k
        self.classifier = None
        self.preprocessor = None
        self.vsm = None
        self.doc_names = []
        
    def load_model(self):
        """Load trained model and data."""
        print("Loading model and data...")
        
        # Load preprocessed corpus
        corpus_file = Path('data/processed/preprocessed_corpus.json')
        if not corpus_file.exists():
            print("❌ Error: preprocessed_corpus.json not found!")
            print("Please run preprocessing first.")
            sys.exit(1)
        
        with open(corpus_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        documents = data['documents']
        
        # Load TF-IDF features
        tfidf_file = Path('data/processed/tfidf_reduced.csv')
        if not tfidf_file.exists():
            print("❌ Error: tfidf_reduced.csv not found!")
            print("Please run feature selection notebook first.")
            sys.exit(1)
        
        tfidf_matrix = pd.read_csv(tfidf_file, index_col=0)
        X = tfidf_matrix.values
        self.doc_names = list(tfidf_matrix.index)
        
        # Create labels (0: FAQ, 1: Profile/Academic, 2: Others)
        y = []
        for doc_name in self.doc_names:
            if 'faq' in doc_name.lower():
                y.append(0)
            elif 'profil' in doc_name.lower() or 'kurikulum' in doc_name.lower():
                y.append(1)
            else:
                y.append(2)
        y = np.array(y)
        
        # Initialize classifier
        self.classifier = KNNClassifier(k=self.k, metric='cosine', weighted=True)
        self.classifier.train(X, y, feature_names=list(tfidf_matrix.columns))
        
        # Initialize preprocessor
        self.preprocessor = DocumentPreprocessor()
        
        # Initialize VSM for vectorization
        self.vsm = VectorSpaceModel(documents)
        
        print(f"✅ Model loaded successfully!")
        print(f"   - Training documents: {len(self.doc_names)}")
        print(f"   - Features: {X.shape[1]}")
        print(f"   - K neighbors: {self.k}")
        print()
    
    def classify_text(self, text: str):
        """Classify input text.
        
        Args:
            text: Input text to classify
        """
        print("=" * 70)
        print("INPUT TEXT:")
        print("-" * 70)
        print(text)
        print()
        
        # Preprocess text
        tokens = self.preprocessor.preprocess_text(text)
        print(f"PREPROCESSED TOKENS: {tokens[:10]}..." if len(tokens) > 10 else f"PREPROCESSED TOKENS: {tokens}")
        print()
        
        # Vectorize using TF-IDF
        # Simple approach: match features with training vocab
        query_vector = np.zeros(len(self.classifier.feature_names))
        
        # Count term frequencies in query
        token_counts = Counter(tokens)
        total_tokens = len(tokens)
        
        # Calculate TF for query terms that match training features
        for i, term in enumerate(self.classifier.feature_names):
            if term in token_counts:
                # Simple TF
                tf = token_counts[term] / total_tokens if total_tokens > 0 else 0
                # Use pre-computed IDF from training (approximate)
                query_vector[i] = tf
        
        # Normalize query vector
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm
        
        # Classify with neighbors
        pred_label, pred_name, confidence, neighbors = self.classifier.predict(
            query_vector, 
            return_neighbors=True
        )
        
        # Display results
        print("=" * 70)
        print("CLASSIFICATION RESULTS:")
        print("-" * 70)
        print(f"🎯 Predicted Category: {pred_name} (Label {pred_label})")
        print(f"📊 Confidence: {confidence:.2%}")
        print()
        
        print("TOP-3 NEAREST NEIGHBORS:")
        print("-" * 70)
        for rank, (doc_idx, neighbor_label, similarity) in enumerate(neighbors[:3], 1):
            doc_name = self.doc_names[doc_idx]
            category = self.classifier.y_labels.get(neighbor_label, 'Unknown')
            print(f"{rank}. Document: {doc_name}")
            print(f"   Category: {category}")
            print(f"   Similarity: {similarity:.4f}")
            print()
        
        print("=" * 70)
    
    def run_interactive(self):
        """Run interactive classification mode."""
        print("\n" + "=" * 70)
        print("🎓 SmartCampus Assistant - Interactive Classification")
        print("=" * 70)
        print()
        
        while True:
            print("\nEnter text to classify (or 'quit' to exit):")
            text = input("> ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye! 👋")
                break
            
            if not text:
                print("⚠️  Please enter some text.")
                continue
            
            try:
                self.classify_text(text)
            except Exception as e:
                print(f"\n❌ Error during classification: {e}")
                import traceback
                traceback.print_exc()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="K-NN Document Classification Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python classify_demo.py
  python classify_demo.py --text "Bagaimana cara daftar kuliah?"
  python classify_demo.py --k 5 --text "Profil fakultas ilmu komputer"
        """
    )
    
    parser.add_argument(
        '--text', '-t',
        type=str,
        help='Text to classify (if not provided, runs in interactive mode)'
    )
    
    parser.add_argument(
        '--k',
        type=int,
        default=3,
        help='Number of neighbors (default: 3)'
    )
    
    args = parser.parse_args()
    
    # Initialize demo
    demo = ClassificationDemo(k=args.k)
    
    try:
        demo.load_model()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)
    
    # Run classification
    if args.text:
        # Single classification mode
        demo.classify_text(args.text)
    else:
        # Interactive mode
        demo.run_interactive()


if __name__ == "__main__":
    main()
