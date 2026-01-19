"""
Document Preprocessor Module
Production-ready preprocessing for SmartCampus Assistant RAG system.
Auto-generated from src/preprocess.ipynb
"""

import os
import json
import re
import ssl
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Indonesian NLP
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data if not present
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class IndonesianStemmer:
    """Simple Indonesian stemmer using common suffix/prefix removal."""
    def stem(self, word):
        """Apply basic Indonesian stemming rules."""
        if len(word) <= 2:
            return word
        
        # Remove common prefixes
        prefixes = ['me', 'di', 'pe', 'ke', 'be', 'te']
        for prefix in prefixes:
            if word.startswith(prefix) and len(word) > len(prefix) + 2:
                word = word[len(prefix):]
                break
        
        # Remove common suffixes
        suffixes = ['kan', 'an', 'i', 'nya', 'ku', 'mu']
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                word = word[:-len(suffix)]
                break
        
        return word


class DocumentPreprocessor:
    """Handles document preprocessing for Indonesian text in RAG system."""
    
    def __init__(self, config: dict = None, stemmer=None):
        self.config = config or {
            'corpus_path': 'data/ir_docs',
            'min_token_length': 3,
            'lowercase': True,
            'remove_punctuation': True,
            'remove_stopwords': True,
            'apply_stemming': True,
            'output_path': 'data/processed'
        }
        self.stemmer = stemmer or IndonesianStemmer()
        self.stop_words = set(stopwords.words('indonesian'))
        self.documents = {}
        self.preprocessed_docs = {}
        
    def load_documents(self, corpus_path: str) -> dict:
        """Load all .txt documents from corpus directory."""
        self.documents = {}
        corpus_path = Path(corpus_path)
        
        for txt_file in corpus_path.glob('*.txt'):
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    self.documents[txt_file.stem] = content
            except Exception as e:
                print(f"Error loading {txt_file.name}: {e}")
        
        print(f"✓ Loaded {len(self.documents)} documents")
        return self.documents
    
    def clean_text(self, text: str) -> str:
        """Remove special characters, extra whitespace."""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        # Remove email
        text = re.sub(r'\S+@\S+', '', text)
        # Remove special characters if configured
        if self.config['remove_punctuation']:
            text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def tokenize(self, text: str) -> list:
        """Tokenize text using NLTK word_tokenize."""
        return word_tokenize(text)
    
    def preprocess_text(self, text: str) -> list:
        """Full preprocessing pipeline."""
        # Step 1: Lowercase
        if self.config['lowercase']:
            text = text.lower()
        
        # Step 2: Clean text
        text = self.clean_text(text)
        
        # Step 3: Tokenize
        tokens = self.tokenize(text)
        
        # Step 4: Remove stopwords
        if self.config['remove_stopwords']:
            tokens = [t for t in tokens if t not in self.stop_words and len(t) >= self.config['min_token_length']]
        else:
            tokens = [t for t in tokens if len(t) >= self.config['min_token_length']]
        
        # Step 5: Stemming
        if self.config['apply_stemming']:
            tokens = [self.stemmer.stem(t) for t in tokens]
        
        return tokens
    
    def preprocess_corpus(self) -> dict:
        """Preprocess all loaded documents."""
        self.preprocessed_docs = {}
        
        for doc_name, content in self.documents.items():
            tokens = self.preprocess_text(content)
            self.preprocessed_docs[doc_name] = {
                'tokens': tokens,
                'token_count': len(tokens),
                'unique_tokens': len(set(tokens)),
                'text': ' '.join(tokens)  # For vectorization
            }
        
        print(f"✓ Preprocessed {len(self.preprocessed_docs)} documents")
        return self.preprocessed_docs
    
    def get_statistics(self) -> pd.DataFrame:
        """Get preprocessing statistics."""
        stats = []
        for doc_name, data in self.preprocessed_docs.items():
            stats.append({
                'Document': doc_name,
                'Total Tokens': data['token_count'],
                'Unique Tokens': data['unique_tokens'],
                'Vocabulary': len(set(data['tokens']))
            })
        return pd.DataFrame(stats)
    
    def save_preprocessed(self, output_path: str = None) -> None:
        """Save preprocessed documents to JSON."""
        if output_path is None:
            output_path = self.config['output_path']
        
        output_file = Path(output_path) / 'preprocessed_corpus.json'
        
        # Convert for JSON serialization
        data = {
            'config': self.config,
            'documents': self.preprocessed_docs
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Saved to {output_file}")


# Alias for compatibility
TextPreprocessor = DocumentPreprocessor


if __name__ == "__main__":
    # Example usage
    config = {
        'corpus_path': 'data/ir_docs',
        'min_token_length': 3,
        'lowercase': True,
        'remove_punctuation': True,
        'remove_stopwords': True,
        'apply_stemming': True,
        'output_path': 'data/processed'
    }
    
    preprocessor = DocumentPreprocessor(config)
    preprocessor.load_documents(config['corpus_path'])
    preprocessor.preprocess_corpus()
    preprocessor.save_preprocessed()
