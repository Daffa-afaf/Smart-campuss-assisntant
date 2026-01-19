"""
Vectorization & Term Weighting Module
Production-ready vectorization for SmartCampus Assistant RAG system.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TermFrequency:
    """Calculate Term Frequency for documents."""
    
    def __init__(self, documents: Dict[str, Dict]):
        """Initialize with preprocessed documents.
        
        Args:
            documents: Dict of {doc_name: {tokens: [...], text: '...'}}
        """
        self.documents = documents
        self.doc_names = list(documents.keys())
        self.tf_matrix = None
        self.vocabulary = set()
        
    def build_vocabulary(self):
        """Build vocabulary from all documents."""
        for doc in self.documents.values():
            self.vocabulary.update(doc['tokens'])
        self.vocabulary = sorted(list(self.vocabulary))
        return self.vocabulary
    
    def calculate_tf(self, doc_name: str, term: str) -> float:
        """Calculate TF for a term in a document."""
        tokens = self.documents[doc_name]['tokens']
        if len(tokens) == 0:
            return 0.0
        term_count = tokens.count(term)
        return term_count / len(tokens)
    
    def build_tf_matrix(self) -> pd.DataFrame:
        """Build TF matrix for all documents."""
        self.build_vocabulary()
        tf_data = []
        
        for doc_name in self.doc_names:
            tf_row = [self.calculate_tf(doc_name, term) for term in self.vocabulary]
            tf_data.append(tf_row)
        
        self.tf_matrix = pd.DataFrame(tf_data, index=self.doc_names, columns=self.vocabulary)
        return self.tf_matrix


class DocumentFrequency:
    """Calculate Document Frequency and IDF."""
    
    def __init__(self, documents: Dict[str, Dict], vocabulary: List[str]):
        self.documents = documents
        self.vocabulary = vocabulary
        self.num_docs = len(documents)
        self.df = {}
        self.idf = {}
    
    def calculate_df(self, term: str) -> int:
        """Count how many documents contain this term."""
        count = sum(1 for doc in self.documents.values() if term in doc['tokens'])
        return count
    
    def calculate_idf(self, term: str) -> float:
        """Calculate IDF for a term."""
        df = self.df.get(term, 1)
        if df == 0:
            df = 1
        return np.log(self.num_docs / df)
    
    def build_df_idf(self) -> pd.DataFrame:
        """Build DF and IDF for all terms."""
        df_data = []
        
        for term in self.vocabulary:
            df = self.calculate_df(term)
            idf = self.calculate_idf(term)
            self.df[term] = df
            self.idf[term] = idf
            df_data.append({'term': term, 'df': df, 'idf': idf})
        
        df_table = pd.DataFrame(df_data).sort_values('df', ascending=False)
        return df_table


class VectorSpaceModel:
    """Vector Space Model for document retrieval using TF-IDF."""
    
    def __init__(self, documents: Dict[str, Dict]):
        """Initialize VSM with preprocessed documents."""
        self.documents = documents
        self.doc_names = list(documents.keys())
        
        # Build corpus for TfidfVectorizer
        self.corpus = [' '.join(doc['tokens']) for doc in documents.values()]
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer()
        self.doc_vectors = self.vectorizer.fit_transform(self.corpus)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
    def query_vector(self, query_tokens: List[str]) -> np.ndarray:
        """Convert query to TF-IDF vector."""
        query_text = ' '.join(query_tokens)
        return self.vectorizer.transform([query_text])
    
    def search(self, query_tokens: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """Search documents using cosine similarity.
        
        Args:
            query_tokens: Preprocessed query tokens
            top_k: Number of top results to return
            
        Returns:
            List of (doc_name, score) tuples
        """
        q_vec = self.query_vector(query_tokens)
        scores = cosine_similarity(q_vec, self.doc_vectors).flatten()
        
        # Get top-k results
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [(self.doc_names[idx], float(scores[idx])) for idx in top_indices]
        
        return results
    
    def get_tfidf_matrix(self) -> pd.DataFrame:
        """Get TF-IDF matrix as DataFrame."""
        tfidf_dense = self.doc_vectors.toarray()
        return pd.DataFrame(tfidf_dense, index=self.doc_names, columns=self.feature_names)


class BooleanRetrieval:
    """Boolean retrieval model with AND, OR, NOT operations."""
    
    def __init__(self, documents: Dict[str, Dict]):
        """Initialize with preprocessed documents."""
        self.documents = documents
        self.doc_names = list(documents.keys())
        self.inverted_index = self._build_inverted_index()
    
    def _build_inverted_index(self) -> Dict[str, set]:
        """Build inverted index mapping terms to document IDs."""
        index = {}
        for doc_name, doc_data in self.documents.items():
            for token in set(doc_data['tokens']):
                if token not in index:
                    index[token] = set()
                index[token].add(doc_name)
        return index
    
    def search_and(self, terms: List[str]) -> set:
        """AND operation - documents containing ALL terms."""
        if not terms:
            return set()
        
        result = self.inverted_index.get(terms[0], set()).copy()
        for term in terms[1:]:
            result &= self.inverted_index.get(term, set())
        return result
    
    def search_or(self, terms: List[str]) -> set:
        """OR operation - documents containing ANY term."""
        result = set()
        for term in terms:
            result |= self.inverted_index.get(term, set())
        return result
    
    def search_not(self, include_terms: List[str], exclude_terms: List[str]) -> set:
        """NOT operation - documents with include_terms but not exclude_terms."""
        result = self.search_or(include_terms) if include_terms else set(self.doc_names)
        for term in exclude_terms:
            result -= self.inverted_index.get(term, set())
        return result
    
    def search(self, query_tokens: List[str], mode: str = 'OR') -> List[str]:
        """Search using boolean retrieval.
        
        Args:
            query_tokens: List of query terms
            mode: 'AND' or 'OR'
            
        Returns:
            List of matching document names
        """
        if mode.upper() == 'AND':
            return list(self.search_and(query_tokens))
        else:
            return list(self.search_or(query_tokens))
