"""
SmartCampus Assistant - RAG System
Production-ready modules for information retrieval and document analysis.
"""

from .preprocessor import DocumentPreprocessor, TextPreprocessor, IndonesianStemmer
from .vectorizer import VectorSpaceModel, BooleanRetrieval, TermFrequency, DocumentFrequency
from .sentiment_analyzer import SentimentAnalyzer
from .summarizer import Summarizer
from .classify import QueryClassifier
from .cluster import DocumentClusterer
from .evaluator import (
    precision_at_k, recall_at_k, average_precision, mean_average_precision,
    reciprocal_rank, mean_reciprocal_rank, dcg, ndcg_at_k, IRMetrics
)
from .rag_engine import RAGEngine
from .search_plus import SmartCampusSearch

__version__ = '1.0.0'

__all__ = [
    # Preprocessing
    'DocumentPreprocessor',
    'TextPreprocessor',
    'IndonesianStemmer',
    
    # Vectorization & Retrieval
    'VectorSpaceModel',
    'BooleanRetrieval',
    'TermFrequency',
    'DocumentFrequency',
    
    # Analysis
    'SentimentAnalyzer',
    'Summarizer',
    
    # ML Models
    'QueryClassifier',
    'DocumentClusterer',
    
    # Evaluation
    'precision_at_k',
    'recall_at_k',
    'average_precision',
    'mean_average_precision',
    'reciprocal_rank',
    'mean_reciprocal_rank',
    'dcg',
    'ndcg_at_k',
    'IRMetrics',
    
    # Main Engine
    'RAGEngine',
    'SmartCampusSearch',
]
