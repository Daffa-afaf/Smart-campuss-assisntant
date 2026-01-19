"""
RAG Engine - Retrieval-Augmented Generation for SmartCampus Assistant
Main integration module combining all RAG components.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from .preprocessor import DocumentPreprocessor
from .vectorizer import VectorSpaceModel, BooleanRetrieval
from .sentiment_analyzer import SentimentAnalyzer
from .summarizer import Summarizer
from .classify import QueryClassifier
from .cluster import DocumentClusterer
from .evaluator import IRMetrics


class RAGEngine:
    """Main RAG engine for SmartCampus Assistant."""
    
    def __init__(self, 
                 corpus_path: str = 'data/ir_docs',
                 processed_data_path: str = 'data/processed'):
        """Initialize RAG engine.
        
        Args:
            corpus_path: Path to document corpus
            processed_data_path: Path to processed data
        """
        self.corpus_path = Path(corpus_path)
        self.processed_data_path = Path(processed_data_path)
        
        # Initialize components
        self.preprocessor = None
        self.vsm = None
        self.boolean_retrieval = None
        self.sentiment_analyzer = SentimentAnalyzer()
        self.summarizer = None
        self.classifier = None
        self.clusterer = None
        
        # Data
        self.documents = {}
        self.doc_texts = {}
        self.is_initialized = False
    
    def initialize(self):
        """Initialize and load all components."""
        print("Initializing RAG Engine...")
        
        # Load preprocessed documents
        preprocessed_file = self.processed_data_path / 'preprocessed_corpus.json'
        if preprocessed_file.exists():
            with open(preprocessed_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.documents = data['documents']
            print(f"✓ Loaded {len(self.documents)} preprocessed documents")
        else:
            print("✗ Preprocessed corpus not found. Run preprocessing first.")
            return False
        
        # Load original texts
        for txt_file in self.corpus_path.glob('*.txt'):
            with open(txt_file, 'r', encoding='utf-8') as f:
                self.doc_texts[txt_file.stem] = f.read()
        print(f"✓ Loaded {len(self.doc_texts)} original texts")
        
        # Initialize preprocessor
        self.preprocessor = DocumentPreprocessor()
        
        # Initialize VSM
        self.vsm = VectorSpaceModel(self.documents)
        print("✓ Initialized Vector Space Model")
        
        # Initialize Boolean Retrieval
        self.boolean_retrieval = BooleanRetrieval(self.documents)
        print("✓ Initialized Boolean Retrieval")
        
        # Initialize Summarizer
        self.summarizer = Summarizer(self.documents, self.doc_texts)
        print("✓ Initialized Summarizer")
        
        self.is_initialized = True
        print("✓ RAG Engine initialized successfully!")
        return True
    
    def search(self, 
               query: str, 
               top_k: int = 5,
               use_sentiment: bool = True,
               return_summaries: bool = True) -> Dict:
        """Search documents with RAG enhancements.
        
        Args:
            query: User query string
            top_k: Number of results to return
            use_sentiment: Whether to use sentiment-aware ranking
            return_summaries: Whether to return summaries
            
        Returns:
            Dict with search results and metadata
        """
        if not self.is_initialized:
            raise ValueError("RAG Engine not initialized. Call initialize() first.")
        
        # Preprocess query
        query_tokens = self.preprocessor.preprocess_text(query)
        
        # Analyze query sentiment
        query_sentiment = self.sentiment_analyzer.analyze(query_tokens)
        
        # Retrieve documents using VSM
        results = self.vsm.search(query_tokens, top_k=top_k)
        
        # Enhance with sentiment (if enabled)
        if use_sentiment:
            results = self.sentiment_analyzer.rerank_by_sentiment(
                results, query_sentiment, alpha=0.2
            )
        
        # Build response
        response = {
            'query': query,
            'query_tokens': query_tokens,
            'query_sentiment': query_sentiment,
            'num_results': len(results),
            'results': []
        }
        
        # Process each result
        for doc_name, score in results:
            result = {
                'doc_name': doc_name,
                'score': round(score, 4),
                'sentiment': self.sentiment_analyzer.analyze(
                    self.documents[doc_name]['tokens']
                )
            }
            
            # Add summary if requested
            if return_summaries:
                result['summary'] = self.summarizer.query_focused_summary(
                    doc_name, query_tokens, max_sentences=3
                )
            
            response['results'].append(result)
        
        return response
    
    def boolean_search(self, query: str, mode: str = 'OR') -> Dict:
        """Perform boolean search.
        
        Args:
            query: User query string
            mode: 'AND' or 'OR'
            
        Returns:
            Dict with search results
        """
        if not self.is_initialized:
            raise ValueError("RAG Engine not initialized. Call initialize() first.")
        
        query_tokens = self.preprocessor.preprocess_text(query)
        matching_docs = self.boolean_retrieval.search(query_tokens, mode=mode)
        
        return {
            'query': query,
            'mode': mode,
            'num_results': len(matching_docs),
            'results': matching_docs
        }
    
    def get_document_summary(self, doc_name: str, ratio: float = 0.3) -> str:
        """Get summary of a specific document.
        
        Args:
            doc_name: Name of document
            ratio: Summary ratio
            
        Returns:
            Summary text
        """
        if not self.is_initialized:
            raise ValueError("RAG Engine not initialized. Call initialize() first.")
        
        return self.summarizer.summarize(doc_name, ratio=ratio)
    
    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment analysis result
        """
        tokens = self.preprocessor.preprocess_text(text)
        return self.sentiment_analyzer.analyze(tokens)
    
    def multi_document_summary(self, doc_names: List[str], max_sentences: int = 10) -> str:
        """Create multi-document summary.
        
        Args:
            doc_names: List of document names
            max_sentences: Maximum sentences
            
        Returns:
            Combined summary
        """
        if not self.is_initialized:
            raise ValueError("RAG Engine not initialized. Call initialize() first.")
        
        return self.summarizer.summarize_multiple(doc_names, max_sentences=max_sentences)
    
    def get_document_info(self, doc_name: str) -> Dict:
        """Get full information about a document.
        
        Args:
            doc_name: Name of document
            
        Returns:
            Dict with document metadata and content
        """
        if not self.is_initialized:
            raise ValueError("RAG Engine not initialized. Call initialize() first.")
        
        if doc_name not in self.documents:
            return {'error': f'Document {doc_name} not found'}
        
        doc_data = self.documents[doc_name]
        
        return {
            'name': doc_name,
            'num_tokens': len(doc_data['tokens']),
            'unique_tokens': len(set(doc_data['tokens'])),
            'text': self.doc_texts.get(doc_name, ''),
            'sentiment': self.sentiment_analyzer.analyze(doc_data['tokens']),
            'summary': self.summarizer.summarize(doc_name, ratio=0.3)
        }
    
    def evaluate(self, 
                 test_queries: Dict[str, str],
                 ground_truth: Dict[str, List[str]]) -> Dict:
        """Evaluate retrieval quality.
        
        Args:
            test_queries: Dict of {query_id: query_text}
            ground_truth: Dict of {query_id: [relevant_doc_names]}
            
        Returns:
            Evaluation metrics
        """
        if not self.is_initialized:
            raise ValueError("RAG Engine not initialized. Call initialize() first.")
        
        # Run queries
        runs = {}
        for qid, query_text in test_queries.items():
            results = self.search(query_text, top_k=10, return_summaries=False)
            runs[qid] = [r['doc_name'] for r in results['results']]
        
        # Calculate metrics
        metrics = IRMetrics(runs, ground_truth)
        evaluation = metrics.evaluate(k_values=[1, 3, 5])
        
        return evaluation
