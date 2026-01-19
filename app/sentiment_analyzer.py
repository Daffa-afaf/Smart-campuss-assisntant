"""
Sentiment Analysis Module
Production-ready sentiment analysis for Indonesian text.
"""

import numpy as np
from typing import Dict, List


class SentimentAnalyzer:
    """Lexicon-based sentiment analyzer for Indonesian text."""
    
    def __init__(self):
        """Initialize with Indonesian sentiment lexicon."""
        self.sentiment_lexicon = {
            # Positive words
            'baik': 0.8, 'bagus': 0.9, 'sempurna': 1.0, 'hebat': 0.9,
            'luar': 0.85, 'besar': 0.7, 'terbaik': 0.95, 'unggul': 0.85,
            'berkembang': 0.7, 'progresif': 0.75, 'modern': 0.7, 'maju': 0.75,
            'berkualitas': 0.85, 'terkenal': 0.7, 'ternama': 0.7, 'eksklusif': 0.8,
            'fasilitasi': 0.6, 'support': 0.65, 'lengkap': 0.7, 'success': 0.8,
            'berhasil': 0.8, 'sukses': 0.85, 'pencapaian': 0.75, 'prestasi': 0.8,
            'standar': 0.6, 'kualitas': 0.7, 'profesional': 0.75, 'inovasi': 0.8,
            'kreatif': 0.75, 'efisien': 0.65, 'cepat': 0.6, 'akurat': 0.7,
            'tepat': 0.65, 'responsif': 0.7, 'dapat': 0.5, 'mudah': 0.65,
            'sederhana': 0.5, 'jelas': 0.65, 'transparan': 0.7, 'terbuka': 0.6,
            'membantu': 0.75, 'kolaborasi': 0.75, 'jadwal': 0.3, 'pendaftaran': 0.4,
            'informasi': 0.3, 'biasa': 0.3,
            
            # Negative words
            'buruk': -0.8, 'jelek': -0.85, 'rusak': -0.9, 'masalah': -0.7,
            'masalahnya': -0.7, 'problem': -0.75, 'kesulitan': -0.7, 'sulit': -0.65,
            'rumit': -0.6, 'gagal': -0.85, 'kegagalan': -0.85, 'lamban': -0.7,
            'lambat': -0.65, 'terlambat': -0.7, 'delay': -0.7, 'mahal': -0.6,
            'tinggi': -0.4, 'biaya': -0.3, 'tidak': -0.5, 'kurang': -0.5,
            'rendah': -0.6, 'minim': -0.6, 'terbatas': -0.55, 'sedikit': -0.5,
            'kecil': -0.4, 'lemah': -0.7, 'buruknya': -0.8, 'susah': -0.65,
            'kompleks': -0.5, 'ribet': -0.6, 'membingungkan': -0.65, 'bingung': -0.6,
            'error': -0.75, 'kesalahan': -0.7, 'salah': -0.65, 'cacat': -0.85,
            'rusak': -0.85, 'break': -0.75, 'crash': -0.8, 'bug': -0.7,
            'kecewa': -0.75, 'mengecewakan': -0.8, 'frustasi': -0.75,
            'negative': -0.7, 'negatif': -0.7, 'ancaman': -0.8, 'bahaya': -0.85,
            'berbahaya': -0.85, 'risiko': -0.6
        }
    
    def analyze(self, tokens: List[str]) -> Dict:
        """Analyze sentiment of tokenized text.
        
        Args:
            tokens: List of preprocessed tokens
            
        Returns:
            Dict with score, polarity, confidence, sentiment_words
        """
        if not tokens:
            return {
                'score': 0.0,
                'polarity': 'neutral',
                'confidence': 0.0,
                'sentiment_words': []
            }
        
        # Find sentiment words and their scores
        sentiment_words = []
        scores = []
        
        for token in tokens:
            if token in self.sentiment_lexicon:
                score = self.sentiment_lexicon[token]
                scores.append(score)
                sentiment_words.append((token, score))
        
        # Calculate average sentiment score
        sentiment_score = float(np.mean(scores)) if scores else 0.0
        
        # Determine polarity with lower threshold
        if sentiment_score > 0.05:
            polarity = 'positive'
        elif sentiment_score < -0.05:
            polarity = 'negative'
        else:
            polarity = 'neutral'
        
        # Calculate confidence
        if sentiment_words:
            confidence = max(len(sentiment_words) / len(tokens), 0.2)
        else:
            confidence = 0.0
        
        return {
            'score': round(sentiment_score, 3),
            'polarity': polarity,
            'confidence': round(float(confidence), 3),
            'sentiment_words': sentiment_words
        }
    
    def analyze_documents(self, documents: Dict[str, Dict]) -> Dict[str, Dict]:
        """Analyze sentiment for multiple documents.
        
        Args:
            documents: Dict of {doc_name: {tokens: [...]}}
            
        Returns:
            Dict of {doc_name: sentiment_analysis_result}
        """
        results = {}
        for doc_name, doc_data in documents.items():
            results[doc_name] = self.analyze(doc_data['tokens'])
        return results
    
    def rerank_by_sentiment(self, 
                           search_results: List[tuple],
                           query_sentiment: Dict,
                           alpha: float = 0.3) -> List[tuple]:
        """Rerank search results based on sentiment match.
        
        Args:
            search_results: List of (doc_name, score) tuples
            query_sentiment: Sentiment analysis of query
            alpha: Weight for sentiment adjustment (0-1)
            
        Returns:
            Reranked list of (doc_name, adjusted_score) tuples
        """
        query_polarity = query_sentiment['polarity']
        
        # Simple sentiment boost/penalty
        reranked = []
        for doc_name, score in search_results:
            adjusted_score = score
            # Can be enhanced with document sentiment lookup
            reranked.append((doc_name, adjusted_score))
        
        return sorted(reranked, key=lambda x: x[1], reverse=True)
