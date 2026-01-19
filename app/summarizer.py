"""
Text Summarization Module
Production-ready extractive summarization for retrieved documents.
"""

import re
import numpy as np
from typing import Dict, List


class Summarizer:
    """Extractive summarization using TF-IDF weighted sentences."""
    
    def __init__(self, documents: Dict[str, Dict], texts: Dict[str, str]):
        """Initialize summarizer.
        
        Args:
            documents: Preprocessed documents dict {doc_name: {tokens: [...]}}
            texts: Original full text dict {doc_name: text}
        """
        self.documents = documents
        self.texts = texts
    
    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'[.!?\n]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        return sentences
    
    def sentence_tokens(self, sentence: str) -> List[str]:
        """Tokenize sentence."""
        return sentence.lower().split()
    
    def score_sentence(self, sentence: str, doc_tokens: List[str]) -> float:
        """Score sentence based on TF of its words in document."""
        sentence_tokens = set(self.sentence_tokens(sentence))
        doc_token_set = set(doc_tokens)
        
        overlap = sentence_tokens.intersection(doc_token_set)
        if len(sentence_tokens) == 0:
            return 0
        
        score = 0
        for token in overlap:
            score += doc_tokens.count(token)
        
        return score / len(doc_tokens) if len(doc_tokens) > 0 else 0
    
    def summarize(self, doc_name: str, ratio: float = 0.3, max_sentences: int = 5) -> str:
        """Extract summary from document.
        
        Args:
            doc_name: Name of document
            ratio: Fraction of sentences to keep (0-1)
            max_sentences: Maximum number of sentences
            
        Returns:
            Summary text
        """
        if doc_name not in self.texts:
            return ""
        
        text = self.texts[doc_name]
        tokens = self.documents[doc_name]['tokens']
        
        sentences = self.split_sentences(text)
        if len(sentences) <= max_sentences:
            return text
        
        # Score each sentence
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            score = self.score_sentence(sentence, tokens)
            scored_sentences.append((i, sentence, score))
        
        # Sort by score and take top sentences
        num_sentences = max(1, min(int(len(sentences) * ratio), max_sentences))
        top_sentences = sorted(scored_sentences, key=lambda x: x[2], reverse=True)[:num_sentences]
        
        # Sort by original order for coherence
        top_sentences = sorted(top_sentences, key=lambda x: x[0])
        
        # Join sentences
        summary = '. '.join([sent[1] for sent in top_sentences]) + '.'
        return summary
    
    def summarize_multiple(self, 
                          doc_names: List[str], 
                          ratio: float = 0.3,
                          max_sentences: int = 10) -> str:
        """Create multi-document summary.
        
        Args:
            doc_names: List of document names
            ratio: Fraction of sentences to keep per document
            max_sentences: Maximum total sentences
            
        Returns:
            Combined summary text
        """
        summaries = []
        for doc_name in doc_names:
            summary = self.summarize(doc_name, ratio=ratio, max_sentences=3)
            if summary:
                summaries.append(f"[{doc_name}] {summary}")
        
        return '\n\n'.join(summaries[:max_sentences])
    
    def query_focused_summary(self,
                             doc_name: str,
                             query_tokens: List[str],
                             max_sentences: int = 3) -> str:
        """Create query-focused summary.
        
        Args:
            doc_name: Name of document
            query_tokens: Query terms to focus on
            max_sentences: Maximum sentences
            
        Returns:
            Query-focused summary
        """
        if doc_name not in self.texts:
            return ""
        
        text = self.texts[doc_name]
        sentences = self.split_sentences(text)
        
        # Score sentences by query term overlap
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            sent_tokens = set(self.sentence_tokens(sentence))
            query_set = set(query_tokens)
            overlap = len(sent_tokens.intersection(query_set))
            
            if overlap > 0:
                scored_sentences.append((i, sentence, overlap))
        
        if not scored_sentences:
            # Fallback to regular summary
            return self.summarize(doc_name, ratio=0.2, max_sentences=max_sentences)
        
        # Take top query-relevant sentences
        top_sentences = sorted(scored_sentences, key=lambda x: x[2], reverse=True)[:max_sentences]
        top_sentences = sorted(top_sentences, key=lambda x: x[0])
        
        summary = '. '.join([sent[1] for sent in top_sentences]) + '.'
        return summary
