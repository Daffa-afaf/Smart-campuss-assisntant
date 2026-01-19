"""
Enhanced Search Module with RAG Integration
Production-ready search interface for SmartCampus Assistant.
"""

import json
from typing import Dict, List, Optional
from .rag_engine import RAGEngine


class SmartCampusSearch:
    """Enhanced search interface with RAG capabilities."""
    
    def __init__(self, corpus_path: str = 'data/ir_docs'):
        """Initialize search engine.
        
        Args:
            corpus_path: Path to document corpus
        """
        self.rag = RAGEngine(corpus_path=corpus_path)
        self.rag.initialize()
    
    def search(self, 
               query: str, 
               mode: str = 'semantic',
               top_k: int = 5,
               include_sentiment: bool = True,
               include_summary: bool = True) -> Dict:
        """Perform enhanced search.
        
        Args:
            query: Search query
            mode: 'semantic' (VSM), 'boolean_and', 'boolean_or'
            top_k: Number of results
            include_sentiment: Include sentiment analysis
            include_summary: Include document summaries
            
        Returns:
            Search results with enhancements
        """
        if mode == 'semantic':
            results = self.rag.search(
                query, 
                top_k=top_k,
                use_sentiment=include_sentiment,
                return_summaries=include_summary
            )
        elif mode in ['boolean_and', 'boolean_or']:
            bool_mode = 'AND' if mode == 'boolean_and' else 'OR'
            bool_results = self.rag.boolean_search(query, mode=bool_mode)
            
            # Convert boolean results to same format as semantic search
            results = {
                'query': query,
                'query_tokens': self.rag.preprocessor.preprocess_text(query),
                'query_sentiment': self.rag.analyze_sentiment(query),
                'num_results': len(bool_results['results'][:top_k]),
                'results': []
            }
            
            # Enhance boolean results with details
            for doc_name in bool_results['results'][:top_k]:
                result = {
                    'doc_name': doc_name,
                    'score': 1.0  # Boolean has no score, use 1.0
                }
                
                if include_sentiment:
                    result['sentiment'] = self.rag.sentiment_analyzer.analyze(
                        self.rag.documents[doc_name]['tokens']
                    )
                
                if include_summary:
                    result['summary'] = self.rag.summarizer.query_focused_summary(
                        doc_name, results['query_tokens'], max_sentences=3
                    )
                
                results['results'].append(result)
        else:
            return {'error': f'Invalid mode: {mode}'}
        
        return results
    
    def ask(self, question: str, context_size: int = 3) -> str:
        """Ask a question and get an answer with context.
        
        Args:
            question: User question
            context_size: Number of documents to use as context
            
        Returns:
            Answer with context summary
        """
        # Search for relevant documents
        results = self.rag.search(question, top_k=context_size, return_summaries=True)
        
        if not results['results']:
            return "Maaf, tidak ada dokumen relevan yang ditemukan."
        
        # Build answer from summaries
        answer_parts = [
            f"Berdasarkan {results['num_results']} dokumen relevan:",
            ""
        ]
        
        for i, result in enumerate(results['results'], 1):
            doc_name = result['doc_name']
            score = result['score']
            summary = result.get('summary', '')
            
            answer_parts.append(f"{i}. {doc_name} (relevansi: {score:.2%})")
            if summary:
                answer_parts.append(f"   {summary[:200]}...")
            answer_parts.append("")
        
        return "\n".join(answer_parts)
    
    def get_document(self, doc_name: str) -> Dict:
        """Get full document information.
        
        Args:
            doc_name: Document name
            
        Returns:
            Document information
        """
        return self.rag.get_document_info(doc_name)
    
    def analyze_query(self, query: str) -> Dict:
        """Analyze query characteristics.
        
        Args:
            query: Query text
            
        Returns:
            Query analysis
        """
        tokens = self.rag.preprocessor.preprocess_text(query)
        sentiment = self.rag.sentiment_analyzer.analyze(tokens)
        
        return {
            'original_query': query,
            'tokens': tokens,
            'num_tokens': len(tokens),
            'sentiment': sentiment,
            'suggested_mode': 'semantic' if len(tokens) > 2 else 'boolean_or'
        }
    
    def compare_documents(self, doc_name1: str, doc_name2: str) -> Dict:
        """Compare two documents.
        
        Args:
            doc_name1: First document name
            doc_name2: Second document name
            
        Returns:
            Comparison results
        """
        doc1 = self.rag.get_document_info(doc_name1)
        doc2 = self.rag.get_document_info(doc_name2)
        
        if 'error' in doc1 or 'error' in doc2:
            return {'error': 'One or both documents not found'}
        
        # Calculate token overlap
        tokens1 = set(self.rag.documents[doc_name1]['tokens'])
        tokens2 = set(self.rag.documents[doc_name2]['tokens'])
        overlap = tokens1.intersection(tokens2)
        jaccard = len(overlap) / len(tokens1.union(tokens2))
        
        return {
            'doc1': doc_name1,
            'doc2': doc_name2,
            'common_tokens': len(overlap),
            'jaccard_similarity': round(jaccard, 3),
            'doc1_sentiment': doc1['sentiment'],
            'doc2_sentiment': doc2['sentiment']
        }
    
    def get_statistics(self) -> Dict:
        """Get corpus statistics.
        
        Returns:
            Corpus statistics
        """
        docs = self.rag.documents
        
        total_tokens = sum(len(doc['tokens']) for doc in docs.values())
        avg_tokens = total_tokens / len(docs) if docs else 0
        
        all_tokens = []
        for doc in docs.values():
            all_tokens.extend(doc['tokens'])
        
        vocab_size = len(set(all_tokens))
        
        return {
            'num_documents': len(docs),
            'total_tokens': total_tokens,
            'avg_tokens_per_doc': round(avg_tokens, 1),
            'vocabulary_size': vocab_size,
            'document_names': list(docs.keys())
        }


def main():
    """Example usage."""
    search = SmartCampusSearch()
    
    # Example queries
    queries = [
        "jadwal pendaftaran mahasiswa baru",
        "informasi beasiswa",
        "tata tertib perpustakaan"
    ]
    
    print("SmartCampus Search Engine - Demo\n")
    print("=" * 60)
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 60)
        
        results = search.search(query, top_k=3)
        
        print(f"Query sentiment: {results['query_sentiment']['polarity']}")
        print(f"Found {results['num_results']} results\n")
        
        for i, result in enumerate(results['results'], 1):
            print(f"{i}. {result['doc_name']} (score: {result['score']:.3f})")
            if 'summary' in result:
                print(f"   Summary: {result['summary'][:150]}...")
            print()


if __name__ == '__main__':
    main()
