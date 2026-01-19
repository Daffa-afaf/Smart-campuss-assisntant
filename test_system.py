"""
Test script for SmartCampus Assistant RAG System
Run this to verify all components are working correctly.
"""

import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

def test_import():
    """Test if all modules can be imported."""
    print("Testing imports...")
    try:
        from app import (
            TextPreprocessor, VectorSpaceModel, SentimentAnalyzer,
            Summarizer, QueryClassifier, DocumentClusterer,
            RAGEngine, SmartCampusSearch
        )
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_rag_engine():
    """Test RAG Engine initialization and basic operations."""
    print("\nTesting RAG Engine...")
    try:
        from app.rag_engine import RAGEngine
        
        rag = RAGEngine(corpus_path='data/ir_docs')
        success = rag.initialize()
        
        if not success:
            print("✗ RAG Engine initialization failed")
            return False
        
        # Test search
        results = rag.search("jadwal pendaftaran", top_k=3)
        assert 'results' in results
        assert len(results['results']) > 0
        print(f"✓ Search returned {len(results['results'])} results")
        
        # Test sentiment
        sentiment = rag.analyze_sentiment("bagus dan berkualitas")
        assert sentiment['polarity'] == 'positive'
        print(f"✓ Sentiment analysis: {sentiment['polarity']}")
        
        # Test summary
        summary = rag.get_document_summary(list(rag.documents.keys())[0])
        assert len(summary) > 0
        print(f"✓ Summary generated: {len(summary)} chars")
        
        return True
    except Exception as e:
        print(f"✗ RAG Engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_search_interface():
    """Test SmartCampusSearch interface."""
    print("\nTesting Search Interface...")
    try:
        from app.search_plus import SmartCampusSearch
        
        search = SmartCampusSearch(corpus_path='data/ir_docs')
        
        # Test semantic search
        results = search.search("informasi beasiswa", mode='semantic', top_k=3)
        assert results['num_results'] > 0
        print(f"✓ Semantic search: {results['num_results']} results")
        
        # Test boolean search
        results = search.search("mahasiswa pendaftaran", mode='boolean_or', top_k=3)
        assert results['num_results'] > 0
        print(f"✓ Boolean search: {results['num_results']} results")
        
        # Test Q&A
        answer = search.ask("Bagaimana cara mendaftar?", context_size=2)
        assert len(answer) > 0
        print(f"✓ Q&A generated: {len(answer)} chars")
        
        # Test statistics
        stats = search.get_statistics()
        assert stats['num_documents'] > 0
        print(f"✓ Statistics: {stats['num_documents']} docs, {stats['vocabulary_size']} vocab")
        
        return True
    except Exception as e:
        print(f"✗ Search interface test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluation_metrics():
    """Test evaluation metrics."""
    print("\nTesting Evaluation Metrics...")
    try:
        from app.evaluator import (
            precision_at_k, recall_at_k, average_precision,
            mean_average_precision, reciprocal_rank, ndcg_at_k
        )
        
        # Test data
        retrieved = ['doc1', 'doc2', 'doc3', 'doc4']
        relevant = ['doc2', 'doc3']
        
        # Test metrics
        p_at_3 = precision_at_k(retrieved, relevant, 3)
        assert 0 <= p_at_3 <= 1
        print(f"✓ Precision@3: {p_at_3:.3f}")
        
        r_at_3 = recall_at_k(retrieved, relevant, 3)
        assert 0 <= r_at_3 <= 1
        print(f"✓ Recall@3: {r_at_3:.3f}")
        
        ap = average_precision(retrieved, relevant)
        assert 0 <= ap <= 1
        print(f"✓ Average Precision: {ap:.3f}")
        
        rr = reciprocal_rank(retrieved, relevant)
        assert 0 <= rr <= 1
        print(f"✓ Reciprocal Rank: {rr:.3f}")
        
        gains = {'doc1': 1, 'doc2': 3, 'doc3': 2, 'doc4': 1}
        ndcg = ndcg_at_k(retrieved, relevant, gains, 3)
        assert 0 <= ndcg <= 1
        print(f"✓ nDCG@3: {ndcg:.3f}")
        
        return True
    except Exception as e:
        print(f"✗ Evaluation metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("SmartCampus Assistant - System Test")
    print("=" * 60)
    
    tests = [
        ("Imports", test_import),
        ("RAG Engine", test_rag_engine),
        ("Search Interface", test_search_interface),
        ("Evaluation Metrics", test_evaluation_metrics),
    ]
    
    results = []
    for name, test_func in tests:
        success = test_func()
        results.append((name, success))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{name:.<40} {status}")
    
    print("=" * 60)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! System ready for deployment.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check errors above.")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
