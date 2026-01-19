"""
SmartCampus Assistant - Streamlit Web Interface
Interactive web application for document search and analysis.
"""

import streamlit as st
import sys
from pathlib import Path
import json

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent / 'app'))

from app.search_plus import SmartCampusSearch
from app.rag_engine import RAGEngine

# Page config
st.set_page_config(
    page_title="SmartCampus Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .sentiment-positive {
        color: #2ecc71;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #e74c3c;
        font-weight: bold;
    }
    .sentiment-neutral {
        color: #95a5a6;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_rag_engine():
    """Load RAG engine (cached)."""
    with st.spinner('Initializing SmartCampus Assistant...'):
        search = SmartCampusSearch(corpus_path='data/ir_docs')
        return search


def format_sentiment(sentiment):
    """Format sentiment with color."""
    polarity = sentiment['polarity']
    score = sentiment['score']
    confidence = sentiment['confidence']
    
    if polarity == 'positive':
        color_class = 'sentiment-positive'
        emoji = '😊'
    elif polarity == 'negative':
        color_class = 'sentiment-negative'
        emoji = '😟'
    else:
        color_class = 'sentiment-neutral'
        emoji = '😐'
    
    return f'<span class="{color_class}">{emoji} {polarity.upper()}</span> (score: {score:.2f}, confidence: {confidence:.2f})'


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<div class="main-header">🎓 SmartCampus Assistant</div>', unsafe_allow_html=True)
    st.markdown('---')
    
    # Load engine
    try:
        search = load_rag_engine()
    except Exception as e:
        st.error(f"Failed to initialize: {str(e)}")
        st.info("Please ensure data/processed/preprocessed_corpus.json exists. Run preprocessing notebooks first.")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        search_mode = st.selectbox(
            "Search Mode",
            ['semantic', 'boolean_or', 'boolean_and'],
            help="Semantic: TF-IDF similarity | Boolean: Exact term matching"
        )
        
        top_k = st.slider("Number of Results", 1, 10, 5)
        
        include_sentiment = st.checkbox("Include Sentiment Analysis", value=True)
        include_summary = st.checkbox("Include Summaries", value=True)
        
        st.markdown("---")
        st.header("📊 Corpus Statistics")
        stats = search.get_statistics()
        st.metric("Documents", stats['num_documents'])
        st.metric("Vocabulary Size", stats['vocabulary_size'])
        st.metric("Avg Tokens/Doc", stats['avg_tokens_per_doc'])
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Search", "📄 Documents", "❓ Q&A", "📈 Analytics"])
    
    # Tab 1: Search
    with tab1:
        st.header("Search Documents")
        
        query = st.text_input(
            "Enter your query:",
            placeholder="e.g., jadwal pendaftaran mahasiswa baru, beasiswa, tata tertib",
            help="Search for information in the SmartCampus document corpus"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            search_button = st.button("🔍 Search", type="primary")
        
        if search_button and query:
            with st.spinner('Searching...'):
                results = search.search(
                    query,
                    mode=search_mode,
                    top_k=top_k,
                    include_sentiment=include_sentiment,
                    include_summary=include_summary
                )
            
            if 'error' in results:
                st.error(results['error'])
            else:
                # Query analysis
                with st.expander("Query Analysis", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Tokens", len(results.get('query_tokens', [])))
                    with col2:
                        if 'query_sentiment' in results:
                            st.markdown("**Query Sentiment:**")
                            st.markdown(format_sentiment(results['query_sentiment']), unsafe_allow_html=True)
                    with col3:
                        st.metric("Results Found", results.get('num_results', 0))
                
                # Results
                st.subheader(f"📋 Search Results ({results.get('num_results', 0)} found)")
                
                for i, result in enumerate(results.get('results', []), 1):
                    with st.container():
                        st.markdown(f"### {i}. {result['doc_name']}")
                        
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            if 'score' in result:
                                st.progress(result['score'])
                                st.caption(f"Relevance Score: {result['score']:.4f}")
                        with col2:
                            if 'sentiment' in result:
                                st.markdown("**Sentiment:**")
                                st.markdown(format_sentiment(result['sentiment']), unsafe_allow_html=True)
                        
                        if 'summary' in result and result['summary']:
                            st.markdown("**Summary:**")
                            st.info(result['summary'])
                        
                        st.markdown("---")
    
    # Tab 2: Documents
    with tab2:
        st.header("Browse Documents")
        
        doc_names = stats['document_names']
        selected_doc = st.selectbox("Select a document:", doc_names)
        
        if selected_doc:
            with st.spinner('Loading document...'):
                doc_info = search.get_document(selected_doc)
            
            if 'error' not in doc_info:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Tokens", doc_info['num_tokens'])
                with col2:
                    st.metric("Unique Tokens", doc_info['unique_tokens'])
                with col3:
                    st.markdown("**Sentiment:**")
                    st.markdown(format_sentiment(doc_info['sentiment']), unsafe_allow_html=True)
                
                st.subheader("📝 Summary")
                st.info(doc_info['summary'])
                
                with st.expander("📄 Full Text", expanded=False):
                    st.text(doc_info['text'])
    
    # Tab 3: Q&A
    with tab3:
        st.header("Ask a Question")
        st.markdown("Ask questions and get answers based on the document corpus.")
        
        question = st.text_area(
            "Your Question:",
            placeholder="e.g., Bagaimana cara mendaftar beasiswa? Apa saja fasilitas di kampus?",
            height=100
        )
        
        context_size = st.slider("Context Documents", 1, 5, 3)
        
        if st.button("💬 Get Answer", type="primary"):
            if question:
                with st.spinner('Generating answer...'):
                    answer = search.ask(question, context_size=context_size)
                
                st.subheader("📝 Answer")
                st.success(answer)
            else:
                st.warning("Please enter a question first.")
    
    # Tab 4: Analytics
    with tab4:
        st.header("Document Analytics")
        
        # Compare documents
        st.subheader("📊 Compare Documents")
        col1, col2 = st.columns(2)
        
        with col1:
            doc1 = st.selectbox("Document 1:", stats['document_names'], key='doc1')
        with col2:
            doc2 = st.selectbox("Document 2:", stats['document_names'], key='doc2')
        
        if st.button("Compare"):
            if doc1 != doc2:
                with st.spinner('Comparing...'):
                    comparison = search.compare_documents(doc1, doc2)
                
                if 'error' not in comparison:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Common Tokens", comparison['common_tokens'])
                    with col2:
                        st.metric("Jaccard Similarity", f"{comparison['jaccard_similarity']:.3f}")
                    with col3:
                        sim_percent = comparison['jaccard_similarity'] * 100
                        st.progress(comparison['jaccard_similarity'])
                        st.caption(f"{sim_percent:.1f}% similar")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**{doc1} Sentiment:**")
                        st.markdown(format_sentiment(comparison['doc1_sentiment']), unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"**{doc2} Sentiment:**")
                        st.markdown(format_sentiment(comparison['doc2_sentiment']), unsafe_allow_html=True)
            else:
                st.warning("Please select two different documents.")
        
        # Query analyzer
        st.subheader("🔬 Query Analyzer")
        test_query = st.text_input("Analyze a query:", placeholder="Enter query to analyze")
        
        if test_query:
            analysis = search.analyze_query(test_query)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Tokens", analysis['num_tokens'])
            with col2:
                st.markdown("**Sentiment:**")
                st.markdown(format_sentiment(analysis['sentiment']), unsafe_allow_html=True)
            with col3:
                st.metric("Suggested Mode", analysis['suggested_mode'])
            
            st.markdown("**Processed Tokens:**")
            st.code(', '.join(analysis['tokens']))
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "SmartCampus Assistant | Powered by RAG & TF-IDF"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == '__main__':
    main()
