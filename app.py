import streamlit as st
import os
from pathlib import Path

# Add src directory to path
import sys
sys.path.append(str(Path(__file__).parent / "src"))

from document_processor import DocumentProcessor
from vector_store import VectorStore
from qa_chain import QAChain
from conversation_manager import ConversationManager
from config import PAGE_TITLE, PAGE_ICON, UPLOADS_DIR

# Page configuration
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide"
)

# Initialize components
@st.cache_resource
def initialize_components():
    """Initialize all components (cached to avoid re-initialization)"""
    doc_processor = DocumentProcessor()
    vector_store = VectorStore()
    qa_chain = QAChain()
    return doc_processor, vector_store, qa_chain

def main():
    st.title("🧠 Advanced Personal Knowledge Assistant")
    st.markdown("*Enhanced with Hybrid Search, Conversation Memory & Adaptive Chunking*")
    
    # Initialize components
    doc_processor, vector_store, qa_chain = initialize_components()
    
    # Initialize conversation manager
    if 'conversation_manager' not in st.session_state:
        st.session_state.conversation_manager = ConversationManager()
    
    conversation_manager = st.session_state.conversation_manager
    
    # Sidebar for document management
    with st.sidebar:
        st.header("📁 Document Management")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload your documents",
            type=['pdf', 'docx', 'txt', 'md'],
            accept_multiple_files=True,
            help="Supported formats: PDF, DOCX, TXT, MD"
        )
        
        if uploaded_files:
            if st.button("Process Documents", type="primary"):
                with st.spinner("Processing documents with adaptive chunking..."):
                    success_count = 0
                    total_stats = {"total_chunks": 0, "doc_types": [], "topics": []}
                    
                    for uploaded_file in uploaded_files:
                        # Extract text
                        text = doc_processor.process_uploaded_file(uploaded_file)
                        
                        if text:
                            # Create chunks with adaptive strategy
                            chunks = doc_processor.chunk_text(text, uploaded_file.name)
                            
                            if chunks:
                                # Get document statistics
                                doc_stats = doc_processor.get_document_stats(chunks)
                                
                                # Add to vector store
                                if vector_store.add_documents(chunks):
                                    success_count += 1
                                    st.success(f"✅ {uploaded_file.name} processed successfully!")
                                    
                                    # Show document insights
                                    with st.expander(f"📊 Document Insights: {uploaded_file.name}"):
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.metric("Chunks Created", doc_stats.get("total_chunks", 0))
                                            st.metric("Avg Chunk Size", doc_stats.get("avg_chunk_size", 0))
                                        with col2:
                                            st.write("**Document Type:**", doc_stats.get("doc_type", "unknown"))
                                            st.write("**Topics Found:**", ", ".join(doc_stats.get("topic_distribution", {}).keys()))
                                    
                                    # Aggregate stats
                                    total_stats["total_chunks"] += doc_stats.get("total_chunks", 0)
                                    total_stats["doc_types"].append(doc_stats.get("doc_type", "unknown"))
                                    total_stats["topics"].extend(doc_stats.get("topic_distribution", {}).keys())
                                else:
                                    st.error(f"❌ Failed to process {uploaded_file.name}")
                            else:
                                st.warning(f"⚠️ No content extracted from {uploaded_file.name}")
                        else:
                            st.error(f"❌ Could not read {uploaded_file.name}")
                    
                    if success_count > 0:
                        st.success(f"🎉 Successfully processed {success_count} documents!")
                        st.balloons()
                        st.rerun()
        
        # Display collection stats
        st.subheader("📊 Collection Stats")
        stats = vector_store.get_collection_stats()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Chunks", stats["total_chunks"])
        with col2:
            if stats["total_chunks"] > 0:
                st.metric("Vector Store", "✅ Ready")
            else:
                st.metric("Vector Store", "Empty")
        
        # Display uploaded files
        st.subheader("📄 Uploaded Files")
        uploaded_files_list = doc_processor.get_uploaded_files()
        if uploaded_files_list:
            for filename in uploaded_files_list:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(filename)
                with col2:
                    if st.button("🗑️", key=f"delete_{filename}", help=f"Delete {filename}"):
                        if vector_store.delete_documents_by_filename(filename):
                            # Also delete the physical file
                            file_path = UPLOADS_DIR / filename
                            if file_path.exists():
                                file_path.unlink()
                            st.success(f"Deleted {filename}")
                            st.rerun()
        else:
            st.info("No files uploaded yet")
        
        # Conversation management
        st.subheader("💬 Conversation")
        if st.button("Clear History"):
            conversation_manager.clear_history()
            st.success("Conversation history cleared!")
            st.rerun()
        
        # Show conversation summary
        conv_summary = conversation_manager.get_conversation_summary()
        if conv_summary != "No conversation history":
            st.info(conv_summary)
        
        # Clear all button
        if st.button("🗑️ Clear All Documents", type="secondary"):
            if vector_store.clear_collection():
                # Clear uploads directory
                for file_path in UPLOADS_DIR.glob("*"):
                    if file_path.is_file():
                        file_path.unlink()
                # Clear conversation history
                conversation_manager.clear_history()
                st.success("All documents and history cleared!")
                st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Ask Questions")
        
        # Check if we have documents
        stats = vector_store.get_collection_stats()
        if stats["total_chunks"] == 0:
            st.info("👈 Please upload some documents first to start asking questions!")
        else:
            # Search method selection
            search_method = st.selectbox(
                "Search Method:",
                ["hybrid", "semantic", "keyword"],
                index=0,
                help="Hybrid combines semantic and keyword search for best results"
            )
            
            # Question input
            question = st.text_input(
                "What would you like to know?",
                placeholder="e.g., What did I learn about machine learning in my research?",
                help="Ask questions about your uploaded documents. The system will remember our conversation!"
            )
            
            if st.button("🔍 Ask", type="primary") and question:
                with st.spinner("Searching and generating answer..."):
                    # Test Ollama connection first
                    if not qa_chain.test_ollama_connection():
                        st.error("❌ Cannot connect to Ollama. Please make sure it's running.")
                        st.info("Run: `ollama serve` in your terminal")
                        return
                    
                    # Get conversation context for follow-up questions
                    enhanced_query = conversation_manager.get_enhanced_query(question)
                    conversation_context = conversation_manager.get_conversation_context(question)
                    
                    # Search for relevant chunks using selected method
                    relevant_chunks = vector_store.similarity_search(enhanced_query, search_type=search_method)
                    
                    if not relevant_chunks:
                        st.warning("🤔 No relevant information found. Try rephrasing your question or using a different search method.")
                    else:
                        # Generate answer with conversation context - PROPERLY STRUCTURED CALL
                        result = qa_chain.generate_answer(
                            question=question, 
                            context_chunks=relevant_chunks, 
                            conversation_context=conversation_context
                        )
                        
                        if result["error"]:
                            st.error(f"Error: {result['error']}")
                        else:
                            # Add to conversation history
                            conversation_manager.add_exchange(question, result["answer"], result["sources"])
                            
                            # Display answer
                            st.subheader("💡 Answer")
                            st.write(result["answer"])
                            
                            # Display search information
                            search_info = result.get("search_info", {})
                            if search_info:
                                with st.expander("🔍 Search Details", expanded=False):
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.write(f"**Chunks Found:** {search_info.get('total_chunks', 0)}")
                                        st.write(f"**Search Methods:** {', '.join(search_info.get('search_methods', []))}")
                                    with col2:
                                        st.write(f"**Sections:** {', '.join(search_info.get('sections_found', []))}")
                                        st.write(f"**Topics:** {', '.join(search_info.get('topics_found', []))}")
                            
                            # Display sources with enhanced information
                            with st.expander("📚 Sources", expanded=True):
                                for i, source in enumerate(result["sources"], 1):
                                    st.write(f"**Source {i}: {source['filename']}**")
                                    
                                    # Show enhanced metadata
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.write(f"Section: {source.get('section', 'unknown')}")
                                    with col2:
                                        st.write(f"Topic: {source.get('estimated_topic', 'general')}")
                                    with col3:
                                        st.write(f"Search: {source.get('search_type', 'unknown')}")
                                    
                                    # Show search scores for hybrid results
                                    if source.get('search_type') == 'hybrid':
                                        st.write(f"🔍 Semantic: {source.get('semantic_score', 0):.3f} | Keyword: {source.get('keyword_score', 0):.3f} | Combined: {source.get('hybrid_score', 0):.3f}")
                                    
                                    st.write(f"Preview: {source['content_preview']}")
                                    st.write("---")
        
        # Display conversation history
        if conversation_manager.get_history():
            st.subheader("📜 Conversation History")
            history = conversation_manager.get_history()
            
            for i, exchange in enumerate(reversed(history[-3:]), 1):  # Show last 3 exchanges
                with st.expander(f"Q{len(history)-i+1}: {exchange['question'][:50]}...", expanded=False):
                    st.write(f"**Question:** {exchange['question']}")
                    st.write(f"**Answer:** {exchange['answer'][:300]}...")
                    st.write(f"**Time:** {exchange['timestamp'][:19]}")
    
    with col2:
        st.header("ℹ️ How to Use")
        st.markdown("""
        **🚀 New Advanced Features:**
        - **Hybrid Search**: Combines semantic + keyword search
        - **Conversation Memory**: Remembers previous questions
        - **Adaptive Chunking**: Smart document processing
        
        **📝 Steps:**
        1. **Upload Documents**: PDF, DOCX, or TXT files
        2. **Process**: Click "Process Documents" 
        3. **Ask Questions**: Natural language queries
        4. **Follow Up**: Ask related questions naturally
        
        **💡 Tips:**
        - Try different search methods for better results
        - Ask follow-up questions - the system remembers context!
        - Check document insights after uploading
        """)
        
        # System status
        st.subheader("🔧 System Status")
        
        # Ollama status
        if qa_chain.test_ollama_connection():
            st.success("✅ Ollama Connected")
        else:
            st.error("❌ Ollama Disconnected")
            st.info("Make sure Ollama is running: `ollama serve`")
        
        # Vector store status
        if stats["total_chunks"] > 0:
            st.success(f"✅ Vector Store: {stats['total_chunks']} chunks")
        else:
            st.info("ℹ️ Vector Store: Empty")
        
        # Advanced features status
        st.success("✅ Hybrid Search: Active")
        st.success("✅ Conversation Memory: Active") 
        st.success("✅ Adaptive Chunking: Active")

if __name__ == "__main__":
    main()