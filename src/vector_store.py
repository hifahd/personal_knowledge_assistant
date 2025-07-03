import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
import streamlit as st
from config import CHROMA_DB_DIR, EMBEDDING_MODEL, MAX_RETRIEVED_CHUNKS
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Try to import sentence-transformers, fallback to basic embeddings if it fails
try:
    from sentence_transformers import SentenceTransformer
    USE_SENTENCE_TRANSFORMERS = True
except ImportError:
    USE_SENTENCE_TRANSFORMERS = False

class VectorStore:
    def __init__(self):
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DB_DIR),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize embedding model
        if USE_SENTENCE_TRANSFORMERS:
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            self.use_sentence_transformers = True
        else:
            # Fallback to TF-IDF for basic embeddings
            self.embedding_model = TfidfVectorizer(max_features=384, stop_words='english')
            self.use_sentence_transformers = False
            st.warning("Using basic TF-IDF embeddings instead of SentenceTransformers")
        
        # Initialize TF-IDF for keyword search (always used for hybrid search)
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams
            lowercase=True
        )
        self.tfidf_fitted = False
        self.document_texts = []  # Store for TF-IDF
        
        # Get or create collection
        self.collection_name = "personal_documents"
        try:
            self.collection = self.client.get_collection(self.collection_name)
        except Exception:  # Catch any exception when collection doesn't exist
            self.collection = self.client.create_collection(self.collection_name)
    
    def add_documents(self, chunks: List[Dict[str, Any]]) -> bool:
        """Add document chunks to vector store"""
        try:
            if not chunks:
                return False
            
            # Prepare data for ChromaDB
            documents = []
            metadatas = []
            ids = []
            
            for chunk in chunks:
                content = chunk["content"]
                metadata = chunk["metadata"]
                
                # Create unique ID
                chunk_id = f"{metadata['filename']}_{metadata['chunk_id']}"
                
                documents.append(content)
                metadatas.append(metadata)
                ids.append(chunk_id)
            
            # Generate embeddings
            if self.use_sentence_transformers:
                embeddings = self.embedding_model.encode(documents).tolist()
            else:
                # Use TF-IDF embeddings
                tfidf_matrix = self.embedding_model.fit_transform(documents)
                embeddings = tfidf_matrix.toarray().tolist()
            
            # Add to ChromaDB
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            # Update TF-IDF index for hybrid search
            self._update_tfidf_index()
            
            return True
            
        except Exception as e:
            st.error(f"Error adding documents to vector store: {str(e)}")
            return False
    
    def _update_tfidf_index(self):
        """Update TF-IDF index with all documents"""
        try:
            # Get all documents from ChromaDB
            all_docs = self.collection.get()
            if all_docs['documents']:
                self.document_texts = all_docs['documents']
                self.tfidf_vectorizer.fit(self.document_texts)
                self.tfidf_fitted = True
        except Exception as e:
            st.warning(f"Could not update TF-IDF index: {str(e)}")

    def _keyword_search(self, query: str, n_results: int = MAX_RETRIEVED_CHUNKS) -> List[Dict[str, Any]]:
        """Perform keyword-based search using TF-IDF"""
        if not self.tfidf_fitted or not self.document_texts:
            return []
        
        try:
            # Transform query and documents
            query_vec = self.tfidf_vectorizer.transform([query])
            doc_vecs = self.tfidf_vectorizer.transform(self.document_texts)
            
            # Calculate similarities
            similarities = cosine_similarity(query_vec, doc_vecs).flatten()
            
            # Get top results
            top_indices = np.argsort(similarities)[::-1][:n_results]
            
            # Get document details
            all_docs = self.collection.get()
            results = []
            
            for idx in top_indices:
                if similarities[idx] > 0:  # Only include non-zero similarities
                    results.append({
                        "content": all_docs['documents'][idx],
                        "metadata": all_docs['metadatas'][idx],
                        "score": float(similarities[idx]),
                        "search_type": "keyword"
                    })
            
            return results
            
        except Exception as e:
            st.warning(f"Keyword search failed: {str(e)}")
            return []

    def similarity_search(self, query: str, n_results: int = MAX_RETRIEVED_CHUNKS, 
                         search_type: str = "hybrid") -> List[Dict[str, Any]]:
        """Enhanced search with multiple strategies"""
        
        if search_type == "semantic":
            return self._semantic_search(query, n_results)
        elif search_type == "keyword":
            return self._keyword_search(query, n_results)
        else:  # hybrid
            return self._hybrid_search(query, n_results)

    def _semantic_search(self, query: str, n_results: int = MAX_RETRIEVED_CHUNKS) -> List[Dict[str, Any]]:
        """Original semantic search"""
        try:
            # Generate query embedding
            if self.use_sentence_transformers:
                query_embedding = self.embedding_model.encode([query]).tolist()
            else:
                # Use TF-IDF for query
                query_tfidf = self.embedding_model.transform([query])
                query_embedding = query_tfidf.toarray().tolist()
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=n_results
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    formatted_results.append({
                        "content": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i],
                        "distance": results['distances'][0][i] if results['distances'] else None,
                        "search_type": "semantic"
                    })
            
            return formatted_results
            
        except Exception as e:
            st.error(f"Error in semantic search: {str(e)}")
            return []

    def _hybrid_search(self, query: str, n_results: int = MAX_RETRIEVED_CHUNKS) -> List[Dict[str, Any]]:
        """Combine semantic and keyword search results"""
        try:
            # Get results from both methods
            semantic_results = self._semantic_search(query, n_results * 2)  # Get more for merging
            keyword_results = self._keyword_search(query, n_results * 2)
            
            # Combine and deduplicate results
            combined_results = {}
            
            # Add semantic results with weight
            for result in semantic_results:
                doc_id = f"{result['metadata']['filename']}_{result['metadata']['chunk_id']}"
                semantic_score = 1 - (result.get('distance', 1) or 1)  # Convert distance to similarity
                combined_results[doc_id] = {
                    **result,
                    'semantic_score': semantic_score,
                    'keyword_score': 0,
                    'search_type': 'hybrid'
                }
            
            # Add keyword results with weight
            for result in keyword_results:
                doc_id = f"{result['metadata']['filename']}_{result['metadata']['chunk_id']}"
                keyword_score = result.get('score', 0)
                
                if doc_id in combined_results:
                    combined_results[doc_id]['keyword_score'] = keyword_score
                else:
                    combined_results[doc_id] = {
                        **result,
                        'semantic_score': 0,
                        'keyword_score': keyword_score,
                        'search_type': 'hybrid'
                    }
            
            # Calculate hybrid scores and sort
            for doc_id, result in combined_results.items():
                # Weighted combination (70% semantic, 30% keyword)
                hybrid_score = (0.7 * result['semantic_score']) + (0.3 * result['keyword_score'])
                result['hybrid_score'] = hybrid_score
            
            # Sort by hybrid score and return top results
            sorted_results = sorted(
                combined_results.values(), 
                key=lambda x: x['hybrid_score'], 
                reverse=True
            )
            
            return sorted_results[:n_results]
            
        except Exception as e:
            st.error(f"Error in hybrid search: {str(e)}")
            # Fallback to semantic search
            return self._semantic_search(query, n_results)
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            count = self.collection.count()
            return {
                "total_chunks": count,
                "collection_name": self.collection_name
            }
        except Exception as e:
            st.error(f"Error getting collection stats: {str(e)}")
            return {"total_chunks": 0, "collection_name": self.collection_name}
    
    def delete_documents_by_filename(self, filename: str) -> bool:
        """Delete all chunks from a specific file"""
        try:
            # Get all documents
            all_results = self.collection.get()
            
            # Find IDs to delete
            ids_to_delete = []
            for i, metadata in enumerate(all_results['metadatas']):
                if metadata.get('filename') == filename:
                    ids_to_delete.append(all_results['ids'][i])
            
            # Delete the documents
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                return True
            
            return False
            
        except Exception as e:
            st.error(f"Error deleting documents: {str(e)}")
            return False
    
    def clear_collection(self) -> bool:
        """Clear all documents from the collection"""
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(self.collection_name)
            return True
        except Exception as e:
            st.error(f"Error clearing collection: {str(e)}")
            return False