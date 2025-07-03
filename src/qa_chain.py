import ollama
from typing import List, Dict, Any
import streamlit as st
from config import LLM_MODEL

class QAChain:
    def __init__(self):
        self.model = LLM_MODEL
        # Use the client method since our debug showed it works
        self.client = ollama.Client(host='http://localhost:11434')
    
    def create_prompt(self, question: str, context_chunks: List[Dict[str, Any]], 
                     conversation_context: str = "") -> str:
        """Create an enhanced prompt with context and conversation history"""
        
        # Build context from retrieved chunks with enhanced metadata
        context_parts = []
        sources = []
        
        for i, chunk in enumerate(context_chunks, 1):
            content = chunk["content"]
            metadata = chunk["metadata"]
            filename = metadata.get("filename", "Unknown")
            section = metadata.get("section", "unknown")
            doc_type = metadata.get("doc_type", "general")
            search_type = chunk.get("search_type", "unknown")
            
            # Add search method info
            search_info = ""
            if search_type == "hybrid":
                semantic_score = chunk.get('semantic_score', 0)
                keyword_score = chunk.get('keyword_score', 0)
                search_info = f" (Hybrid: {semantic_score:.2f}S + {keyword_score:.2f}K)"
            elif search_type == "semantic":
                search_info = " (Semantic search)"
            elif search_type == "keyword":
                search_info = " (Keyword search)"
            
            context_parts.append(f"""
--- Source {i}: {filename} | {section} section | {doc_type} document{search_info} ---
{content}
""")
            sources.append(f"Source {i}: {filename} ({section})")
        
        context = "\n".join(context_parts)
        
        # Enhanced prompt with conversation awareness
        prompt = f"""You are an advanced AI assistant that answers questions based on the user's personal documents and conversation history.

INSTRUCTIONS:
1. Answer the question using the provided context and conversation history
2. If this is a follow-up question, reference previous answers appropriately
3. Be specific and detailed, using information from the most relevant sources
4. Always cite sources by referring to source numbers (e.g., "According to Source 1...")
5. If the context doesn't contain enough information, say so clearly
6. Maintain conversation continuity when appropriate
7. Consider the document types and sections when prioritizing information

{f"PREVIOUS CONVERSATION CONTEXT:{conversation_context}" if conversation_context else ""}

CURRENT CONTEXT:
{context}

CURRENT QUESTION: {question}

AVAILABLE SOURCES:
{chr(10).join(sources)}

ANSWER:"""
        
        return prompt
    
    def generate_answer(self, question: str, context_chunks: List[Dict[str, Any]], 
                       conversation_context: str = "") -> Dict[str, Any]:
        """Generate answer using the LLM with conversation awareness"""
        try:
            if not context_chunks:
                return {
                    "answer": "I couldn't find any relevant information in your documents to answer this question. Please make sure you have uploaded relevant documents and try rephrasing your question.",
                    "sources": [],
                    "search_info": {"total_chunks": 0, "search_methods": []},
                    "error": None
                }
            
            # Create enhanced prompt
            prompt = self.create_prompt(question, context_chunks, conversation_context)
            
            # Generate response using Ollama client
            response = self.client.chat(
                model=self.model,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                options={
                    'temperature': 0.1,
                    'top_p': 0.9,
                    'top_k': 40
                }
            )
            
            answer = response['message']['content']
            
            # Extract enhanced sources with metadata
            sources = []
            search_methods = set()
            
            for chunk in context_chunks:
                metadata = chunk["metadata"]
                search_type = chunk.get("search_type", "unknown")
                search_methods.add(search_type)
                
                source_info = {
                    "filename": metadata.get("filename", "Unknown"),
                    "chunk_id": metadata.get("chunk_id", 0),
                    "section": metadata.get("section", "unknown"),
                    "doc_type": metadata.get("doc_type", "general"),
                    "estimated_topic": metadata.get("estimated_topic", "general"),
                    "search_type": search_type,
                    "content_preview": chunk["content"][:200] + "..." if len(chunk["content"]) > 200 else chunk["content"]
                }
                
                # Add search scores if available
                if search_type == "hybrid":
                    source_info["semantic_score"] = chunk.get('semantic_score', 0)
                    source_info["keyword_score"] = chunk.get('keyword_score', 0)
                    source_info["hybrid_score"] = chunk.get('hybrid_score', 0)
                
                sources.append(source_info)
            
            # Search information summary
            search_info = {
                "total_chunks": len(context_chunks),
                "search_methods": list(search_methods),
                "sections_found": list(set(s["section"] for s in sources)),
                "topics_found": list(set(s["estimated_topic"] for s in sources))
            }
            
            return {
                "answer": answer,
                "sources": sources,
                "search_info": search_info,
                "error": None
            }
            
        except Exception as e:
            error_msg = f"Error generating answer: {str(e)}"
            st.error(error_msg)
            return {
                "answer": "Sorry, I encountered an error while generating the answer. Please try again.",
                "sources": [],
                "search_info": {"total_chunks": 0, "search_methods": []},
                "error": error_msg
            }
    
    def test_ollama_connection(self) -> bool:
        """Test if Ollama is working"""
        try:
            # Test with the client method that we know works
            response = self.client.chat(
                model=self.model,
                messages=[
                    {
                        'role': 'user',
                        'content': 'Hello, this is a test. Please respond with "Test successful".'
                    }
                ]
            )
            return True
        except Exception as e:
            st.error(f"Ollama test failed: {str(e)}")
            return False