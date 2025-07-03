from typing import List, Dict, Any, Optional
import streamlit as st
from datetime import datetime
import json

class ConversationManager:
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.conversation_key = "conversation_history"
        
        # Initialize session state for conversation history
        if self.conversation_key not in st.session_state:
            st.session_state[self.conversation_key] = []
    
    def add_exchange(self, question: str, answer: str, sources: List[Dict[str, Any]]):
        """Add a Q&A exchange to conversation history"""
        exchange = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "sources": sources,
            "id": len(st.session_state[self.conversation_key])
        }
        
        # Add to history
        st.session_state[self.conversation_key].append(exchange)
        
        # Keep only recent exchanges
        if len(st.session_state[self.conversation_key]) > self.max_history:
            st.session_state[self.conversation_key] = st.session_state[self.conversation_key][-self.max_history:]
    
    def get_conversation_context(self, current_question: str, max_context: int = 3) -> str:
        """Get recent conversation context for follow-up questions"""
        history = st.session_state[self.conversation_key]
        
        if not history:
            return ""
        
        # Get recent exchanges
        recent_exchanges = history[-max_context:]
        
        # Check if current question is a follow-up
        if self._is_followup_question(current_question):
            context_parts = []
            for exchange in recent_exchanges:
                context_parts.append(f"Previous Q: {exchange['question']}")
                context_parts.append(f"Previous A: {exchange['answer'][:200]}...")  # Truncate for context
            
            return "\n".join(context_parts)
        
        return ""
    
    def _is_followup_question(self, question: str) -> bool:
        """Detect if this is a follow-up question"""
        followup_indicators = [
            "what about", "tell me more", "can you explain", "what else",
            "also", "additionally", "furthermore", "elaborate", "expand",
            "what other", "any other", "more details", "continue", "go on",
            "that", "this", "it", "them", "those", "these"  # Pronouns indicating reference
        ]
        
        question_lower = question.lower()
        return any(indicator in question_lower for indicator in followup_indicators)
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get full conversation history"""
        return st.session_state[self.conversation_key]
    
    def clear_history(self):
        """Clear conversation history"""
        st.session_state[self.conversation_key] = []
    
    def get_enhanced_query(self, current_question: str) -> str:
        """Enhance current question with conversation context"""
        context = self.get_conversation_context(current_question)
        
        if context:
            enhanced_query = f"""
            Previous conversation context:
            {context}
            
            Current question: {current_question}
            
            Please answer the current question taking into account the previous conversation context.
            """
            return enhanced_query
        
        return current_question
    
    def export_conversation(self) -> str:
        """Export conversation as JSON string"""
        return json.dumps(st.session_state[self.conversation_key], indent=2)
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation topics"""
        history = st.session_state[self.conversation_key]
        if not history:
            return "No conversation history"
        
        topics = []
        for exchange in history[-5:]:  # Last 5 exchanges
            # Extract key topics from questions
            question = exchange['question'].lower()
            if any(keyword in question for keyword in ['machine learning', 'ml', 'ai']):
                topics.append("Machine Learning")
            elif any(keyword in question for keyword in ['project', 'work', 'experience']):
                topics.append("Projects & Experience")
            elif any(keyword in question for keyword in ['skill', 'technology', 'programming']):
                topics.append("Technical Skills")
            else:
                topics.append("General")
        
        unique_topics = list(set(topics))
        return f"Recent topics: {', '.join(unique_topics)}"