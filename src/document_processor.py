import os
import PyPDF2
from docx import Document
from typing import List, Dict, Any
from pathlib import Path
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
from config import CHUNK_SIZE, CHUNK_OVERLAP, UPLOADS_DIR

class DocumentProcessor:
    def __init__(self):
        # Multiple chunking strategies
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Sentence-aware chunking for better context preservation
        try:
            from langchain.text_splitter import SentenceTransformersTokenTextSplitter
            self.sentence_splitter = SentenceTransformersTokenTextSplitter(
                chunk_overlap=50,
                model_name="all-MiniLM-L6-v2"
            )
            self.use_sentence_splitting = True
        except:
            self.use_sentence_splitting = False
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file with enhanced structure preservation"""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    # Clean up the text while preserving structure
                    page_text = self._clean_pdf_text(page_text)
                    text += f"\n=== PAGE {page_num + 1} ===\n{page_text}\n"
            return text
        except Exception as e:
            st.error(f"Error reading PDF {file_path}: {str(e)}")
            return ""
    
    def _clean_pdf_text(self, text: str) -> str:
        """Clean PDF text while preserving structure"""
        # Remove excessive whitespace but preserve paragraph breaks
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        # Fix common PDF extraction issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between joined words
        text = re.sub(r'(\d+)([A-Za-z])', r'\1 \2', text)  # Space between numbers and letters
        # Remove excessive spaces
        text = re.sub(r' +', ' ', text)
        return text.strip()
    
    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file with structure preservation"""
        try:
            doc = Document(file_path)
            text_parts = []
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    # Detect headings and structure
                    if paragraph.style.name.startswith('Heading'):
                        text_parts.append(f"\n# {paragraph.text}\n")
                    else:
                        text_parts.append(paragraph.text)
            
            return '\n'.join(text_parts)
        except Exception as e:
            st.error(f"Error reading DOCX {file_path}: {str(e)}")
            return ""
    
    def extract_text_from_txt(self, file_path: str) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    return file.read()
            except Exception as e:
                st.error(f"Error reading TXT {file_path}: {str(e)}")
                return ""
        except Exception as e:
            st.error(f"Error reading TXT {file_path}: {str(e)}")
            return ""
    
    def process_uploaded_file(self, uploaded_file) -> str:
        """Process uploaded file and extract text"""
        # Save uploaded file
        file_path = UPLOADS_DIR / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Extract text based on file type
        file_extension = uploaded_file.name.lower().split('.')[-1]
        
        if file_extension == 'pdf':
            text = self.extract_text_from_pdf(str(file_path))
        elif file_extension == 'docx':
            text = self.extract_text_from_docx(str(file_path))
        elif file_extension in ['txt', 'md']:
            text = self.extract_text_from_txt(str(file_path))
        else:
            st.error(f"Unsupported file type: {file_extension}")
            return ""
        
        return text
    
    def _detect_document_type(self, text: str, filename: str) -> str:
        """Detect document type for adaptive chunking"""
        text_lower = text.lower()
        filename_lower = filename.lower()
        
        # CV/Resume detection
        if any(keyword in text_lower for keyword in ['resume', 'cv', 'curriculum vitae', 'experience', 'education', 'skills']):
            return "resume"
        
        # Research paper detection
        elif any(keyword in text_lower for keyword in ['abstract', 'introduction', 'methodology', 'conclusion', 'references']):
            return "research"
        
        # Code documentation
        elif any(keyword in text_lower for keyword in ['api', 'function', 'class', 'method', 'documentation']):
            return "documentation"
        
        # Meeting notes
        elif any(keyword in text_lower for keyword in ['meeting', 'agenda', 'action items', 'attendees']):
            return "meeting_notes"
        
        return "general"
    
    def _adaptive_chunk_size(self, doc_type: str) -> tuple:
        """Return optimal chunk size and overlap for document type"""
        chunk_configs = {
            "resume": (800, 100),        # Smaller chunks for structured data
            "research": (1200, 200),     # Larger chunks to preserve context
            "documentation": (1000, 150), # Medium chunks for code docs
            "meeting_notes": (600, 100),  # Small chunks for discrete topics
            "general": (1000, 200)       # Default
        }
        return chunk_configs.get(doc_type, (1000, 200))
    
    def chunk_text(self, text: str, filename: str) -> List[Dict[str, Any]]:
        """Enhanced chunking with adaptive strategies"""
        if not text.strip():
            return []
        
        # Detect document type
        doc_type = self._detect_document_type(text, filename)
        chunk_size, chunk_overlap = self._adaptive_chunk_size(doc_type)
        
        # Create adaptive text splitter
        adaptive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=self._get_separators_for_type(doc_type)
        )
        
        # Split text into chunks
        if self.use_sentence_splitting and doc_type in ["research", "documentation"]:
            # Use sentence-aware splitting for academic/technical content
            try:
                chunks = self.sentence_splitter.split_text(text)
            except:
                chunks = adaptive_splitter.split_text(text)
        else:
            chunks = adaptive_splitter.split_text(text)
        
        # Create chunk objects with enhanced metadata
        chunk_objects = []
        for i, chunk in enumerate(chunks):
            # Extract section information
            section_info = self._extract_section_info(chunk, doc_type)
            
            chunk_objects.append({
                "content": chunk,
                "metadata": {
                    "filename": filename,
                    "chunk_id": i,
                    "total_chunks": len(chunks),
                    "doc_type": doc_type,
                    "chunk_size": len(chunk),
                    "section": section_info.get("section", "unknown"),
                    "has_dates": self._contains_dates(chunk),
                    "has_numbers": self._contains_numbers(chunk),
                    "estimated_topic": self._estimate_topic(chunk)
                }
            })
        
        return chunk_objects
    
    def _get_separators_for_type(self, doc_type: str) -> List[str]:
        """Get document-type specific separators"""
        separators = {
            "resume": ["\n\n", "\n•", "\n-", "\n", " "],
            "research": ["\n\n", "\n", ". ", " "],
            "documentation": ["\n\n", "\n```", "\n", " "],
            "meeting_notes": ["\n\n", "\n•", "\n-", "\n", " "],
            "general": ["\n\n", "\n", " ", ""]
        }
        return separators.get(doc_type, ["\n\n", "\n", " ", ""])
    
    def _extract_section_info(self, chunk: str, doc_type: str) -> Dict[str, Any]:
        """Extract section information from chunk"""
        chunk_lower = chunk.lower()
        
        if doc_type == "resume":
            if any(keyword in chunk_lower for keyword in ['education', 'degree', 'university']):
                return {"section": "education"}
            elif any(keyword in chunk_lower for keyword in ['experience', 'work', 'internship']):
                return {"section": "experience"}
            elif any(keyword in chunk_lower for keyword in ['skills', 'technologies', 'programming']):
                return {"section": "skills"}
            elif any(keyword in chunk_lower for keyword in ['project', 'built', 'developed']):
                return {"section": "projects"}
        
        elif doc_type == "research":
            if any(keyword in chunk_lower for keyword in ['abstract', 'summary']):
                return {"section": "abstract"}
            elif any(keyword in chunk_lower for keyword in ['introduction', 'background']):
                return {"section": "introduction"}
            elif any(keyword in chunk_lower for keyword in ['method', 'approach', 'implementation']):
                return {"section": "methodology"}
            elif any(keyword in chunk_lower for keyword in ['result', 'finding', 'outcome']):
                return {"section": "results"}
        
        return {"section": "content"}
    
    def _contains_dates(self, text: str) -> bool:
        """Check if chunk contains date information"""
        date_patterns = [
            r'\d{4}',  # Year
            r'\d{1,2}/\d{1,2}/\d{4}',  # MM/DD/YYYY
            r'\d{1,2}-\d{1,2}-\d{4}',  # MM-DD-YYYY
            r'(January|February|March|April|May|June|July|August|September|October|November|December)',
            r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in date_patterns)
    
    def _contains_numbers(self, text: str) -> bool:
        """Check if chunk contains numerical data"""
        number_patterns = [
            r'\d+%',  # Percentages
            r'\$\d+',  # Money
            r'\d+\.\d+',  # Decimals
            r'\d{3,}'  # Large numbers
        ]
        return any(re.search(pattern, text) for pattern in number_patterns)
    
    def _estimate_topic(self, chunk: str) -> str:
        """Estimate the main topic of the chunk"""
        chunk_lower = chunk.lower()
        
        # Technical topics
        if any(keyword in chunk_lower for keyword in ['python', 'javascript', 'programming', 'code', 'algorithm']):
            return "technical"
        elif any(keyword in chunk_lower for keyword in ['machine learning', 'ai', 'data science', 'ml']):
            return "ai_ml"
        elif any(keyword in chunk_lower for keyword in ['project', 'developed', 'built', 'implemented']):
            return "projects"
        elif any(keyword in chunk_lower for keyword in ['experience', 'work', 'internship', 'job']):
            return "experience"
        elif any(keyword in chunk_lower for keyword in ['education', 'university', 'degree', 'school']):
            return "education"
        elif any(keyword in chunk_lower for keyword in ['skill', 'technology', 'tool', 'framework']):
            return "skills"
        else:
            return "general"
    
    def get_uploaded_files(self) -> List[str]:
        """Get list of uploaded files"""
        if not UPLOADS_DIR.exists():
            return []
        
        files = []
        for file_path in UPLOADS_DIR.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in ['.pdf', '.docx', '.txt', '.md']:
                files.append(file_path.name)
        
        return sorted(files)
    
    def get_document_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about processed document"""
        if not chunks:
            return {}
        
        total_chars = sum(len(chunk["content"]) for chunk in chunks)
        avg_chunk_size = total_chars / len(chunks)
        
        # Count topics
        topics = [chunk["metadata"]["estimated_topic"] for chunk in chunks]
        topic_counts = {topic: topics.count(topic) for topic in set(topics)}
        
        # Count sections
        sections = [chunk["metadata"]["section"] for chunk in chunks]
        section_counts = {section: sections.count(section) for section in set(sections)}
        
        return {
            "total_chunks": len(chunks),
            "total_characters": total_chars,
            "avg_chunk_size": int(avg_chunk_size),
            "doc_type": chunks[0]["metadata"]["doc_type"],
            "topic_distribution": topic_counts,
            "section_distribution": section_counts,
            "has_dates": sum(1 for chunk in chunks if chunk["metadata"]["has_dates"]),
            "has_numbers": sum(1 for chunk in chunks if chunk["metadata"]["has_numbers"])
        }