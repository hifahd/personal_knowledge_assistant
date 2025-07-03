# Advanced Personal Knowledge Assistant - RAG System

A production-ready Retrieval-Augmented Generation (RAG) system that allows users to upload personal documents and ask natural language questions to get accurate answers with source citations.

## Features

- **Hybrid Search**: Combines semantic and keyword search for enhanced retrieval accuracy
- **Conversation Memory**: Remembers previous questions and maintains context
- **Adaptive Chunking**: Smart document processing based on document type
- **Multi-format Support**: PDF, DOCX, TXT, and Markdown files
- **Local LLM**: Privacy-focused with local Ollama integration
- **Source Attribution**: Detailed citations with document insights
- **Real-time Processing**: Interactive web interface with Streamlit

## Tech Stack

- **Python 3.10+** - Core programming language
- **LangChain** - RAG framework
- **ChromaDB** - Local vector database
- **Sentence-Transformers** - Embedding models
- **Ollama + Llama 3.2** - Local LLM
- **Streamlit** - Web interface
- **Scikit-learn** - Machine learning utilities

## Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd personal_knowledge_assistant
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Install and setup Ollama**
```bash
# Download from https://ollama.ai
ollama pull llama3.2:1b
```

4. **Run the application**
```bash
streamlit run app.py
```

## Usage

1. **Upload Documents**: Use the sidebar to upload PDF, DOCX, or TXT files
2. **Process Documents**: Click "Process Documents" to add them to your knowledge base
3. **Ask Questions**: Type natural language questions about your documents
4. **Get Answers**: Receive detailed answers with source citations and search insights

## Project Structure

```
personal_knowledge_assistant/
├── app.py                      # Main Streamlit application
├── src/
│   ├── document_processor.py   # Document processing and chunking
│   ├── vector_store.py         # ChromaDB and hybrid search
│   ├── qa_chain.py             # Question-answering with LLM
│   └── conversation_manager.py # Conversation memory management
├── data/
│   ├── uploads/                # Uploaded documents
│   └── chroma_db/              # Vector database storage
├── config.py                   # Configuration settings
└── requirements.txt            # Python dependencies
```

## Key Features

### Hybrid Search
Combines semantic similarity search with keyword matching using weighted scoring algorithms for improved retrieval accuracy.

### Conversation Memory
Maintains context across multiple questions, enabling natural follow-up conversations and reference to previous exchanges.

### Adaptive Chunking
Automatically detects document types (resume, research paper, documentation) and applies optimized chunking strategies for better context preservation.

### Privacy-Focused
All processing happens locally with no data sent to external APIs, ensuring complete privacy of your documents.

## Requirements

- Python 3.10+
- 8GB+ RAM (for local LLM)
- 10GB+ free storage space
- Ollama installed and running

## Configuration

Key settings in `config.py`:
- `CHUNK_SIZE`: Document chunk size (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `LLM_MODEL`: Ollama model to use (default: "llama3.2:1b")
- `MAX_RETRIEVED_CHUNKS`: Number of chunks to retrieve (default: 5)

## Contributing

This is a personal project built for learning and portfolio purposes. Feel free to fork and adapt for your own use.

## Author

**Fahd Ahmad**  
Software Engineering Graduate | AI/ML Enthusiast  

---

Built with ❤️ for personal knowledge management and RAG system learning.