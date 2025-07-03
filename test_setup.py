#!/usr/bin/env python3
"""
Test script to verify the RAG system setup
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import streamlit
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import chromadb
        print("✅ ChromaDB imported successfully")
    except ImportError as e:
        print(f"❌ ChromaDB import failed: {e}")
        return False
    
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ SentenceTransformers imported successfully")
    except ImportError as e:
        print(f"❌ SentenceTransformers import failed: {e}")
        return False
    
    try:
        import ollama
        print("✅ Ollama imported successfully")
    except ImportError as e:
        print(f"❌ Ollama import failed: {e}")
        return False
    
    try:
        import PyPDF2
        print("✅ PyPDF2 imported successfully")
    except ImportError as e:
        print(f"❌ PyPDF2 import failed: {e}")
        return False
    
    return True

def test_directories():
    """Test if required directories exist"""
    print("\nTesting directory structure...")
    
    from config import DATA_DIR, UPLOADS_DIR, CHROMA_DB_DIR
    
    directories = [
        ("Data directory", DATA_DIR),
        ("Uploads directory", UPLOADS_DIR),
        ("ChromaDB directory", CHROMA_DB_DIR)
    ]
    
    for name, path in directories:
        if path.exists():
            print(f"✅ {name}: {path}")
        else:
            print(f"❌ {name} missing: {path}")
            return False
    
    return True

def test_embedding_model():
    """Test if embedding model can be loaded"""
    print("\nTesting embedding model...")
    
    try:
        from sentence_transformers import SentenceTransformer
        from config import EMBEDDING_MODEL
        
        model = SentenceTransformer(EMBEDDING_MODEL)
        
        # Test encoding
        test_text = "This is a test sentence."
        embedding = model.encode([test_text])
        
        print(f"✅ Embedding model loaded: {EMBEDDING_MODEL}")
        print(f"✅ Test embedding shape: {embedding.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Embedding model test failed: {e}")
        return False

def test_ollama_connection():
    """Test Ollama connection"""
    print("\nTesting Ollama connection...")
    
    try:
        import ollama
        from config import LLM_MODEL, OLLAMA_BASE_URL
        
        client = ollama.Client(host=OLLAMA_BASE_URL)
        
        # Test connection
        response = client.chat(
            model=LLM_MODEL,
            messages=[
                {
                    'role': 'user',
                    'content': 'Hello, this is a test. Please respond with "Test successful".'
                }
            ]
        )
        
        print(f"✅ Ollama connected successfully")
        print(f"✅ Model response: {response['message']['content'][:50]}...")
        return True
        
    except Exception as e:
        print(f"❌ Ollama test failed: {e}")
        print("Make sure Ollama is running: ollama serve")
        print(f"Make sure model is downloaded: ollama pull {LLM_MODEL}")
        return False

def main():
    """Run all tests"""
    print("🧪 RAG System Setup Test\n")
    
    tests = [
        test_imports,
        test_directories,
        test_embedding_model,
        test_ollama_connection
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()  # Add spacing between tests
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your RAG system is ready to go.")
        print("Run: streamlit run app.py")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        return 1