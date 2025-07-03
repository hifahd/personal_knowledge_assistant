#!/usr/bin/env python3
"""
Quick setup checker for advanced RAG features
"""

def check_imports():
    """Check if all required packages are available"""
    print("🔍 Checking imports...")
    
    try:
        import streamlit
        print("✅ Streamlit")
    except ImportError:
        print("❌ Streamlit - run: pip install streamlit")
        return False
    
    try:
        import chromadb
        print("✅ ChromaDB")
    except ImportError:
        print("❌ ChromaDB - run: pip install chromadb")
        return False
    
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ SentenceTransformers")
    except ImportError:
        print("❌ SentenceTransformers - run: pip install sentence-transformers")
        return False
    
    try:
        import sklearn
        print("✅ Scikit-learn")
    except ImportError:
        print("❌ Scikit-learn - run: pip install scikit-learn")
        return False
    
    try:
        import ollama
        print("✅ Ollama")
    except ImportError:
        print("❌ Ollama - run: pip install ollama")
        return False
    
    return True

def check_files():
    """Check if all required files exist"""
    print("\n📁 Checking file structure...")
    
    required_files = [
        "app.py",
        "config.py",
        "src/document_processor.py",
        "src/vector_store.py", 
        "src/qa_chain.py",
        "src/conversation_manager.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        try:
            with open(file_path, 'r') as f:
                print(f"✅ {file_path}")
        except FileNotFoundError:
            print(f"❌ {file_path}")
            missing_files.append(file_path)
    
    return len(missing_files) == 0

def check_ollama():
    """Check if Ollama is working"""
    print("\n🤖 Checking Ollama...")
    
    try:
        import subprocess
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Ollama is installed")
            if 'llama3.2:1b' in result.stdout:
                print("✅ llama3.2:1b model found")
                return True
            else:
                print("❌ llama3.2:1b model not found - run: ollama pull llama3.2:1b")
                return False
        else:
            print("❌ Ollama not working properly")
            return False
    except Exception as e:
        print(f"❌ Ollama check failed: {e}")
        return False

def main():
    print("🧪 Advanced RAG System Setup Check\n")
    
    imports_ok = check_imports()
    files_ok = check_files()
    ollama_ok = check_ollama()
    
    print(f"\n📊 Results:")
    print(f"Imports: {'✅' if imports_ok else '❌'}")
    print(f"Files: {'✅' if files_ok else '❌'}")
    print(f"Ollama: {'✅' if ollama_ok else '❌'}")
    
    if imports_ok and files_ok and ollama_ok:
        print("\n🎉 All systems ready! Run: streamlit run app.py")
    else:
        print("\n🔧 Please fix the issues above before running the app")

if __name__ == "__main__":
    main()