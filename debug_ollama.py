#!/usr/bin/env python3
"""
Debug script to test Ollama connection
"""

import subprocess
import ollama

def test_subprocess():
    """Test Ollama via subprocess"""
    print("Testing Ollama via subprocess...")
    try:
        result = subprocess.run(
            ['ollama', 'run', 'llama3.2:1b', 'Hello'],
            capture_output=True,
            text=True,
            timeout=30
        )
        print(f"Return code: {result.returncode}")
        print(f"Output: {result.stdout}")
        if result.stderr:
            print(f"Error: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"Subprocess test failed: {e}")
        return False

def test_client():
    """Test Ollama via client"""
    print("\nTesting Ollama via client...")
    try:
        client = ollama.Client()
        response = client.chat(
            model='llama3.2:1b',
            messages=[
                {
                    'role': 'user',
                    'content': 'Hello'
                }
            ]
        )
        print(f"Client response: {response['message']['content']}")
        return True
    except Exception as e:
        print(f"Client test failed: {e}")
        return False

def test_client_with_host():
    """Test Ollama via client with specific host"""
    print("\nTesting Ollama via client with host...")
    try:
        client = ollama.Client(host='http://localhost:11434')
        response = client.chat(
            model='llama3.2:1b',
            messages=[
                {
                    'role': 'user',
                    'content': 'Hello'
                }
            ]
        )
        print(f"Client with host response: {response['message']['content']}")
        return True
    except Exception as e:
        print(f"Client with host test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Ollama Connection Debug Test\n")
    
    subprocess_works = test_subprocess()
    client_works = test_client()
    client_host_works = test_client_with_host()
    
    print(f"\n📊 Results:")
    print(f"Subprocess: {'✅' if subprocess_works else '❌'}")
    print(f"Client: {'✅' if client_works else '❌'}")
    print(f"Client with host: {'✅' if client_host_works else '❌'}")
    
    if subprocess_works:
        print("\n✅ Subprocess method works - we can use this!")
    elif client_works or client_host_works:
        print("\n✅ Client method works - we can use this!")
    else:
        print("\n❌ No method works - need to debug further")