"""
Simple RAG Service Test - No Dependencies
Bu dosya, RAG Service'in temel yapısını test eder
"""

import os
import sys
import json
import tempfile

def test_rag_config():
    """RAGConfig import ve instance test"""
    try:
        # Add src to path
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        
        print("📋 Testing RAGConfig...")
        
        # Basic import test
        from rag_service import RAGConfig
        
        # Create basic config
        config = RAGConfig()
        
        print(f"✅ RAGConfig created successfully")
        print(f"   Base model: {config.base_model_name}")
        print(f"   Vector store type: {config.vector_store_type}")
        print(f"   Device: {config.device}")
        print(f"   Top-k retrieval: {config.top_k_retrieval}")
        
        # Test custom config
        custom_config = RAGConfig(
            top_k_retrieval=10,
            temperature=0.8,
            vector_store_type="pinecone"
        )
        
        assert custom_config.top_k_retrieval == 10
        assert custom_config.temperature == 0.8
        assert custom_config.vector_store_type == "pinecone"
        
        print("✅ Custom RAGConfig test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ RAGConfig test failed: {e}")
        return False


def test_rag_config_examples():
    """RAG config examples test"""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        
        print("\n📋 Testing RAG Config Examples...")
        
        from rag_config_examples import get_rag_config
        
        # Test all config types
        config_names = ["basic", "production", "memory_optimized", "fast_inference", "debug"]
        
        for name in config_names:
            try:
                config = get_rag_config(name)
                print(f"✅ {name}: {config.base_model_name}")
            except Exception as e:
                print(f"❌ {name}: {e}")
                return False
        
        # Test invalid config
        try:
            get_rag_config("invalid")
            print("❌ Should have raised ValueError")
            return False
        except ValueError:
            print("✅ Invalid config correctly rejected")
        
        return True
        
    except Exception as e:
        print(f"❌ RAG Config Examples test failed: {e}")
        return False


def test_rag_service_structure():
    """RAG Service class structure test"""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        
        print("\n📋 Testing RAG Service Structure...")
        
        from rag_service import RAGService, RAGConfig
        
        # Check if class has required methods
        required_methods = [
            'generate_response',
            '_encode_query',
            '_search_context',
            '_format_context_for_prompt',
            '_construct_prompt',
            'get_service_stats'
        ]
        
        for method in required_methods:
            if hasattr(RAGService, method):
                print(f"✅ Method {method} exists")
            else:
                print(f"❌ Method {method} missing")
                return False
        
        # Check utility functions
        from rag_service import create_rag_service, test_rag_service
        print("✅ Utility functions imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ RAG Service Structure test failed: {e}")
        return False


def test_file_creation():
    """Test edilecek dosyaların varlığını kontrol eder"""
    try:
        print("\n📁 Testing File Creation...")
        
        base_path = os.path.dirname(__file__)
        required_files = [
            'src/rag_service.py',
            'src/rag_config_examples.py',
            'test_rag_service.py'
        ]
        
        for file_path in required_files:
            full_path = os.path.join(base_path, file_path)
            if os.path.exists(full_path):
                size = os.path.getsize(full_path)
                print(f"✅ {file_path} ({size} bytes)")
            else:
                print(f"❌ {file_path} not found")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ File Creation test failed: {e}")
        return False


def main():
    """Run all simple tests"""
    print("🧪 RAG Service Simple Test Suite")
    print("=" * 50)
    
    tests = [
        ("File Creation", test_file_creation),
        ("RAG Config", test_rag_config),
        ("RAG Config Examples", test_rag_config_examples),
        ("RAG Service Structure", test_rag_service_structure)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name} Test...")
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("💥 Some tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
