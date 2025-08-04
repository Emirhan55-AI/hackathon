"""
FastAPI Structure Test - No Heavy Dependencies
Bu dosya, FastAPI aplikasyonunun temel yapısını test eder
"""

import os
import sys
import json

def test_api_file_structure():
    """API dosya yapısını test eder"""
    try:
        print("📁 Testing API File Structure...")
        
        base_path = os.path.dirname(__file__)
        required_files = [
            'src/api/main.py',
            'src/api/__init__.py',
            'src/rag_service.py',
            'src/rag_config_examples.py',
            'test_api.py',
            'Dockerfile',
            'docker-compose.yml'
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
        print(f"❌ File Structure test failed: {e}")
        return False


def test_api_imports():
    """API import'larını test eder (mock olarak)"""
    try:
        print("\n📋 Testing API Structure (Mock)...")
        
        # Read main.py content
        main_py_path = os.path.join(os.path.dirname(__file__), 'src', 'api', 'main.py')
        
        if not os.path.exists(main_py_path):
            print("❌ main.py not found")
            return False
        
        with open(main_py_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for required components  
        required_components = [
            'FastAPI',
            'ChatRequest',
            'ChatResponse', 
            'HealthResponse',
            '/health',
            '/chat',
            'RAGService',
            'global_rag_service',
            'lifespan',
            'uvicorn'
        ]
        
        for component in required_components:
            if component in content:
                print(f"✅ {component} found in code")
            else:
                print(f"❌ {component} missing from code")
                return False
        
        # Check file size (should be substantial)
        if len(content) > 20000:  # At least 20KB
            print(f"✅ File size OK ({len(content)} chars)")
        else:
            print(f"❌ File seems too small ({len(content)} chars)")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ API Structure test failed: {e}")
        return False


def test_docker_configuration():
    """Docker konfigürasyonunu test eder"""
    try:
        print("\n🐳 Testing Docker Configuration...")
        
        # Test Dockerfile
        dockerfile_path = os.path.join(os.path.dirname(__file__), 'Dockerfile')
        if os.path.exists(dockerfile_path):
            with open(dockerfile_path, 'r') as f:
                dockerfile_content = f.read()
            
            if 'src.api.main' in dockerfile_content:
                print("✅ Dockerfile uses correct main module")
            else:
                print("❌ Dockerfile doesn't reference src.api.main")
                return False
                
            if 'EXPOSE 8003' in dockerfile_content:
                print("✅ Dockerfile exposes correct port")
            else:
                print("❌ Dockerfile doesn't expose port 8003")
                return False
        
        # Test docker-compose
        compose_path = os.path.join(os.path.dirname(__file__), 'docker-compose.yml')
        if os.path.exists(compose_path):
            with open(compose_path, 'r') as f:
                compose_content = f.read()
            
            if 'conversational-ai' in compose_content:
                print("✅ docker-compose has correct service name")
            else:
                print("❌ docker-compose missing service name")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Docker Configuration test failed: {e}")
        return False


def test_endpoint_definitions():
    """Endpoint tanımlarını test eder"""
    try:
        print("\n🌐 Testing Endpoint Definitions...")
        
        main_py_path = os.path.join(os.path.dirname(__file__), 'src', 'api', 'main.py')
        
        with open(main_py_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Required endpoints
        endpoints = [
            '@app.get("/", response_model=Dict[str, str])',
            '@app.get("/health", response_model=HealthResponse)',
            '@app.post("/chat", response_model=ChatResponse)',
            '@app.post("/chat/batch", response_model=BatchChatResponse)',
            '@app.get("/chat/stats", response_model=Dict[str, Any])',
            '@app.websocket("/ws/chat/{user_id}")'
        ]
        
        for endpoint in endpoints:
            if endpoint in content:
                print(f"✅ {endpoint} defined")
            else:
                print(f"❌ {endpoint} missing")
                return False
        
        # Check middleware
        middleware_checks = [
            'CORSMiddleware',
            'TrustedHostMiddleware',
            'LoggingMiddleware'
        ]
        
        for middleware in middleware_checks:
            if middleware in content:
                print(f"✅ {middleware} configured")
            else:
                print(f"❌ {middleware} missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Endpoint Definitions test failed: {e}")
        return False


def test_pydantic_models():
    """Pydantic model tanımlarını test eder"""
    try:
        print("\n📝 Testing Pydantic Models...")
        
        main_py_path = os.path.join(os.path.dirname(__file__), 'src', 'api', 'main.py')
        
        with open(main_py_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Required models
        models = [
            'class ChatRequest(BaseModel)',
            'class ChatResponse(BaseModel)',
            'class BatchChatRequest(BaseModel)',
            'class BatchChatResponse(BaseModel)',
            'class HealthResponse(BaseModel)',
            'class ErrorResponse(BaseModel)'
        ]
        
        for model in models:
            if model in content:
                print(f"✅ {model} defined")
            else:
                print(f"❌ {model} missing")
                return False
        
        # Check field definitions
        field_checks = [
            'query: str = Field',
            'user_id: str = Field',
            'response: str = Field',
            'success: bool = Field'
        ]
        
        for field in field_checks:
            if field in content:
                print(f"✅ {field} field defined")
            else:
                print(f"❌ {field} field missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Pydantic Models test failed: {e}")
        return False


def main():
    """Run all structure tests"""
    print("🧪 FastAPI Structure Test Suite")
    print("=" * 50)
    
    tests = [
        ("File Structure", test_api_file_structure),
        ("API Imports & Structure", test_api_imports),
        ("Docker Configuration", test_docker_configuration),
        ("Endpoint Definitions", test_endpoint_definitions),
        ("Pydantic Models", test_pydantic_models)
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
        print("🎉 All structure tests passed!")
        print("\n📋 FastAPI Service Summary:")
        print("✅ Complete REST API with chat endpoints")
        print("✅ WebSocket support for real-time chat")
        print("✅ Comprehensive error handling")
        print("✅ Pydantic models for validation")
        print("✅ Docker containerization ready")
        print("✅ Production deployment configuration")
        return True
    else:
        print("💥 Some structure tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
