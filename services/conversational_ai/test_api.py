"""
Test Suite for Conversational AI FastAPI Service
Bu dosya, FastAPI endpoints ve RAG Service entegrasyonunu test eder.
"""

import os
import sys
import json
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import FastAPI

# Test environment setup
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Test imports
try:
    from src.api.main import app, ChatRequest, ChatResponse, HealthResponse
    from src.rag_service import RAGService, RAGConfig
    API_IMPORTS_AVAILABLE = True
except ImportError as e:
    API_IMPORTS_AVAILABLE = False
    print(f"Warning: API imports failed: {e}")


class TestFastAPIEndpoints:
    """FastAPI endpoints'leri test eder"""
    
    @pytest.fixture
    def client(self):
        """Test client oluştur"""
        if not API_IMPORTS_AVAILABLE:
            pytest.skip("API imports not available")
        
        # Mock RAG service to avoid loading heavy models
        with patch('src.api.main.global_rag_service') as mock_rag:
            mock_rag.generate_response.return_value = {
                "success": True,
                "response": "Test fashion advice response",
                "context_used": [
                    {"content": "Blue jacket", "similarity": 0.9}
                ],
                "metadata": {
                    "processing_time": 0.5,
                    "user_id": "test_user"
                }
            }
            mock_rag.get_service_stats.return_value = {
                "service_status": "active",
                "models_loaded": {"llm_model": True, "embedding_model": True}
            }
            
            with TestClient(app) as test_client:
                yield test_client
    
    def test_root_endpoint(self, client):
        """Root endpoint testi"""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "Conversational AI" in data["message"]
    
    def test_health_endpoint(self, client):
        """Health endpoint testi"""
        with patch('src.api.main.global_rag_service') as mock_rag:
            mock_rag.get_service_stats.return_value = {
                "service_status": "active",
                "models_loaded": True
            }
            
            response = client.get("/health")
            assert response.status_code == 200
            
            data = response.json()
            assert data["service"] == "Conversational AI"
            assert "status" in data
            assert "timestamp" in data
    
    def test_chat_endpoint_success(self, client):
        """Başarılı chat endpoint testi"""
        chat_request = {
            "query": "What should I wear today?",
            "user_id": "test_user_123",
            "session_id": "session_abc",
            "context": {"season": "winter"}
        }
        
        with patch('src.api.main.global_rag_service') as mock_rag:
            mock_rag.generate_response.return_value = {
                "success": True,
                "response": "I recommend wearing a warm winter coat with boots.",
                "context_used": [
                    {"content": "Winter coat", "similarity": 0.95}
                ],
                "metadata": {"processing_time": 0.3}
            }
            
            response = client.post("/chat", json=chat_request)
            assert response.status_code == 200
            
            data = response.json()
            assert data["success"] == True
            assert "response" in data
            assert data["user_id"] == "test_user_123"
            assert "confidence" in data
            assert "suggestions" in data
    
    def test_chat_endpoint_validation_error(self, client):
        """Chat endpoint validation error testi"""
        # Empty query
        invalid_request = {
            "query": "",
            "user_id": "test_user"
        }
        
        response = client.post("/chat", json=invalid_request)
        assert response.status_code == 422
    
    def test_chat_endpoint_missing_fields(self, client):
        """Chat endpoint eksik field testi"""
        incomplete_request = {
            "query": "Test query"
            # user_id missing
        }
        
        response = client.post("/chat", json=incomplete_request)
        assert response.status_code == 422
    
    def test_batch_chat_endpoint(self, client):
        """Batch chat endpoint testi"""
        batch_request = {
            "messages": [
                {
                    "query": "What to wear for work?",
                    "user_id": "user1"
                },
                {
                    "query": "Casual outfit suggestion?", 
                    "user_id": "user2"
                }
            ]
        }
        
        with patch('src.api.main.global_rag_service') as mock_rag:
            mock_rag.generate_response.return_value = {
                "success": True,
                "response": "Fashion advice response",
                "context_used": [],
                "metadata": {"processing_time": 0.2}
            }
            
            response = client.post("/chat/batch", json=batch_request)
            assert response.status_code == 200
            
            data = response.json()
            assert "responses" in data
            assert data["processed_count"] == 2
            assert len(data["responses"]) == 2
    
    def test_chat_stats_endpoint(self, client):
        """Chat stats endpoint testi"""
        with patch('src.api.main.global_rag_service') as mock_rag:
            mock_rag.get_service_stats.return_value = {
                "service_status": "active",
                "models_loaded": {"llm_model": True}
            }
            
            response = client.get("/chat/stats")
            assert response.status_code == 200
            
            data = response.json()
            assert "rag_service" in data
            assert "api_service" in data
    
    def test_rag_service_not_loaded(self, client):
        """RAG service yüklenmemiş durumu testi"""
        with patch('src.api.main.global_rag_service', None):
            response = client.post("/chat", json={
                "query": "Test query",
                "user_id": "test_user"
            })
            assert response.status_code == 503


class TestPydanticModels:
    """Pydantic model'ları test eder"""
    
    def test_chat_request_valid(self):
        """Geçerli ChatRequest testi"""
        if not API_IMPORTS_AVAILABLE:
            pytest.skip("API imports not available")
            
        request = ChatRequest(
            query="What should I wear?",
            user_id="user123",
            session_id="session456",
            context={"season": "summer"}
        )
        
        assert request.query == "What should I wear?"
        assert request.user_id == "user123"
        assert request.session_id == "session456"
        assert request.context["season"] == "summer"
    
    def test_chat_request_validation(self):
        """ChatRequest validation testi"""
        if not API_IMPORTS_AVAILABLE:
            pytest.skip("API imports not available")
            
        # Empty query should be rejected
        with pytest.raises(ValueError):
            ChatRequest(
                query="",
                user_id="user123"
            )
        
        # Whitespace only query should be rejected
        with pytest.raises(ValueError):
            ChatRequest(
                query="   ",
                user_id="user123"
            )
    
    def test_chat_response_creation(self):
        """ChatResponse oluşturma testi"""
        if not API_IMPORTS_AVAILABLE:
            pytest.skip("API imports not available")
            
        response = ChatResponse(
            success=True,
            response="Fashion advice here",
            user_id="user123",
            session_id="session456",
            context_used=[{"content": "Blue shirt"}],
            confidence=0.95,
            suggestions=["Try a blazer", "Add accessories"],
            metadata={"processing_time": 0.5}
        )
        
        assert response.success == True
        assert response.confidence == 0.95
        assert len(response.suggestions) == 2


class TestUtilityFunctions:
    """Utility fonksiyonlarını test eder"""
    
    def test_generate_suggestions(self):
        """Suggestion generation testi"""
        if not API_IMPORTS_AVAILABLE:
            pytest.skip("API imports not available")
            
        from src.api.main import generate_suggestions
        
        # Test color-related suggestions
        suggestions = generate_suggestions("hangi renk yakışır", "mavi ceket öneriyorum")
        assert len(suggestions) > 0
        assert any("renk" in s.lower() for s in suggestions)
        
        # Test work-related suggestions
        suggestions = generate_suggestions("iş toplantısı için ne giysem", "formal ceket")
        assert len(suggestions) > 0
        assert any("iş" in s.lower() or "formal" in s.lower() for s in suggestions)
    
    def test_check_rag_service_availability(self):
        """RAG service availability check testi"""
        if not API_IMPORTS_AVAILABLE:
            pytest.skip("API imports not available")
            
        from src.api.main import check_rag_service_availability
        from fastapi import HTTPException
        
        # Mock global_rag_service as None
        with patch('src.api.main.global_rag_service', None):
            with pytest.raises(HTTPException) as exc_info:
                check_rag_service_availability()
            assert exc_info.value.status_code == 503


class TestWebSocketEndpoint:
    """WebSocket endpoint'i test eder"""
    
    @pytest.mark.asyncio
    async def test_websocket_connection(self):
        """WebSocket connection testi"""
        if not API_IMPORTS_AVAILABLE:
            pytest.skip("API imports not available")
            
        with patch('src.api.main.global_rag_service') as mock_rag:
            mock_rag.generate_response.return_value = {
                "success": True,
                "response": "WebSocket test response",
                "context_used": [],
                "metadata": {"processing_time": 0.1}
            }
            
            with TestClient(app) as client:
                with client.websocket_connect("/ws/chat/test_user") as websocket:
                    # Send test message
                    websocket.send_json({
                        "message": "Hello WebSocket",
                        "session_id": "ws_session"
                    })
                    
                    # Receive response
                    data = websocket.receive_json()
                    assert data["success"] == True
                    assert "response" in data
                    assert "timestamp" in data


def run_api_tests():
    """API testlerini çalıştır"""
    print("🧪 Conversational AI API Test Suite başlatılıyor...\n")
    
    if not API_IMPORTS_AVAILABLE:
        print("❌ API imports not available. Skipping tests.")
        return False
    
    # Run tests with pytest
    import subprocess
    result = subprocess.run([
        sys.executable, "-m", "pytest", 
        __file__, 
        "-v",
        "--tb=short"
    ], capture_output=True, text=True)
    
    print("📊 Test Results:")
    print(result.stdout)
    
    if result.stderr:
        print("❌ Errors:")
        print(result.stderr)
    
    return result.returncode == 0


def simple_api_test():
    """Basit API testi - pytest olmadan"""
    print("🧪 Simple API Test başlatılıyor...\n")
    
    if not API_IMPORTS_AVAILABLE:
        print("❌ API imports not available")
        return False
    
    try:
        # Test client oluştur
        client = TestClient(app)
        
        # Test 1: Root endpoint
        print("🔍 Testing root endpoint...")
        response = client.get("/")
        assert response.status_code == 200
        print("✅ Root endpoint OK")
        
        # Test 2: Health endpoint (RAG service olmadan)
        print("🔍 Testing health endpoint...")
        with patch('src.api.main.global_rag_service', None):
            response = client.get("/health")
            assert response.status_code == 200
            print("✅ Health endpoint OK")
        
        # Test 3: Chat endpoint validation
        print("🔍 Testing chat validation...")
        response = client.post("/chat", json={"query": "", "user_id": "test"})
        assert response.status_code == 422
        print("✅ Chat validation OK")
        
        # Test 4: Pydantic models
        print("🔍 Testing Pydantic models...")
        request = ChatRequest(query="Test query", user_id="test_user")
        assert request.query == "Test query"
        print("✅ Pydantic models OK")
        
        print("\n🎉 All simple tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    # Run simple test first
    simple_success = simple_api_test()
    
    if simple_success:
        print("\n" + "="*50)
        print("Running full test suite...")
        full_success = run_api_tests()
        
        if full_success:
            print("🎉 All tests passed!")
            sys.exit(0)
        else:
            print("💥 Some tests failed!")
            sys.exit(1)
    else:
        print("💥 Simple tests failed!")
        sys.exit(1)
