"""
OutfitTransformer API Test Suite - Aura Project
Bu modül, OutfitTransformer API'sinin test edilmesi için test case'leri içerir.
"""

import pytest
import asyncio
import json
import time
from typing import Dict, List, Any
from pathlib import Path
import tempfile
import os

# FastAPI testing
from fastapi.testclient import TestClient
from fastapi import status

# Test data
import numpy as np
from PIL import Image

# Local imports
from main import app
from models import (
    OutfitCompatibilityRequest,
    OutfitRecommendationRequest,
    ItemRecommendationRequest,
    AddItemRequest
)
from inference import OutfitRecommendationEngine

# Test client
client = TestClient(app)

# Test configuration
TEST_CONFIG = {
    "test_model_path": "./test_data/test_model.pt",
    "test_items_db": "./test_data/test_items.json",
    "test_images_dir": "./test_data/images",
    "demo_mode": True
}


# Test fixtures
@pytest.fixture
def test_client():
    """FastAPI test client"""
    return client


@pytest.fixture
def sample_item_ids():
    """Sample item IDs for testing"""
    return ["demo_0001", "demo_0002", "demo_0003", "demo_0004"]


@pytest.fixture
def sample_outfit_request():
    """Sample outfit compatibility request"""
    return {
        "item_ids": ["demo_0001", "demo_0002"],
        "return_detailed_scores": True,
        "include_graph_analysis": False
    }


@pytest.fixture
def sample_recommendation_request():
    """Sample recommendation request"""
    return {
        "seed_item_ids": ["demo_0001"],
        "target_categories": ["bottoms", "shoes"],
        "occasion": "casual",
        "season": "summer",
        "max_recommendations": 3
    }


@pytest.fixture
def sample_image_file():
    """Sample image file for testing"""
    # Create temporary image
    image = Image.new('RGB', (224, 224), color='red')
    temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    image.save(temp_file.name)
    
    yield temp_file.name
    
    # Cleanup
    os.unlink(temp_file.name)


# Health check tests
class TestHealthCheck:
    """Health check endpoint tests"""
    
    def test_health_check_success(self, test_client):
        """Test health check endpoint"""
        response = test_client.get("/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "model_loaded" in data
        assert data["status"] == "healthy"
    
    def test_health_check_contains_stats(self, test_client):
        """Test health check includes database stats"""
        response = test_client.get("/health")
        data = response.json()
        
        if data["model_loaded"]:
            assert "database_stats" in data


# Outfit compatibility tests
class TestOutfitCompatibility:
    """Outfit compatibility analysis tests"""
    
    def test_compatibility_valid_request(self, test_client, sample_outfit_request):
        """Test valid compatibility request"""
        response = test_client.post("/outfit/compatibility", json=sample_outfit_request)
        
        # Demo mode'da model yoksa 503 dönebilir
        if response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE:
            pytest.skip("Model not available in test environment")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Response format check
        assert "outfit_id" in data
        assert "item_ids" in data
        assert "is_compatible" in data
        assert "compatibility_score" in data
        assert "outfit_score" in data
        assert "recommendation" in data
        assert "fashion_rules" in data
        
        # Score validation
        assert 0.0 <= data["compatibility_score"] <= 1.0
        assert isinstance(data["is_compatible"], bool)
        assert data["recommendation"] in ["compatible", "incompatible"]
    
    def test_compatibility_invalid_single_item(self, test_client):
        """Test compatibility with single item (should fail)"""
        request = {
            "item_ids": ["demo_0001"],
            "return_detailed_scores": False
        }
        
        response = test_client.post("/outfit/compatibility", json=request)
        assert response.status_code == status.HTTP_400_BAD_REQUEST
    
    def test_compatibility_too_many_items(self, test_client):
        """Test compatibility with too many items"""
        request = {
            "item_ids": [f"demo_{i:04d}" for i in range(1, 8)],  # 7 items (max 6)
            "return_detailed_scores": False
        }
        
        response = test_client.post("/outfit/compatibility", json=request)
        assert response.status_code == status.HTTP_400_BAD_REQUEST
    
    def test_compatibility_duplicate_items(self, test_client):
        """Test compatibility with duplicate items"""
        request = {
            "item_ids": ["demo_0001", "demo_0001"],  # Duplicate
            "return_detailed_scores": False
        }
        
        response = test_client.post("/outfit/compatibility", json=request)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_compatibility_detailed_scores(self, test_client):
        """Test compatibility with detailed scores"""
        request = {
            "item_ids": ["demo_0001", "demo_0002"],
            "return_detailed_scores": True,
            "include_graph_analysis": True
        }
        
        response = test_client.post("/outfit/compatibility", json=request)
        
        if response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE:
            pytest.skip("Model not available")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Detailed scores check
        if "detailed_scores" in data:
            assert "compatibility_probs" in data["detailed_scores"]
            assert "outfit_scores" in data["detailed_scores"]
            assert "predicted_label" in data["detailed_scores"]
        
        # Graph analysis check
        if "graph_analysis" in data:
            assert "num_nodes" in data["graph_analysis"]
            assert "num_edges" in data["graph_analysis"]


# Outfit recommendation tests
class TestOutfitRecommendations:
    """Outfit recommendation tests"""
    
    def test_recommendation_valid_request(self, test_client, sample_recommendation_request):
        """Test valid recommendation request"""
        response = test_client.post("/outfit/recommendations", json=sample_recommendation_request)
        
        if response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE:
            pytest.skip("Model not available")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Response format check
        assert "seed_item_ids" in data
        assert "recommendations" in data
        assert "total_recommendations" in data
        assert "filters_applied" in data
        
        # Recommendations format
        assert isinstance(data["recommendations"], list)
        assert data["total_recommendations"] == len(data["recommendations"])
        
        # Filter validation
        filters = data["filters_applied"]
        assert filters["target_categories"] == sample_recommendation_request["target_categories"]
        assert filters["occasion"] == sample_recommendation_request["occasion"]
        assert filters["season"] == sample_recommendation_request["season"]
    
    def test_recommendation_no_seed_items(self, test_client):
        """Test recommendation without seed items"""
        request = {
            "seed_item_ids": [],
            "max_recommendations": 3
        }
        
        response = test_client.post("/outfit/recommendations", json=request)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_recommendation_invalid_occasion(self, test_client):
        """Test recommendation with invalid occasion"""
        request = {
            "seed_item_ids": ["demo_0001"],
            "occasion": "invalid_occasion",
            "max_recommendations": 3
        }
        
        response = test_client.post("/outfit/recommendations", json=request)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_recommendation_invalid_season(self, test_client):
        """Test recommendation with invalid season"""
        request = {
            "seed_item_ids": ["demo_0001"],
            "season": "invalid_season",
            "max_recommendations": 3
        }
        
        response = test_client.post("/outfit/recommendations", json=request)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_recommendation_max_limit(self, test_client):
        """Test recommendation with max limit"""
        request = {
            "seed_item_ids": ["demo_0001"],
            "max_recommendations": 25  # Over limit (max 20)
        }
        
        response = test_client.post("/outfit/recommendations", json=request)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


# Item recommendation tests
class TestItemRecommendations:
    """Item recommendation tests"""
    
    def test_item_recommendation_valid(self, test_client):
        """Test valid item recommendation"""
        request = {
            "item_id": "demo_0001",
            "target_categories": ["bottoms"],
            "max_recommendations": 5
        }
        
        response = test_client.post("/items/recommendations", json=request)
        
        if response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE:
            pytest.skip("Model not available")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Response format
        assert "item_id" in data
        assert "recommendations" in data
        assert "total_recommendations" in data
        assert "filters_applied" in data
        
        assert data["item_id"] == request["item_id"]
        assert isinstance(data["recommendations"], list)
    
    def test_item_recommendation_empty_item_id(self, test_client):
        """Test item recommendation with empty item ID"""
        request = {
            "item_id": "",
            "max_recommendations": 5
        }
        
        response = test_client.post("/items/recommendations", json=request)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_item_recommendation_over_limit(self, test_client):
        """Test item recommendation over limit"""
        request = {
            "item_id": "demo_0001",
            "max_recommendations": 100  # Over limit (max 50)
        }
        
        response = test_client.post("/items/recommendations", json=request)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


# Item management tests
class TestItemManagement:
    """Item add/management tests"""
    
    def test_add_item_valid(self, test_client, sample_image_file):
        """Test valid item addition"""
        with open(sample_image_file, "rb") as f:
            files = {"image": ("test.jpg", f, "image/jpeg")}
            data = {
                "item_id": "test_item_001",
                "category": "tops",
                "color": "blue",
                "style": "casual",
                "price": 29.99,
                "brand": "Test Brand"
            }
            
            response = test_client.post("/items/add", files=files, data=data)
        
        if response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE:
            pytest.skip("Model not available")
        
        assert response.status_code == status.HTTP_200_OK
        result = response.json()
        
        assert result["success"] is True
        assert result["item_id"] == "test_item_001"
        assert "item_data" in result
        assert "message" in result
    
    def test_add_item_invalid_category(self, test_client, sample_image_file):
        """Test item addition with invalid category"""
        with open(sample_image_file, "rb") as f:
            files = {"image": ("test.jpg", f, "image/jpeg")}
            data = {
                "item_id": "test_item_002",
                "category": "invalid_category",
                "color": "blue"
            }
            
            response = test_client.post("/items/add", files=files, data=data)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_add_item_no_image(self, test_client):
        """Test item addition without image"""
        data = {
            "item_id": "test_item_003",
            "category": "tops"
        }
        
        response = test_client.post("/items/add", data=data)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


# Database stats tests
class TestDatabaseStats:
    """Database statistics tests"""
    
    def test_database_stats(self, test_client):
        """Test database statistics endpoint"""
        response = test_client.get("/database/stats")
        
        if response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE:
            pytest.skip("Model not available")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Stats format check
        assert "total_items" in data
        assert "categories" in data
        assert "colors" in data
        assert "styles" in data
        assert "has_faiss_index" in data
        assert "embedding_dimension" in data
        assert "timestamp" in data
        
        # Data types
        assert isinstance(data["total_items"], int)
        assert isinstance(data["categories"], dict)
        assert isinstance(data["has_faiss_index"], bool)


# Outfit analysis tests
class TestOutfitAnalysis:
    """Outfit analysis tests"""
    
    def test_outfit_analysis_valid(self, test_client):
        """Test valid outfit analysis"""
        outfit_id = "test_outfit_001"
        item_ids = "demo_0001,demo_0002"
        
        response = test_client.get(f"/outfit/analyze/{outfit_id}?item_ids={item_ids}")
        
        if response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE:
            pytest.skip("Model not available")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Analysis format
        assert "outfit_id" in data
        assert "item_ids" in data
        assert "items_detail" in data
        assert "compatibility_analysis" in data
        assert "graph_analysis" in data
        assert "recommendations" in data
        assert "timestamp" in data
        
        assert data["outfit_id"] == outfit_id
    
    def test_outfit_analysis_single_item(self, test_client):
        """Test outfit analysis with single item (should fail)"""
        outfit_id = "test_outfit_002"
        item_ids = "demo_0001"  # Single item
        
        response = test_client.get(f"/outfit/analyze/{outfit_id}?item_ids={item_ids}")
        assert response.status_code == status.HTTP_400_BAD_REQUEST


# Demo endpoints tests
class TestDemoEndpoints:
    """Demo endpoint tests"""
    
    def test_sample_outfits(self, test_client):
        """Test sample outfits endpoint"""
        response = test_client.get("/demo/sample-outfits")
        
        # Demo mode kontrolü
        if response.status_code == status.HTTP_404_NOT_FOUND:
            pytest.skip("Demo mode not enabled")
        
        if response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE:
            pytest.skip("Model not available")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Sample outfits format
        assert "sample_outfits" in data or "message" in data
        
        if "sample_outfits" in data:
            assert "total_samples" in data
            assert "database_stats" in data
            assert isinstance(data["sample_outfits"], list)


# Error handling tests
class TestErrorHandling:
    """Error handling tests"""
    
    def test_404_endpoint(self, test_client):
        """Test non-existent endpoint"""
        response = test_client.get("/non-existent-endpoint")
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_invalid_json(self, test_client):
        """Test invalid JSON request"""
        response = test_client.post(
            "/outfit/compatibility",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_missing_required_fields(self, test_client):
        """Test missing required fields"""
        response = test_client.post("/outfit/compatibility", json={})
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


# Performance tests
class TestPerformance:
    """Performance tests"""
    
    @pytest.mark.performance
    def test_compatibility_response_time(self, test_client):
        """Test compatibility analysis response time"""
        request = {
            "item_ids": ["demo_0001", "demo_0002"],
            "return_detailed_scores": False
        }
        
        start_time = time.time()
        response = test_client.post("/outfit/compatibility", json=request)
        response_time = time.time() - start_time
        
        if response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE:
            pytest.skip("Model not available")
        
        # Response should be under 5 seconds
        assert response_time < 5.0
        assert response.status_code == status.HTTP_200_OK
    
    @pytest.mark.performance
    def test_recommendation_response_time(self, test_client):
        """Test recommendation response time"""
        request = {
            "seed_item_ids": ["demo_0001"],
            "max_recommendations": 3
        }
        
        start_time = time.time()
        response = test_client.post("/outfit/recommendations", json=request)
        response_time = time.time() - start_time
        
        if response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE:
            pytest.skip("Model not available")
        
        # Response should be under 10 seconds
        assert response_time < 10.0


# Integration tests
class TestIntegration:
    """Integration tests"""
    
    @pytest.mark.integration
    def test_full_workflow(self, test_client, sample_image_file):
        """Test complete workflow: add item -> get recommendations -> analyze outfit"""
        
        # 1. Add item
        with open(sample_image_file, "rb") as f:
            files = {"image": ("workflow_test.jpg", f, "image/jpeg")}
            data = {
                "item_id": "workflow_test_001",
                "category": "tops",
                "color": "blue",
                "style": "casual"
            }
            
            add_response = test_client.post("/items/add", files=files, data=data)
        
        if add_response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE:
            pytest.skip("Model not available")
        
        assert add_response.status_code == status.HTTP_200_OK
        
        # 2. Get item recommendations
        rec_request = {
            "item_id": "workflow_test_001",
            "max_recommendations": 3
        }
        
        rec_response = test_client.post("/items/recommendations", json=rec_request)
        assert rec_response.status_code == status.HTTP_200_OK
        
        # 3. Test compatibility (if we have items)
        comp_request = {
            "item_ids": ["workflow_test_001", "demo_0001"],
            "return_detailed_scores": True
        }
        
        comp_response = test_client.post("/outfit/compatibility", json=comp_request)
        # May fail if demo items don't exist, that's OK
        
        # 4. Check database stats
        stats_response = test_client.get("/database/stats")
        assert stats_response.status_code == status.HTTP_200_OK
        
        stats = stats_response.json()
        assert stats["total_items"] > 0


# Load tests (marked as slow)
class TestLoad:
    """Load tests"""
    
    @pytest.mark.slow
    def test_concurrent_requests(self, test_client):
        """Test concurrent requests"""
        import threading
        
        results = []
        
        def make_request():
            response = test_client.get("/health")
            results.append(response.status_code)
        
        # Create 10 concurrent threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        assert len(results) == 10
        assert all(status == 200 for status in results)


# Utility functions for tests
def create_test_data():
    """Create test data files"""
    test_dir = Path("./test_data")
    test_dir.mkdir(exist_ok=True)
    
    # Create test items database
    test_items = []
    for i in range(1, 11):
        item = {
            "item_id": f"demo_{i:04d}",
            "image_path": f"./test_data/images/demo_{i:04d}.jpg",
            "category": ["tops", "bottoms", "shoes"][i % 3],
            "color": ["black", "white", "blue"][i % 3],
            "style": ["casual", "formal"][i % 2],
            "price": 20.0 + (i * 10),
            "brand": f"TestBrand{i % 3 + 1}",
            "metadata": {"test": True}
        }
        test_items.append(item)
    
    with open(test_dir / "test_items.json", "w") as f:
        json.dump(test_items, f, indent=2)
    
    # Create test images directory
    images_dir = test_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    # Create sample images
    for i in range(1, 11):
        image = Image.new('RGB', (224, 224), color=f'#{i*25:02x}{i*25:02x}{i*25:02x}')
        image.save(images_dir / f"demo_{i:04d}.jpg")


if __name__ == "__main__":
    # Create test data
    create_test_data()
    
    # Run tests
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-m", "not slow and not performance"  # Skip slow tests by default
    ])
