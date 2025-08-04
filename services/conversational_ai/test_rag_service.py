"""
Test Suite for RAG Service
Bu dosya, RAG Service'in tüm fonksiyonlarını kapsamlı olarak test eder.
"""

import os
import sys
import json
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import torch

# Test environment setup
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_service import RAGService, RAGConfig, create_rag_service, test_rag_service


class TestRAGConfig(unittest.TestCase):
    """RAGConfig sınıfını test eder"""
    
    def test_default_config(self):
        """Varsayılan konfigürasyon testi"""
        config = RAGConfig()
        
        self.assertEqual(config.base_model_name, "meta-llama/Meta-Llama-3-8B-Instruct")
        self.assertEqual(config.vector_store_type, "faiss")
        self.assertEqual(config.top_k_retrieval, 5)
        self.assertTrue(config.use_4bit_quantization)
        
    def test_device_auto_selection(self):
        """Otomatik device seçimi testi"""
        config = RAGConfig(device="auto")
        
        # Device should be set to cuda or cpu
        self.assertIn(config.device, ["cuda", "cpu"])
        
    def test_torch_dtype_conversion(self):
        """Torch dtype dönüşüm testi"""
        config = RAGConfig(torch_dtype="float16")
        self.assertEqual(config.torch_dtype, torch.float16)
        
        config = RAGConfig(torch_dtype="bfloat16") 
        self.assertEqual(config.torch_dtype, torch.bfloat16)
        
    def test_custom_config(self):
        """Özel konfigürasyon testi"""
        config = RAGConfig(
            top_k_retrieval=10,
            temperature=0.8,
            max_new_tokens=500
        )
        
        self.assertEqual(config.top_k_retrieval, 10)
        self.assertEqual(config.temperature, 0.8)
        self.assertEqual(config.max_new_tokens, 500)


class TestRAGServiceMocked(unittest.TestCase):
    """RAG Service'i mock'larla test eder (model yüklemeden)"""
    
    def setUp(self):
        """Test setup - mock'ları hazırla"""
        self.config = RAGConfig(
            finetuned_model_path="./test_models/mock_model",
            vector_store_path="./test_data/mock_index.faiss",
            use_4bit_quantization=False  # Test için daha hızlı
        )
    
    @patch('src.rag_service.AutoTokenizer')
    @patch('src.rag_service.AutoModelForCausalLM')
    @patch('src.rag_service.SentenceTransformer')
    @patch('src.rag_service.faiss')
    def test_rag_service_initialization(self, mock_faiss, mock_sentence_transformer, 
                                       mock_model, mock_tokenizer):
        """RAG Service başlatma testi"""
        
        # Mock setup
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()
        mock_sentence_transformer.return_value = Mock()
        mock_faiss.read_index.return_value = Mock()
        
        # Mock embedding model behavior
        mock_embedding_model = Mock()
        mock_embedding_model.encode.return_value = np.random.random(384)
        mock_sentence_transformer.return_value = mock_embedding_model
        
        # Mock FAISS index
        mock_index = Mock()
        mock_index.ntotal = 1000
        mock_faiss.read_index.return_value = mock_index
        
        # Test initialization
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open_metadata()):
                service = RAGService(self.config)
                
                self.assertIsNotNone(service.model)
                self.assertIsNotNone(service.tokenizer)
                self.assertIsNotNone(service.embedding_model)
                self.assertIsNotNone(service.vector_store)
    
    def test_encode_query(self):
        """Query encoding testi"""
        with patch('src.rag_service.SentenceTransformer') as mock_st:
            mock_embedding_model = Mock()
            mock_embedding_model.encode.return_value = np.random.random(384)
            mock_st.return_value = mock_embedding_model
            
            service = Mock()
            service.embedding_model = mock_embedding_model
            service._encode_query = RAGService._encode_query.__get__(service)
            
            result = service._encode_query("test query")
            
            self.assertIsInstance(result, np.ndarray)
            mock_embedding_model.encode.assert_called_once()
    
    def test_format_context_for_prompt(self):
        """Context formatting testi"""
        service = Mock()
        service.config = self.config
        service._format_context_for_prompt = RAGService._format_context_for_prompt.__get__(service)
        
        # Test data
        retrieved_items = [
            {
                "content": "Mavi denim ceket",
                "metadata": {"category": "jacket", "color": "blue"}
            },
            {
                "content": "Beyaz pamuk tişört", 
                "metadata": {"category": "shirt", "color": "white"}
            }
        ]
        
        result = service._format_context_for_prompt(retrieved_items)
        
        self.assertIn("Mavi denim ceket", result)
        self.assertIn("Beyaz pamuk tişört", result)
        self.assertIn("Kategori: jacket", result)
    
    def test_construct_prompt(self):
        """Prompt construction testi"""
        service = Mock()
        service.system_prompt = "System: {context}\nUser: {query}\nAssistant:"
        service.fallback_prompt = "User: {query}\nAssistant:"
        service._construct_prompt = RAGService._construct_prompt.__get__(service)
        
        # Test with context
        result1 = service._construct_prompt("test query", "test context")
        self.assertIn("test context", result1)
        self.assertIn("test query", result1)
        
        # Test without context
        result2 = service._construct_prompt("test query", "Gardırop bilgisi bulunamadı.")
        self.assertNotIn("test context", result2)
        self.assertIn("test query", result2)


class TestRAGServiceIntegration(unittest.TestCase):
    """Integration testleri - gerçek dosyalar gerekir"""
    
    def setUp(self):
        """Test için geçici dosyalar oluştur"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock metadata dosyası
        self.metadata_path = os.path.join(self.temp_dir, "metadata.json")
        self.mock_metadata = {
            "0": {
                "content": "Mavi denim ceket, slim fit",
                "user_id": "test_user",
                "category": "jacket",
                "color": "blue"
            },
            "1": {
                "content": "Beyaz pamuk tişört, basic",
                "user_id": "test_user", 
                "category": "shirt",
                "color": "white"
            }
        }
        
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.mock_metadata, f)
    
    def tearDown(self):
        """Test sonrası temizlik"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_metadata(self):
        """Metadata yükleme testi"""
        config = RAGConfig(metadata_path=self.metadata_path)
        
        with patch.multiple('src.rag_service.RAGService',
                          _load_llm_model=Mock(),
                          _load_embedding_model=Mock(), 
                          _load_vector_store=Mock(),
                          _setup_prompt_templates=Mock()):
            
            service = RAGService(config)
            service._load_metadata()
            
            self.assertEqual(len(service.metadata_store), 2)
            self.assertIn("0", service.metadata_store)
            self.assertEqual(service.metadata_store["0"]["color"], "blue")
    
    @patch('src.rag_service.faiss')
    def test_search_faiss_mock(self, mock_faiss):
        """FAISS search mock testi"""
        # Mock FAISS index
        mock_index = Mock()
        mock_index.search.return_value = (
            np.array([[0.9, 0.8, 0.7]]),  # similarities
            np.array([[0, 1, 2]])         # indices
        )
        
        service = Mock()
        service.config = RAGConfig(
            metadata_path=self.metadata_path,
            similarity_threshold=0.6
        )
        service.vector_store = mock_index
        service.metadata_store = self.mock_metadata
        service._search_faiss = RAGService._search_faiss.__get__(service)
        
        query_embedding = np.random.random(384)
        results = service._search_faiss(query_embedding, "test_user", 5)
        
        self.assertGreater(len(results), 0)
        self.assertIn("similarity", results[0])
        self.assertIn("content", results[0])
    
    def test_batch_generate_responses_mock(self):
        """Batch generation mock testi"""
        service = Mock()
        service.batch_generate_responses = RAGService.batch_generate_responses.__get__(service)
        
        # Mock generate_response method
        def mock_generate_response(query, user_id):
            return {
                "success": True,
                "response": f"Mock response for: {query}",
                "metadata": {"user_id": user_id}
            }
        
        service.generate_response = mock_generate_response
        
        queries = [
            {"query": "Ne giysem?", "user_id": "user1"},
            {"query": "Hangi ayakkabı?", "user_id": "user2"}
        ]
        
        results = service.batch_generate_responses(queries)
        
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r["success"] for r in results))
    
    def test_service_stats(self):
        """Service stats testi"""
        service = Mock()
        service.config = self.config = RAGConfig()
        service.model = Mock()
        service.embedding_model = Mock()
        service.vector_store = Mock()
        service.vector_store.ntotal = 1000
        service.vector_store.d = 384
        service.device = torch.device("cpu")
        service.metadata_store = self.mock_metadata
        service.get_service_stats = RAGService.get_service_stats.__get__(service)
        
        stats = service.get_service_stats()
        
        self.assertEqual(stats["service_status"], "active")
        self.assertTrue(stats["models_loaded"]["llm_model"])
        self.assertEqual(stats["vector_store_stats"]["total_vectors"], 1000)
        self.assertEqual(stats["metadata_stats"]["total_items"], 2)


class TestUtilityFunctions(unittest.TestCase):
    """Utility fonksiyonlarını test eder"""
    
    def test_create_rag_service_with_kwargs(self):
        """create_rag_service with kwargs testi"""
        with patch('src.rag_service.RAGService') as mock_service:
            create_rag_service(
                base_model_name="test_model",
                top_k_retrieval=10
            )
            
            # RAGConfig'in doğru parametrelerle çağrıldığını kontrol et
            mock_service.assert_called_once()
            config_arg = mock_service.call_args[0][0]
            self.assertEqual(config_arg.base_model_name, "test_model")
            self.assertEqual(config_arg.top_k_retrieval, 10)
    
    def test_create_rag_service_with_config_file(self):
        """create_rag_service with config file testi"""
        temp_config = {
            "base_model_name": "test_model_from_file",
            "temperature": 0.9
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(temp_config, f)
            config_path = f.name
        
        try:
            with patch('src.rag_service.RAGService') as mock_service:
                create_rag_service(config_path=config_path)
                
                config_arg = mock_service.call_args[0][0]
                self.assertEqual(config_arg.base_model_name, "test_model_from_file")
                self.assertEqual(config_arg.temperature, 0.9)
        finally:
            os.unlink(config_path)


def mock_open_metadata():
    """Metadata dosyası için mock open"""
    mock_metadata = {
        "0": {"content": "test item", "user_id": "test"}
    }
    
    from unittest.mock import mock_open
    return mock_open(read_data=json.dumps(mock_metadata))


def run_tests():
    """Tüm testleri çalıştır"""
    print("🧪 RAG Service Test Suite başlatılıyor...\n")
    
    # Test suite oluştur
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Test sınıflarını ekle
    test_classes = [
        TestRAGConfig,
        TestRAGServiceMocked,
        TestRAGServiceIntegration,
        TestUtilityFunctions
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Testleri çalıştır
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Sonuçları raporla
    print(f"\n📊 Test Sonuçları:")
    print(f"✅ Başarılı: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Başarısız: {len(result.failures)}")  
    print(f"💥 Hata: {len(result.errors)}")
    
    if result.failures:
        print(f"\n📝 Başarısız Testler:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.splitlines()[-1]}")
    
    if result.errors:
        print(f"\n🔥 Hatalı Testler:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.splitlines()[-1]}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    
    if success:
        print("\n🎉 Tüm testler başarılı!")
        sys.exit(0)
    else:
        print("\n💥 Bazı testler başarısız!")
        sys.exit(1)
