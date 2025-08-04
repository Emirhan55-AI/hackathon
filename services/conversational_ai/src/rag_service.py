"""
RAG Service - Retrieval-Augmented Generation for Aura Conversational AI
Bu modül, hibrit QLoRA fine-tuning + RAG sistemi kullanarak kişiselleştirilmiş
fashion asistanı yanıtları üretir.

Sistem Mimarisi:
1. Query Embedding: Kullanıcı sorgusunu vektöre çevirir
2. Vector Search: Kullanıcının gardırobunda semantik arama yapar
3. Context Retrieval: İlgili kıyafet bilgilerini geri çağırır
4. LLM Generation: Fine-tuned model ile bağlamsal yanıt üretir
"""

import os
import sys
import json
import logging
import time
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import traceback
import numpy as np
from dataclasses import dataclass, field

# Core ML libraries
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig
)
from peft import PeftModel, PeftConfig
from sentence_transformers import SentenceTransformer

# Vector store libraries
import faiss
try:
    import pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    pinecone = None

# Logging setup
from loguru import logger

# Local imports
try:
    from src.finetune import FineTuningConfig, load_model_and_tokenizer
    from src.build_vector_store import VectorStoreConfig, load_embedding_model, initialize_vector_store
except ImportError:
    # Fallback for development
    try:
        from finetune import FineTuningConfig, load_model_and_tokenizer
        from build_vector_store import VectorStoreConfig, load_embedding_model, initialize_vector_store
    except ImportError:
        # Graceful degradation
        FineTuningConfig = None
        load_model_and_tokenizer = None
        VectorStoreConfig = None
        load_embedding_model = None
        initialize_vector_store = None

# RAG Configuration
@dataclass
class RAGConfig:
    """RAG servisi için konfigürasyon sınıfı"""
    
    # Model paths
    base_model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    finetuned_model_path: str = "./saved_models/aura_fashion_assistant"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Vector store configuration
    vector_store_type: str = "faiss"  # "faiss" or "pinecone"
    vector_store_path: str = "./vector_stores/wardrobe_faiss.index"
    metadata_path: str = "./vector_stores/wardrobe_metadata.json"
    
    # Pinecone configuration (if used)
    pinecone_api_key: Optional[str] = None
    pinecone_environment: str = "us-west1-gcp-free"
    pinecone_index_name: str = "aura-wardrobe"
    
    # Retrieval parameters
    top_k_retrieval: int = 5
    similarity_threshold: float = 0.7
    max_context_length: int = 1500
    
    # Generation parameters
    max_new_tokens: int = 300
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    repetition_penalty: float = 1.1
    
    # System configuration
    device: str = "auto"
    torch_dtype: str = "float16"
    use_4bit_quantization: bool = True
    
    def __post_init__(self):
        """Konfigürasyon doğrulama ve otomatik değer ayarlama"""
        # Device selection - Force CPU for embedding in Docker
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Force CPU for embedding model in containerized environments
        self.embedding_device = "cpu"  # Always use CPU for embeddings
        
        # Torch dtype conversion
        if self.torch_dtype == "float16":
            self.torch_dtype = torch.float16
        elif self.torch_dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16
        else:
            self.torch_dtype = torch.float32
        
        # Pinecone API key from environment
        if not self.pinecone_api_key:
            self.pinecone_api_key = os.getenv("PINECONE_API_KEY")


class RAGService:
    """
    A2. Retrieval-Augmented Generation Service
    
    Bu sınıf, hibrit QLoRA fine-tuning + RAG sistemi kullanarak
    kişiselleştirilmiş fashion asistanı yanıtları üretir.
    """
    
    def __init__(self, config: RAGConfig):
        """
        RAG servisini başlatır ve gerekli modelleri yükler
        
        Args:
            config: RAG konfigürasyon nesnesi
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # Force CPU mode if CUDA not available or device is cpu
        if self.device.type == "cpu" or not torch.cuda.is_available():
            # Disable CUDA globally for this process
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            self.device = torch.device("cpu")
            logger.info("🖥️ CUDA devre dışı - CPU modunda çalışılıyor")
        
        logger.info("🚀 RAG Service başlatılıyor...")
        logger.info(f"📱 Device: {self.device}")
        logger.info(f"🧠 Base Model: {config.base_model_name}")
        logger.info(f"🎯 Fine-tuned Model: {config.finetuned_model_path}")
        logger.info(f"🔍 Vector Store: {config.vector_store_type}")
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.embedding_model = None
        self.vector_store = None
        self.metadata_store = None
        
        # Load all components
        self._load_llm_model()
        self._load_embedding_model()
        self._load_vector_store()
        
        # Setup prompt templates
        self._setup_prompt_templates()
        
        logger.info("✅ RAG Service başarıyla başlatıldı!")
    
    def _load_llm_model(self):
        """
        A2a. Fine-tuned LLM modelini ve tokenizer'ı yükler
        """
        try:
            logger.info("🔄 Fine-tuned LLM modeli yükleniyor...")
            
            # Check for HuggingFace authentication
            hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
            use_auth_token = hf_token if hf_token and hf_token != "your_huggingface_token_here" else None
            
            if use_auth_token:
                logger.info("🔑 HuggingFace authentication token bulundu")
            else:
                logger.warning("⚠️ HuggingFace token bulunamadı - gated modeller için sorun olabilir")
            
            # Quantization config for memory efficiency (only for CUDA)
            if self.config.use_4bit_quantization and self.device.type == "cuda" and torch.cuda.is_available():
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=self.config.torch_dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_storage=self.config.torch_dtype,
                )
            else:
                quantization_config = None
                logger.info("🔄 CPU mode: 4-bit quantization devre dışı")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.base_model_name,
                trust_remote_code=True,
                padding_side="left",
                use_auth_token=use_auth_token
            )
            
            # Ensure pad token exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Device configuration for model loading
            use_device_map = self.device.type == "cuda" and torch.cuda.is_available()
            
            # Load base model
            model_kwargs = {
                "quantization_config": quantization_config,
                "torch_dtype": self.config.torch_dtype,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
                "use_auth_token": use_auth_token,
                "offload_state_dict": True
            }
            
            if use_device_map:
                model_kwargs.update({
                    "device_map": "auto",
                    "max_memory": {0: "6GiB"},
                    "offload_folder": "./offload"
                })
                
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model_name,
                **model_kwargs
            )
            
            # Move to device if not using device_map
            if not use_device_map:
                self.model = self.model.to(self.device)
            
            # Load PEFT adapters if available
            if os.path.exists(self.config.finetuned_model_path):
                logger.info(f"🎯 PEFT adaptörleri yükleniyor: {self.config.finetuned_model_path}")
                self.model = PeftModel.from_pretrained(
                    self.model,
                    self.config.finetuned_model_path,
                    torch_dtype=self.config.torch_dtype
                )
                self.model = self.model.merge_and_unload()  # Merge LoRA weights
            else:
                logger.warning(f"⚠️ Fine-tuned model bulunamadı: {self.config.finetuned_model_path}")
                logger.warning("📝 Base model kullanılacak")
            
            # Set model to evaluation mode
            self.model.eval()
            
            logger.info("✅ LLM modeli başarıyla yüklendi!")
            
        except Exception as e:
            logger.error(f"❌ LLM model yükleme hatası: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def _load_embedding_model(self):
        """
        A2b. Sentence embedding modelini yükler
        """
        try:
            logger.info("🔄 Embedding modeli yükleniyor...")
            
            # Use dedicated embedding device (CPU)
            embedding_device = getattr(self.config, 'embedding_device', 'cpu')
            logger.info(f"🖥️ Embedding device: {embedding_device}")
            
            self.embedding_model = SentenceTransformer(
                self.config.embedding_model_name,
                device=embedding_device
            )
            
            # Test embedding to ensure model works
            test_embedding = self.embedding_model.encode("test", convert_to_tensor=False)
            logger.info(f"📏 Embedding dimension: {len(test_embedding)}")
            
            logger.info("✅ Embedding modeli başarıyla yüklendi!")
            
        except Exception as e:
            logger.error(f"❌ Embedding model yükleme hatası: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def _load_vector_store(self):
        """
        A2c. Vector store ve metadata'yı yükler
        """
        try:
            logger.info("🔄 Vector store yükleniyor...")
            
            if self.config.vector_store_type == "faiss":
                self._load_faiss_store()
            elif self.config.vector_store_type == "pinecone":
                self._load_pinecone_store()
            else:
                raise ValueError(f"Desteklenmeyen vector store türü: {self.config.vector_store_type}")
            
            # Load metadata
            self._load_metadata()
            
            logger.info("✅ Vector store başarıyla yüklendi!")
            
        except Exception as e:
            logger.error(f"❌ Vector store yükleme hatası: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def _load_faiss_store(self):
        """FAISS vector store'u yükler"""
        if not os.path.exists(self.config.vector_store_path):
            raise FileNotFoundError(f"FAISS index bulunamadı: {self.config.vector_store_path}")
        
        self.vector_store = faiss.read_index(self.config.vector_store_path)
        logger.info(f"📊 FAISS index yüklendi: {self.vector_store.ntotal} vektör")
    
    def _load_pinecone_store(self):
        """Pinecone vector store'u yükler"""
        if not PINECONE_AVAILABLE:
            raise ImportError("Pinecone kütüphanesi yüklenmemiş. pip install pinecone-client")
        
        if not self.config.pinecone_api_key:
            raise ValueError("Pinecone API key gerekli. PINECONE_API_KEY environment variable'ını ayarlayın.")
        
        # Initialize Pinecone
        pinecone.init(
            api_key=self.config.pinecone_api_key,
            environment=self.config.pinecone_environment
        )
        
        # Connect to index
        self.vector_store = pinecone.Index(self.config.pinecone_index_name)
        
        # Get index stats
        stats = self.vector_store.describe_index_stats()
        logger.info(f"📊 Pinecone index bağlandı: {stats['total_vector_count']} vektör")
    
    def _load_metadata(self):
        """Metadata dosyasını yükler"""
        if os.path.exists(self.config.metadata_path):
            with open(self.config.metadata_path, 'r', encoding='utf-8') as f:
                self.metadata_store = json.load(f)
            logger.info(f"📄 Metadata yüklendi: {len(self.metadata_store)} kayıt")
        else:
            logger.warning(f"⚠️ Metadata dosyası bulunamadı: {self.config.metadata_path}")
            self.metadata_store = {}
    
    def _setup_prompt_templates(self):
        """
        A2d. Prompt şablonlarını ayarlar
        """
        self.system_prompt = """Sen Aura, kullanıcının kişisel moda asistanısın. Görüştüğün kullanıcıya ait gardırop bilgilerine erişimin var ve bu bilgileri kullanarak kişiselleştirilmiş moda önerileri veriyorsun.

Özellikler:
- Samimi, dostane ve yardımsever bir tonda konuş
- Kullanıcının mevcut kıyafetlerini dikkate al
- Renk kombinasyonları, stil uyumu ve uygunluk öner
- Mevsim ve durum bazlı öneriler yap
- Kısa ve net cevaplar ver

Gardırop Bilgileri:
{context}

Kullanıcı Sorusu: {query}

Aura'nın Yanıtı:"""

        self.fallback_prompt = """Sen Aura, kişisel moda asistanısın. Kullanıcının sorusunu genel moda bilgin ile yanıtla.

Kullanıcı Sorusu: {query}

Aura'nın Yanıtı:"""
    
    def generate_response(self, query: str, user_id: str) -> Dict[str, Any]:
        """
        A2e. Ana RAG pipeline - Kullanıcı sorgusuna yanıt üretir
        
        Args:
            query: Kullanıcının doğal dil sorgusu
            user_id: Kullanıcının benzersiz kimliği
            
        Returns:
            Dict: Yanıt ve metadata içeren sözlük
        """
        start_time = time.time()
        
        try:
            logger.info(f"🎯 RAG Pipeline başlatılıyor: {user_id}")
            logger.info(f"❓ Query: {query}")
            
            # 1. Query embedding
            query_embedding = self._encode_query(query)
            
            # 2. Vector search
            retrieved_items = self._search_context(query_embedding, user_id, self.config.top_k_retrieval)
            
            # 3. Context formatting
            context_str = self._format_context_for_prompt(retrieved_items)
            
            # 4. Prompt construction
            prompt = self._construct_prompt(query, context_str)
            
            # 5. LLM generation
            response_text = self._generate_with_llm(prompt)
            
            # 6. Response formatting
            processing_time = time.time() - start_time
            
            result = {
                "success": True,
                "response": response_text.strip(),
                "context_used": retrieved_items,
                "metadata": {
                    "user_id": user_id,
                    "query": query,
                    "retrieved_items_count": len(retrieved_items),
                    "processing_time": processing_time,
                    "model_used": self.config.base_model_name,
                    "vector_store_type": self.config.vector_store_type,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            }
            
            logger.info(f"✅ Response generated: {processing_time:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ RAG pipeline hatası: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Fallback response
            return {
                "success": False,
                "response": "Üzgünüm, şu anda bir teknik sorun yaşıyorum. Lütfen sorunuzu tekrar deneyin.",
                "error": str(e),
                "metadata": {
                    "user_id": user_id,
                    "query": query,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            }
    
    def _encode_query(self, query: str) -> np.ndarray:
        """
        A3a. Kullanıcı sorgusunu vektöre çevirir
        
        Args:
            query: Kullanıcı sorgusu
            
        Returns:
            np.ndarray: Query embedding vektörü
        """
        try:
            embedding = self.embedding_model.encode(
                query,
                convert_to_tensor=False,
                normalize_embeddings=True
            )
            return embedding
            
        except Exception as e:
            logger.error(f"❌ Query encoding hatası: {str(e)}")
            raise
    
    def _search_context(self, query_embedding: np.ndarray, user_id: str, top_k: int) -> List[Dict[str, Any]]:
        """
        A3b. Vector store'da benzerlik araması yapar
        
        Args:
            query_embedding: Query'nin embedding vektörü
            user_id: Kullanıcı kimliği
            top_k: Döndürülecek maksimum sonuç sayısı
            
        Returns:
            List[Dict]: Bulunan benzer öğeler listesi
        """
        try:
            if self.config.vector_store_type == "faiss":
                return self._search_faiss(query_embedding, user_id, top_k)
            elif self.config.vector_store_type == "pinecone":
                return self._search_pinecone(query_embedding, user_id, top_k)
            else:
                raise ValueError(f"Desteklenmeyen vector store: {self.config.vector_store_type}")
                
        except Exception as e:
            logger.error(f"❌ Vector search hatası: {str(e)}")
            return []
    
    def _search_faiss(self, query_embedding: np.ndarray, user_id: str, top_k: int) -> List[Dict[str, Any]]:
        """FAISS ile benzerlik araması"""
        # Query embedding'i FAISS format'ına çevir
        query_vector = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Search
        similarities, indices = self.vector_store.search(query_vector, top_k * 2)  # Extra for filtering
        
        retrieved_items = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx == -1:  # Invalid index
                continue
                
            # Get metadata for this item
            if str(idx) in self.metadata_store:
                item_metadata = self.metadata_store[str(idx)]
                
                # Filter by user_id if specified
                if user_id and item_metadata.get("user_id") != user_id:
                    continue
                
                # Filter by similarity threshold
                if similarity < self.config.similarity_threshold:
                    continue
                
                retrieved_item = {
                    "id": idx,
                    "similarity": float(similarity),
                    "content": item_metadata.get("content", ""),
                    "metadata": item_metadata
                }
                retrieved_items.append(retrieved_item)
                
                if len(retrieved_items) >= top_k:
                    break
        
        logger.info(f"🔍 FAISS search: {len(retrieved_items)} items found")
        return retrieved_items
    
    def _search_pinecone(self, query_embedding: np.ndarray, user_id: str, top_k: int) -> List[Dict[str, Any]]:
        """Pinecone ile benzerlik araması"""
        # Build filter for user_id
        filter_dict = {"user_id": {"$eq": user_id}} if user_id else None
        
        # Search
        search_results = self.vector_store.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            filter=filter_dict,
            include_metadata=True,
            include_values=False
        )
        
        retrieved_items = []
        for match in search_results.matches:
            if match.score < self.config.similarity_threshold:
                continue
                
            retrieved_item = {
                "id": match.id,
                "similarity": float(match.score),
                "content": match.metadata.get("content", ""),
                "metadata": match.metadata
            }
            retrieved_items.append(retrieved_item)
        
        logger.info(f"🔍 Pinecone search: {len(retrieved_items)} items found")
        return retrieved_items
    
    def _format_context_for_prompt(self, retrieved_items: List[Dict[str, Any]]) -> str:
        """
        A3c. Geri çağırılan öğeleri LLM prompt'u için formatlar
        
        Args:
            retrieved_items: Vector search sonuçları
            
        Returns:
            str: Formatlanmış context string
        """
        if not retrieved_items:
            return "Gardırop bilgisi bulunamadı."
        
        context_parts = []
        total_length = 0
        
        for item in retrieved_items:
            content = item.get("content", "")
            metadata = item.get("metadata", {})
            
            # Format item information
            formatted_item = f"- {content}"
            
            # Add additional metadata if available
            if metadata.get("category"):
                formatted_item += f" (Kategori: {metadata['category']})"
            if metadata.get("color"):
                formatted_item += f" (Renk: {metadata['color']})"
            if metadata.get("brand"):
                formatted_item += f" (Marka: {metadata['brand']})"
            
            # Check length limit
            if total_length + len(formatted_item) > self.config.max_context_length:
                break
            
            context_parts.append(formatted_item)
            total_length += len(formatted_item)
        
        context_str = "\n".join(context_parts)
        logger.debug(f"📝 Context formatted: {len(context_str)} characters")
        
        return context_str
    
    def _construct_prompt(self, query: str, context_str: str) -> str:
        """
        A3d. LLM için nihai prompt'u oluşturur
        
        Args:
            query: Kullanıcı sorgusu
            context_str: Formatlanmış context
            
        Returns:
            str: LLM için hazır prompt
        """
        if context_str and context_str != "Gardırop bilgisi bulunamadı.":
            prompt = self.system_prompt.format(
                context=context_str,
                query=query
            )
        else:
            prompt = self.fallback_prompt.format(query=query)
        
        logger.debug(f"📋 Prompt constructed: {len(prompt)} characters")
        return prompt
    
    def _generate_with_llm(self, prompt: str) -> str:
        """
        LLM ile text generation yapar
        
        Args:
            prompt: LLM için hazırlanmış prompt
            
        Returns:
            str: Generated response
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                padding=False
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generation config
            generation_config = GenerationConfig(
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=self.config.do_sample,
                repetition_penalty=self.config.repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=generation_config
                )
            
            # Decode response
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            return response
            
        except Exception as e:
            logger.error(f"❌ LLM generation hatası: {str(e)}")
            raise
    
    def batch_generate_responses(self, queries: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Çoklu sorgu için batch processing
        
        Args:
            queries: [{"query": "...", "user_id": "..."}, ...] formatında liste
            
        Returns:
            List[Dict]: Her sorgu için yanıt listesi
        """
        logger.info(f"📦 Batch processing başlatılıyor: {len(queries)} sorgu")
        
        results = []
        for i, query_data in enumerate(queries):
            logger.info(f"⏳ Batch: {i+1}/{len(queries)}")
            
            result = self.generate_response(
                query=query_data["query"],
                user_id=query_data["user_id"]
            )
            results.append(result)
        
        logger.info(f"✅ Batch processing tamamlandı: {len(results)} sonuç")
        return results
    
    def get_service_stats(self) -> Dict[str, Any]:
        """
        RAG servisinin durumu ve istatistikleri
        
        Returns:
            Dict: Servis istatistikleri
        """
        stats = {
            "service_status": "active",
            "models_loaded": {
                "llm_model": self.model is not None,
                "embedding_model": self.embedding_model is not None,
                "vector_store": self.vector_store is not None
            },
            "configuration": {
                "base_model": self.config.base_model_name,
                "embedding_model": self.config.embedding_model_name,
                "vector_store_type": self.config.vector_store_type,
                "device": str(self.device)
            },
            "vector_store_stats": {},
            "generation_params": {
                "max_new_tokens": self.config.max_new_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "top_k_retrieval": self.config.top_k_retrieval
            }
        }
        
        # Vector store specific stats
        if self.config.vector_store_type == "faiss" and self.vector_store:
            stats["vector_store_stats"] = {
                "total_vectors": self.vector_store.ntotal,
                "dimension": self.vector_store.d,
                "index_type": str(type(self.vector_store).__name__)
            }
        elif self.config.vector_store_type == "pinecone" and self.vector_store:
            try:
                pinecone_stats = self.vector_store.describe_index_stats()
                stats["vector_store_stats"] = {
                    "total_vectors": pinecone_stats.get('total_vector_count', 0),
                    "dimension": pinecone_stats.get('dimension', 0)
                }
            except:
                stats["vector_store_stats"] = {"error": "Could not fetch Pinecone stats"}
        
        # Metadata stats
        if self.metadata_store:
            stats["metadata_stats"] = {
                "total_items": len(self.metadata_store),
                "unique_users": len(set(
                    item.get("user_id", "unknown") 
                    for item in self.metadata_store.values() 
                    if isinstance(item, dict)
                ))
            }
        
        return stats
    
    def clear_cache(self):
        """Memory cache'i temizler"""
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("🧹 Cache temizlendi")
    
    def __del__(self):
        """Destructor - cleanup işlemleri"""
        try:
            self.clear_cache()
        except:
            pass


# Utility Functions
def create_rag_service(config_path: Optional[str] = None, **kwargs) -> RAGService:
    """
    RAG servisini kolayca oluşturmak için factory function
    
    Args:
        config_path: JSON config dosyası yolu (opsiyonel)
        **kwargs: RAGConfig parametreleri
        
    Returns:
        RAGService: Hazır RAG servisi
    """
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        config_dict.update(kwargs)
        config = RAGConfig(**config_dict)
    else:
        config = RAGConfig(**kwargs)
    
    return RAGService(config)


def test_rag_service(
    model_path: str = "./saved_models/aura_fashion_assistant",
    vector_store_path: str = "./vector_stores/wardrobe_faiss.index"
) -> bool:
    """
    RAG servisini test eder
    
    Args:
        model_path: Fine-tuned model yolu
        vector_store_path: Vector store yolu
        
    Returns:
        bool: Test başarılı oldu mu
    """
    try:
        logger.info("🧪 RAG Service test başlatılıyor...")
        
        # Test config
        config = RAGConfig(
            finetuned_model_path=model_path,
            vector_store_path=vector_store_path,
            max_new_tokens=50  # Quick test
        )
        
        # Create service
        service = RAGService(config)
        
        # Test query
        test_result = service.generate_response(
            query="Bugün ne giysem iyi olur?",
            user_id="test_user"
        )
        
        if test_result["success"]:
            logger.info("✅ RAG Service test başarılı!")
            logger.info(f"📝 Test yanıtı: {test_result['response'][:100]}...")
            return True
        else:
            logger.error("❌ RAG Service test başarısız!")
            return False
            
    except Exception as e:
        logger.error(f"❌ RAG Service test hatası: {str(e)}")
        return False


if __name__ == "__main__":
    """Test ve debug amaçlı direkt çalıştırma"""
    
    # Enable debug logging
    logger.add("rag_service_debug.log", level="DEBUG")
    
    # Run test
    success = test_rag_service()
    
    if success:
        logger.info("🎉 RAG Service hazır!")
    else:
        logger.error("💥 RAG Service test başarısız!")
        sys.exit(1)
