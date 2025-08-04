"""
RAG Service Configuration Examples
Bu dosya, RAG Service için farklı use case'lere uygun örnek konfigürasyonlar içerir.
"""

from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from src.rag_service import RAGConfig
else:
    try:
        from src.rag_service import RAGConfig
    except ImportError:
        try:
            from rag_service import RAGConfig
        except ImportError:
            # Graceful degradation
            RAGConfig = None

def basic_rag_config() -> Union["RAGConfig", None]:
    """
    Temel RAG konfigürasyonu - Development ve test için
    """
    return RAGConfig(
        # Model paths
        base_model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        finetuned_model_path="/app/models/aura_fashion_assistant",
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        
        # Vector store
        vector_store_type="faiss",
        vector_store_path="/app/vector_store/wardrobe_faiss.index",
        metadata_path="/app/vector_store/wardrobe_metadata.json",
        
        # Retrieval params
        top_k_retrieval=5,
        similarity_threshold=0.7,
        max_context_length=1500,
        
        # Generation params
        max_new_tokens=300,
        temperature=0.7,
        top_p=0.9,
        
        # System
        device="auto",
        use_4bit_quantization=True
    )


def production_rag_config() -> Union["RAGConfig", None]:
    """
    Production RAG konfigürasyonu - Yüksek performans ve kalite
    """
    return RAGConfig(
        # Model paths
        base_model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        finetuned_model_path="/app/models/aura_fashion_assistant_v2",
        embedding_model_name="sentence-transformers/all-mpnet-base-v2",  # Daha iyi embedding
        
        # Vector store
        vector_store_type="pinecone",  # Scalable cloud vector store
        pinecone_index_name="aura-wardrobe-prod",
        pinecone_environment="us-west1-gcp-free",
        
        # Retrieval params
        top_k_retrieval=8,  # Daha fazla context
        similarity_threshold=0.75,  # Daha yüksek threshold
        max_context_length=2000,  # Daha uzun context
        
        # Generation params
        max_new_tokens=400,
        temperature=0.6,  # Daha conservative
        top_p=0.85,
        repetition_penalty=1.15,
        
        # System
        device="cuda",
        torch_dtype="bfloat16",  # Daha iyi precision
        use_4bit_quantization=False  # Production'da full precision
    )


def memory_optimized_rag_config() -> Union["RAGConfig", None]:
    """
    Bellek optimizeli RAG konfigürasyonu - Düşük memory sistemler için
    """
    return RAGConfig(
        # Model paths
        base_model_name="microsoft/DialoGPT-medium",  # Daha küçük model
        finetuned_model_path="/app/models/aura_fashion_assistant_light",
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        
        # Vector store
        vector_store_type="faiss",
        vector_store_path="/app/vector_store/wardrobe_faiss.index",
        metadata_path="/app/vector_store/wardrobe_metadata.json",
        
        # Retrieval params
        top_k_retrieval=3,  # Daha az context
        similarity_threshold=0.65,
        max_context_length=800,  # Kısa context
        
        # Generation params
        max_new_tokens=150,  # Kısa responses
        temperature=0.8,
        top_p=0.9,
        
        # System
        device="cpu",  # CPU only
        torch_dtype="float32",
        use_4bit_quantization=True  # Aggressive quantization
    )


def fast_inference_rag_config() -> Union["RAGConfig", None]:
    """
    Hızlı inference RAG konfigürasyonu - Düşük latency için
    """
    return RAGConfig(
        # Model paths
        base_model_name="microsoft/DialoGPT-small",  # En küçük model
        finetuned_model_path="/app/models/aura_fashion_assistant_fast",
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        
        # Vector store
        vector_store_type="faiss",
        vector_store_path="/app/vector_store/wardrobe_faiss.index",
        metadata_path="/app/vector_store/wardrobe_metadata.json",
        
        # Retrieval params
        top_k_retrieval=3,
        similarity_threshold=0.6,
        max_context_length=500,
        
        # Generation params
        max_new_tokens=100,  # Çok kısa responses
        temperature=0.9,  # Hızlı sampling
        top_p=0.95,
        do_sample=True,
        
        # System
        device="cuda",
        torch_dtype="float16",
        use_4bit_quantization=True
    )


def multilingual_rag_config() -> Union["RAGConfig", None]:
    """
    Çok dilli RAG konfigürasyonu - Türkçe + İngilizce
    """
    return RAGConfig(
        # Model paths
        base_model_name="microsoft/DialoGPT-medium",
        finetuned_model_path="./saved_models/aura_fashion_assistant_multilingual",
        embedding_model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        
        # Vector store
        vector_store_type="faiss",
        vector_store_path="./vector_stores/wardrobe_multilingual_faiss.index",
        metadata_path="./vector_stores/wardrobe_multilingual_metadata.json",
        
        # Retrieval params
        top_k_retrieval=6,
        similarity_threshold=0.65,  # Multilingual için daha düşük
        max_context_length=1800,
        
        # Generation params
        max_new_tokens=350,
        temperature=0.75,
        top_p=0.9,
        
        # System
        device="auto",
        use_4bit_quantization=True
    )


def debug_rag_config() -> Union["RAGConfig", None]:
    """
    Debug RAG konfigürasyonu - Development ve testing için
    """
    return RAGConfig(
        # Model paths
        base_model_name="gpt2",  # En küçük test modeli
        finetuned_model_path="./saved_models/debug_model",
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        
        # Vector store
        vector_store_type="faiss",
        vector_store_path="./test_data/test_faiss.index",
        metadata_path="./test_data/test_metadata.json",
        
        # Retrieval params
        top_k_retrieval=2,
        similarity_threshold=0.5,
        max_context_length=300,
        
        # Generation params
        max_new_tokens=50,
        temperature=1.0,
        top_p=1.0,
        
        # System
        device="cpu",
        torch_dtype="float32",
        use_4bit_quantization=False
    )


def cpu_mode_rag_config() -> Union["RAGConfig", None]:
    """
    CPU-only RAG konfigürasyonu - Docker ve CUDA olmayan sistemler için
    """
    return RAGConfig(
        # Model paths - Daha küçük model CPU için
        base_model_name="microsoft/DialoGPT-medium",
        finetuned_model_path="./saved_models/aura_fashion_assistant",
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        
        # Vector store
        vector_store_type="faiss",
        vector_store_path="./vector_stores/wardrobe_faiss.index",
        metadata_path="./vector_stores/wardrobe_metadata.json",
        
        # Retrieval params
        top_k_retrieval=3,
        similarity_threshold=0.65,
        max_context_length=800,
        
        # Generation params
        max_new_tokens=150,
        temperature=0.8,
        top_p=0.9,
        
        # System - Force CPU mode
        device="cpu",
        torch_dtype="float32",
        use_4bit_quantization=False  # CPU'da quantization yok
    )


# Config seçici fonksiyon
def get_rag_config(config_name: str) -> Union["RAGConfig", None]:
    """
    Config adına göre RAG konfigürasyonu döndürür
    
    Args:
        config_name: Konfigürasyon adı
        
    Returns:
        RAGConfig: İlgili konfigürasyon
        
    Raises:
        ValueError: Geçersiz config adı
    """
    configs = {
        "basic": basic_rag_config,
        "production": production_rag_config,
        "memory_optimized": memory_optimized_rag_config,
        "fast_inference": fast_inference_rag_config,
        "multilingual": multilingual_rag_config,
        "debug": debug_rag_config,
        "cpu_mode": cpu_mode_rag_config
    }
    
    if config_name not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Geçersiz config adı: {config_name}. Mevcut: {available}")
    
    return configs[config_name]()


# JSON config örnekleri
def save_example_configs(output_dir: str = "./configs"):
    """
    Örnek konfigürasyonları JSON dosyaları olarak kaydeder
    
    Args:
        output_dir: Kayıt dizini
    """
    import os
    import json
    from dataclasses import asdict
    
    os.makedirs(output_dir, exist_ok=True)
    
    configs = {
        "basic_rag_config.json": basic_rag_config(),
        "production_rag_config.json": production_rag_config(),
        "memory_optimized_rag_config.json": memory_optimized_rag_config(),
        "fast_inference_rag_config.json": fast_inference_rag_config(),
        "multilingual_rag_config.json": multilingual_rag_config(),
        "debug_rag_config.json": debug_rag_config()
    }
    
    for filename, config in configs.items():
        filepath = os.path.join(output_dir, filename)
        
        # Convert to dict and handle non-serializable types
        config_dict = asdict(config)
        
        # Convert torch.dtype to string
        if 'torch_dtype' in config_dict:
            config_dict['torch_dtype'] = str(config_dict['torch_dtype']).split('.')[-1]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Example configs saved to {output_dir}")


if __name__ == "__main__":
    """Config örneklerini kaydet"""
    save_example_configs()
    
    # Test all configs
    config_names = ["basic", "production", "memory_optimized", "fast_inference", "multilingual", "debug"]
    
    print("\n📋 Available RAG Configurations:")
    for name in config_names:
        try:
            config = get_rag_config(name)
            print(f"✅ {name}: {config.base_model_name} + {config.vector_store_type}")
        except Exception as e:
            print(f"❌ {name}: Error - {e}")
