"""
Aura Conversational AI - Fine-Tuning, Vector Store and RAG Service Module
=========================================================================

Bu paket, Aura moda asistanı için QLoRA tabanlı fine-tuning, 
RAG sistemi vector store işlemleri ve hibrit RAG servisi içerir.

Modules:
    finetune: QLoRA fine-tuning script
    build_vector_store: RAG vector store builder
    rag_service: Hibrit QLoRA + RAG pipeline servisi
    config_examples: Örnek konfigürasyonlar
    
Usage:
    python -m src.finetune --help
    python -m src.build_vector_store --help
    python -m src.rag_service
    
Author: Aura AI Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Aura AI Team"

# Import main modules with error handling
try:
    from src.finetune import main as run_finetune
    FINETUNE_AVAILABLE = True
except ImportError as e:
    try:
        from .finetune import main as run_finetune
        FINETUNE_AVAILABLE = True
    except ImportError:
        print(f"Warning: Fine-tuning import failed: {e}")
        FINETUNE_AVAILABLE = False
        run_finetune = None

try:
    from src.build_vector_store import main as run_vector_store_builder
    VECTOR_STORE_AVAILABLE = True
except ImportError as e:
    try:
        from .build_vector_store import main as run_vector_store_builder
        VECTOR_STORE_AVAILABLE = True
    except ImportError:
        print(f"Warning: Vector store import failed: {e}")
        VECTOR_STORE_AVAILABLE = False
        run_vector_store_builder = None

try:
    from src.config_examples import basic_config, memory_optimized_config, fast_config, production_config
    CONFIG_AVAILABLE = True
except ImportError as e:
    try:
        from .config_examples import basic_config, memory_optimized_config, fast_config, production_config
        CONFIG_AVAILABLE = True
    except ImportError:
        print(f"Warning: Config examples import failed: {e}")
        CONFIG_AVAILABLE = False

# Import RAG service
try:
    from src.rag_service import RAGService, RAGConfig, create_rag_service, test_rag_service
    RAG_AVAILABLE = True
except ImportError as e:
    try:
        from .rag_service import RAGService, RAGConfig, create_rag_service, test_rag_service
        RAG_AVAILABLE = True
    except ImportError:
        RAG_AVAILABLE = False
        print(f"Warning: RAG Service import failed: {e}")

__all__ = []

# Add available exports
if FINETUNE_AVAILABLE and run_finetune:
    __all__.append("run_finetune")

if VECTOR_STORE_AVAILABLE and run_vector_store_builder:
    __all__.append("run_vector_store_builder")

if CONFIG_AVAILABLE:
    __all__.extend([
        "basic_config", 
        "memory_optimized_config",
        "fast_config",
        "production_config"
    ])

# Add RAG exports if available
if RAG_AVAILABLE:
    __all__.extend([
        "RAGService",
        "RAGConfig", 
        "create_rag_service",
        "test_rag_service"
    ])
