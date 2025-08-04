#!/usr/bin/env python3
"""
Test Script for Vector Store Builder
====================================

Bu script, vector store builder modülünün doğru çalışıp çalışmadığını test eder.

Usage:
    python test_vector_store.py [--vector_store_type TYPE] [--quick] [--verbose]
"""

import os
import sys
import subprocess
import argparse
import tempfile
import json
from pathlib import Path
from loguru import logger

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

def test_imports():
    """Test all required imports"""
    logger.info("🧪 Testing imports...")
    
    try:
        import sentence_transformers
        import faiss
        import numpy as np
        logger.success("✅ Core packages imported successfully")
        
        # Version check
        logger.info(f"📦 Package versions:")
        logger.info(f"  - sentence-transformers: {sentence_transformers.__version__}")
        logger.info(f"  - faiss: {faiss.__version__}")
        logger.info(f"  - numpy: {np.__version__}")
        
        # Test Pinecone import (optional)
        try:
            import pinecone
            logger.info(f"  - pinecone: Available")
        except ImportError:
            logger.warning("⚠️ Pinecone not available (optional)")
        
        return True
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False


def test_embedding_model():
    """Test embedding model loading"""
    logger.info("🧪 Testing embedding model...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Test small model loading
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        logger.info(f"Loading model: {model_name}")
        
        model = SentenceTransformer(model_name)
        
        # Test encoding
        test_texts = [
            "This is a blue cotton t-shirt",
            "Elegant black dress for formal occasions",
            "Comfortable running shoes"
        ]
        
        embeddings = model.encode(test_texts)
        logger.success(f"✅ Embeddings generated: {embeddings.shape}")
        
        # Test dimension
        expected_dim = 384  # for all-MiniLM-L6-v2
        actual_dim = embeddings.shape[1]
        
        if actual_dim == expected_dim:
            logger.success(f"✅ Embedding dimension correct: {actual_dim}")
        else:
            logger.warning(f"⚠️ Unexpected dimension: {actual_dim} (expected {expected_dim})")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Embedding model test error: {e}")
        return False


def test_faiss():
    """Test FAISS functionality"""
    logger.info("🧪 Testing FAISS...")
    
    try:
        import faiss
        import numpy as np
        
        # Create test index
        dimension = 384
        index = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIDMap(index)
        
        logger.success("✅ FAISS index created")
        
        # Test adding vectors
        test_vectors = np.random.random((5, dimension)).astype(np.float32)
        test_ids = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        
        index.add_with_ids(test_vectors, test_ids)
        logger.success(f"✅ Added {len(test_vectors)} vectors to FAISS")
        
        # Test search
        query_vector = np.random.random((1, dimension)).astype(np.float32)
        distances, indices = index.search(query_vector, k=3)
        
        logger.success(f"✅ FAISS search successful: found {len(indices[0])} results")
        logger.info(f"  - Distances: {distances[0]}")
        logger.info(f"  - Indices: {indices[0]}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ FAISS test error: {e}")
        return False


def test_data_processing():
    """Test wardrobe data processing"""
    logger.info("🧪 Testing data processing...")
    
    try:
        # Import the module
        from build_vector_store import create_sample_wardrobe_data, process_wardrobe_data, VectorStoreConfig
        
        # Create sample data
        sample_data = create_sample_wardrobe_data()
        logger.success(f"✅ Sample data created: {len(sample_data)} items")
        
        # Test processing
        config = VectorStoreConfig()
        processed_data = process_wardrobe_data(sample_data, config)
        
        logger.success(f"✅ Data processed: {len(processed_data)} items")
        
        # Validate processed data structure
        if processed_data:
            item = processed_data[0]
            required_fields = ['id', 'processed_detections', 'processed_categories']
            
            for field in required_fields:
                if field not in item:
                    logger.error(f"❌ Missing field in processed data: {field}")
                    return False
            
            logger.success("✅ Processed data structure validated")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Data processing test error: {e}")
        return False


def test_text_generation():
    """Test description text generation"""
    logger.info("🧪 Testing text generation...")
    
    try:
        from build_vector_store import create_item_description, create_sample_wardrobe_data
        
        # Get sample data
        sample_data = create_sample_wardrobe_data()
        item = sample_data[0]
        
        # Generate description
        description = create_item_description(item)
        
        logger.success("✅ Description generated")
        logger.info(f"  Sample description: {description[:100]}...")
        
        # Validate description
        if len(description) < 10:
            logger.error("❌ Description too short")
            return False
        
        if not description.endswith('.'):
            logger.warning("⚠️ Description doesn't end with period")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Text generation test error: {e}")
        return False


def test_full_pipeline(vector_store_type: str = "faiss"):
    """Test the full vector store building pipeline"""
    logger.info(f"🧪 Testing full pipeline with {vector_store_type}...")
    
    try:
        # Create temporary directories
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test data file
            data_path = temp_path / "test_wardrobe.json"
            output_path = temp_path / "vector_store"
            
            # Generate test data
            from build_vector_store import create_sample_wardrobe_data
            test_data = create_sample_wardrobe_data()
            
            with open(data_path, 'w') as f:
                json.dump(test_data, f, indent=2)
            
            logger.info(f"✅ Test data created: {data_path}")
            
            # Build command
            cmd = [
                sys.executable, "src/build_vector_store.py",
                "--input_data_path", str(data_path),
                "--output_dir", str(output_path),
                "--vector_store_type", vector_store_type,
                "--batch_size", "2",  # Small batch for testing
                "--embedding_model_name", "sentence-transformers/all-MiniLM-L6-v2"
            ]
            
            # Skip Pinecone if no API key
            if vector_store_type == "pinecone":
                if not os.getenv("PINECONE_API_KEY"):
                    logger.warning("⚠️ Skipping Pinecone test - no API key")
                    return True
                cmd.extend([
                    "--pinecone_api_key", os.getenv("PINECONE_API_KEY"),
                    "--pinecone_index_name", "test-aura-wardrobe"
                ])
            
            logger.info("🚀 Running vector store builder...")
            logger.info(f"Command: {' '.join(cmd)}")
            
            # Run the command
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                logger.success("✅ Vector store builder completed successfully")
                
                # Check output files (for FAISS)
                if vector_store_type == "faiss":
                    expected_files = ["faiss_index.index", "metadata.json", "README.md"]
                    
                    for filename in expected_files:
                        file_path = output_path / filename
                        if file_path.exists():
                            logger.success(f"✅ Output file created: {filename}")
                        else:
                            logger.error(f"❌ Missing output file: {filename}")
                            return False
                
                return True
            else:
                logger.error(f"❌ Vector store builder failed with code: {result.returncode}")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
                return False
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Vector store builder timed out")
        return False
    except Exception as e:
        logger.error(f"❌ Full pipeline test error: {e}")
        return False


def run_tests(vector_store_type: str = "faiss", quick: bool = False, verbose: bool = False):
    """Run all tests"""
    logger.info("🚀 Starting Vector Store Builder Tests")
    logger.info("=" * 50)
    
    if verbose:
        logger.info("Running in verbose mode")
    
    results = {}
    
    # Basic tests
    results["imports"] = test_imports()
    results["embedding_model"] = test_embedding_model()
    results["faiss"] = test_faiss()
    results["data_processing"] = test_data_processing()
    results["text_generation"] = test_text_generation()
    
    if not quick:
        # Full pipeline test
        results[f"full_pipeline_{vector_store_type}"] = test_full_pipeline(vector_store_type)
    else:
        logger.info("⚡ Quick mode - skipping full pipeline test")
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 TEST RESULTS:")
    logger.info("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name:.<30} {status}")
        if result:
            passed += 1
    
    logger.info("=" * 50)
    logger.info(f"📈 SUMMARY: {passed}/{total} tests passed")
    
    if passed == total:
        logger.success("🎉 All tests passed! Vector store builder is ready.")
        return True
    else:
        logger.error("❌ Some tests failed. Please check the requirements.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test Vector Store Builder")
    parser.add_argument(
        "--vector_store_type", 
        choices=["faiss", "pinecone"], 
        default="faiss",
        help="Vector store type to test"
    )
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    success = run_tests(
        vector_store_type=args.vector_store_type,
        quick=args.quick, 
        verbose=args.verbose
    )
    
    if success:
        logger.info("\n🚀 Ready to build vector stores!")
        logger.info("Next steps:")
        logger.info("1. Prepare wardrobe data in JSON format")
        logger.info("2. Run: python src/build_vector_store.py --help")
        logger.info("3. For Pinecone: Set PINECONE_API_KEY environment variable")
    else:
        logger.error("\n❌ Environment not ready. Please fix the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
