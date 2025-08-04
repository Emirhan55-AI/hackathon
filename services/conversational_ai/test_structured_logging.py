"""
Test script to verify structured logging implementation
"""

import sys
import os
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir.parent / "src"
sys.path.insert(0, str(src_dir))

# Test imports
try:
    from src.logging_config import (
        setup_logging,
        get_structured_logger,
        log_service_event,
        log_performance_metric,
        log_ml_inference,
        log_error_with_context
    )
    
    print("✅ All logging functions imported successfully")
    
    # Setup logging
    setup_logging()
    logger = get_structured_logger("test", service="conversational_ai_test")
    
    # Test basic logging
    logger.info("test_message", test_key="test_value")
    
    # Test different log types one by one
    print("Testing service event...")
    log_service_event(logger, "test_started", status="success")
    
    print("Testing performance metric...")
    log_performance_metric(logger, "test_metric", 42.5, "ms")
    
    print("Testing ML inference...")
    log_ml_inference(logger, "test_model", 100, 50, 123.4)
    
    print("Testing error logging...")
    # Test error logging
    try:
        raise ValueError("Test error for logging")
    except Exception as e:
        log_error_with_context(logger, e, {"test_context": "test_value"})
    
    logger.info("structured_logging_test_completed", success=True)
    print("✅ Structured logging test completed successfully")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Test error: {e}")
    sys.exit(1)
