"""
Inference engine for Visual Analysis Service
"""

import asyncio
import time
from typing import Dict, Any, List
from PIL import Image
import io
import sys
import os

# Add shared modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from shared.utils.image_processing import load_image_from_bytes, resize_image, get_dominant_colors
from shared.config.logging import get_logger
from app.core.model import DETRFashionModel, FashionAttributeClassifier, SegmentationModel

logger = get_logger(__name__)

class VisualAnalysisEngine:
    """Main inference engine for visual analysis"""
    
    def __init__(self):
        """Initialize the analysis engine"""
        self.detr_model = None
        self.attribute_classifier = None
        self.segmentation_model = None
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize models asynchronously"""
        if self.is_initialized:
            return
        
        logger.info("Initializing Visual Analysis Engine...")
        
        try:
            # Initialize models in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            
            # Load DETR model
            self.detr_model = await loop.run_in_executor(
                None, 
                lambda: DETRFashionModel()
            )
            
            # Load attribute classifier
            self.attribute_classifier = FashionAttributeClassifier()
            
            # Load segmentation model
            self.segmentation_model = SegmentationModel()
            
            self.is_initialized = True
            logger.info("Visual Analysis Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Visual Analysis Engine: {str(e)}")
            raise
    
    async def analyze_image(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Analyze fashion image and extract structured data
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Dictionary with analysis results
        """
        if not self.is_initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Load and preprocess image
            image = load_image_from_bytes(image_bytes)
            
            # Resize image for consistent processing
            image = resize_image(image, (800, 600), keep_aspect_ratio=True)
            
            # Run detection
            detection_results = await self._run_detection(image)
            
            # Extract attributes for each detected item
            enriched_items = await self._enrich_detections(image, detection_results["items"])
            
            # Calculate processing metrics
            processing_time = time.time() - start_time
            
            return {
                "items": enriched_items,
                "image_metadata": {
                    "width": image.width,
                    "height": image.height,
                    "dominant_colors": get_dominant_colors(image, num_colors=3)
                },
                "processing_metadata": {
                    "processing_time": processing_time,
                    "model_version": "detr-fashionpedia-v1",
                    "num_items_detected": len(enriched_items)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in image analysis: {str(e)}")
            raise
    
    async def _run_detection(self, image: Image.Image) -> Dict[str, Any]:
        """Run object detection on image"""
        loop = asyncio.get_event_loop()
        
        # Run detection in thread pool
        results = await loop.run_in_executor(
            None,
            lambda: self.detr_model.predict(image, confidence_threshold=0.3)
        )
        
        return results
    
    async def _enrich_detections(self, image: Image.Image, items: List[Dict]) -> List[Dict]:
        """Enrich detected items with additional attributes"""
        enriched_items = []
        
        for item in items:
            try:
                # Extract crop for attribute classification
                bbox = item["bounding_box"]
                crop = self._extract_crop(image, bbox)
                
                # Predict attributes
                color = await self._predict_color(crop)
                style = await self._predict_style(crop)
                pattern = await self._predict_pattern(crop)
                
                # Generate segmentation mask
                segmentation_mask = await self._generate_mask(image, bbox)
                
                # Create enriched item
                enriched_item = {
                    **item,
                    "color": color,
                    "style": style,
                    "pattern": pattern,
                    "segmentation_mask": segmentation_mask
                }
                
                enriched_items.append(enriched_item)
                
            except Exception as e:
                logger.warning(f"Failed to enrich detection: {str(e)}")
                # Add item without enrichment
                enriched_items.append(item)
        
        return enriched_items
    
    def _extract_crop(self, image: Image.Image, bbox: Dict[str, float]) -> Image.Image:
        """Extract crop from image using bounding box"""
        x, y, width, height = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
        
        # Convert normalized coordinates to pixel coordinates
        img_width, img_height = image.size
        left = int(x * img_width)
        top = int(y * img_height)
        right = int((x + width) * img_width)
        bottom = int((y + height) * img_height)
        
        return image.crop((left, top, right, bottom))
    
    async def _predict_color(self, crop: Image.Image) -> str:
        """Predict color of clothing item"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.attribute_classifier.predict_color(crop)
        )
    
    async def _predict_style(self, crop: Image.Image) -> str:
        """Predict style of clothing item"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.attribute_classifier.predict_style(crop)
        )
    
    async def _predict_pattern(self, crop: Image.Image) -> str:
        """Predict pattern of clothing item"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.attribute_classifier.predict_pattern(crop)
        )
    
    async def _generate_mask(self, image: Image.Image, bbox: Dict[str, float]) -> Dict[str, Any]:
        """Generate segmentation mask"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.segmentation_model.generate_mask(image, bbox)
        )

# Global inference engine instance
inference_engine = VisualAnalysisEngine()
