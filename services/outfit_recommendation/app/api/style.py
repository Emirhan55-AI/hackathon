"""
Style Analysis API Endpoints
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any, Optional
import sys
import os

# Add shared modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from shared.models.base import ClothingItem, StyleProfile
from shared.config.logging import get_logger
from app.core.inference import recommendation_engine

logger = get_logger(__name__)
router = APIRouter()

@router.post("/analyze", response_model=Dict[str, Any])
async def analyze_style_preferences(
    user_id: str,
    wardrobe_items: List[ClothingItem],
    analyze_patterns: bool = True,
    analyze_colors: bool = True
):
    """
    Analyze user's style preferences from their wardrobe
    
    Args:
        user_id: User identifier
        wardrobe_items: List of clothing items in user's wardrobe
        analyze_patterns: Whether to analyze pattern preferences
        analyze_colors: Whether to analyze color preferences
    
    Returns:
        Style analysis results
    """
    try:
        logger.info(f"Analyzing style preferences for user {user_id}")
        
        if not wardrobe_items:
            raise HTTPException(
                status_code=400,
                detail="At least one wardrobe item required for analysis"
            )
        
        # Analyze style preferences
        style_analysis = await recommendation_engine.analyze_user_style(
            user_id=user_id,
            wardrobe_items=wardrobe_items,
            analyze_patterns=analyze_patterns,
            analyze_colors=analyze_colors
        )
        
        return {
            "user_id": user_id,
            "style_analysis": style_analysis,
            "wardrobe_size": len(wardrobe_items),
            "analysis_date": "2024-01-01T00:00:00Z"  # TODO: Use actual timestamp
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing style preferences: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/profile/{user_id}", response_model=Dict[str, str])
async def update_style_profile(
    user_id: str,
    style_profile: StyleProfile
):
    """
    Update user's style profile
    
    Args:
        user_id: User identifier
        style_profile: Updated style profile
    
    Returns:
        Update status
    """
    try:
        logger.info(f"Updating style profile for user {user_id}")
        
        # Update style profile
        await recommendation_engine.update_style_profile(
            user_id=user_id,
            style_profile=style_profile
        )
        
        return {
            "status": "updated",
            "message": f"Style profile updated for user {user_id}"
        }
        
    except Exception as e:
        logger.error(f"Error updating style profile: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/profile/{user_id}", response_model=Dict[str, Any])
async def get_style_profile(user_id: str):
    """
    Get user's style profile
    
    Args:
        user_id: User identifier
    
    Returns:
        User's style profile
    """
    try:
        logger.info(f"Getting style profile for user {user_id}")
        
        profile = await recommendation_engine.get_style_profile(user_id)
        
        if not profile:
            raise HTTPException(
                status_code=404,
                detail=f"Style profile not found for user {user_id}"
            )
        
        return {
            "user_id": user_id,
            "style_profile": profile,
            "last_updated": "2024-01-01T00:00:00Z"  # TODO: Use actual timestamp
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting style profile: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/trends", response_model=Dict[str, Any])
async def get_current_trends(
    category: Optional[str] = None,
    season: Optional[str] = None,
    limit: int = 10
):
    """
    Get current fashion trends
    
    Args:
        category: Optional category filter
        season: Optional season filter
        limit: Maximum number of trends
    
    Returns:
        Current fashion trends
    """
    try:
        logger.info("Getting current fashion trends")
        
        trends = await recommendation_engine.get_fashion_trends(
            category=category,
            season=season,
            limit=limit
        )
        
        return {
            "trends": trends,
            "filters": {
                "category": category,
                "season": season
            },
            "total_trends": len(trends)
        }
        
    except Exception as e:
        logger.error(f"Error getting fashion trends: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/similarity", response_model=Dict[str, Any])
async def find_similar_styles(
    reference_items: List[ClothingItem],
    user_wardrobe: List[ClothingItem],
    similarity_threshold: float = 0.7
):
    """
    Find similar styles in user's wardrobe
    
    Args:
        reference_items: Reference clothing items
        user_wardrobe: User's wardrobe items
        similarity_threshold: Minimum similarity score
    
    Returns:
        Similar style matches
    """
    try:
        logger.info("Finding similar styles")
        
        if not reference_items or not user_wardrobe:
            raise HTTPException(
                status_code=400,
                detail="Both reference items and wardrobe items are required"
            )
        
        similar_items = await recommendation_engine.find_similar_styles(
            reference_items=reference_items,
            user_wardrobe=user_wardrobe,
            similarity_threshold=similarity_threshold
        )
        
        return {
            "reference_items": [item.dict() for item in reference_items],
            "similar_items": similar_items,
            "similarity_threshold": similarity_threshold,
            "total_matches": len(similar_items)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finding similar styles: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
