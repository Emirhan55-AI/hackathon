"""
Outfit Recommendation API Endpoints
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Dict, Any, Optional
import sys
import os

# Add shared modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from shared.models.base import ClothingItem, OutfitRecommendation, RecommendationContext
from shared.config.logging import get_logger
from app.core.inference import recommendation_engine

logger = get_logger(__name__)
router = APIRouter()

@router.post("/generate", response_model=Dict[str, Any])
async def generate_recommendations(
    user_id: str,
    context: RecommendationContext,
    max_recommendations: int = 5,
    include_compatibility_scores: bool = True
):
    """
    Generate outfit recommendations for a user
    
    Args:
        user_id: User identifier
        context: Recommendation context (weather, occasion, etc.)
        max_recommendations: Maximum number of recommendations to return
        include_compatibility_scores: Whether to include compatibility scores
    
    Returns:
        Generated outfit recommendations
    """
    try:
        logger.info(f"Generating recommendations for user {user_id}")
        
        # Generate recommendations using the engine
        recommendations = await recommendation_engine.generate_recommendations(
            user_id=user_id,
            context=context,
            max_recommendations=max_recommendations
        )
        
        # Add compatibility scores if requested
        if include_compatibility_scores:
            recommendations = await recommendation_engine.add_compatibility_scores(
                recommendations
            )
        
        return {
            "user_id": user_id,
            "context": context.dict(),
            "recommendations": recommendations,
            "generated_at": "2024-01-01T00:00:00Z"  # TODO: Use actual timestamp
        }
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/compatibility", response_model=Dict[str, Any])
async def score_compatibility(
    items: List[ClothingItem],
    context: Optional[RecommendationContext] = None
):
    """
    Score compatibility between clothing items
    
    Args:
        items: List of clothing items to score
        context: Optional context for scoring
    
    Returns:
        Compatibility scores
    """
    try:
        logger.info(f"Scoring compatibility for {len(items)} items")
        
        if len(items) < 2:
            raise HTTPException(
                status_code=400, 
                detail="At least 2 items required for compatibility scoring"
            )
        
        # Calculate compatibility scores
        scores = await recommendation_engine.calculate_compatibility(
            items=items,
            context=context
        )
        
        return {
            "items": [item.dict() for item in items],
            "compatibility_scores": scores,
            "overall_compatibility": scores.get("overall", 0.0)
        }
        
    except Exception as e:
        logger.error(f"Error scoring compatibility: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/user/{user_id}", response_model=Dict[str, Any])
async def get_user_recommendations(
    user_id: str,
    limit: int = 10,
    include_history: bool = False
):
    """
    Get recommendations for a specific user
    
    Args:
        user_id: User identifier
        limit: Maximum number of recommendations
        include_history: Whether to include recommendation history
    
    Returns:
        User recommendations
    """
    try:
        logger.info(f"Getting recommendations for user {user_id}")
        
        # Get user's wardrobe and preferences
        user_data = await recommendation_engine.get_user_data(user_id)
        
        if not user_data:
            raise HTTPException(
                status_code=404, 
                detail=f"User {user_id} not found"
            )
        
        # Generate personalized recommendations
        recommendations = await recommendation_engine.get_personalized_recommendations(
            user_id=user_id,
            limit=limit
        )
        
        result = {
            "user_id": user_id,
            "recommendations": recommendations,
            "total_items_in_wardrobe": len(user_data.get("wardrobe", [])),
            "style_preferences": user_data.get("style_preferences", {})
        }
        
        if include_history:
            history = await recommendation_engine.get_recommendation_history(user_id)
            result["history"] = history
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/feedback", response_model=Dict[str, str])
async def submit_feedback(
    user_id: str,
    recommendation_id: str,
    feedback: Dict[str, Any],
    background_tasks: BackgroundTasks
):
    """
    Submit feedback on a recommendation
    
    Args:
        user_id: User identifier
        recommendation_id: Recommendation identifier
        feedback: Feedback data
        background_tasks: Background task runner
    
    Returns:
        Feedback submission status
    """
    try:
        logger.info(f"Receiving feedback for recommendation {recommendation_id}")
        
        # Validate feedback format
        required_fields = ["rating", "worn"]
        for field in required_fields:
            if field not in feedback:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required field: {field}"
                )
        
        # Process feedback in background
        background_tasks.add_task(
            recommendation_engine.process_feedback,
            user_id=user_id,
            recommendation_id=recommendation_id,
            feedback=feedback
        )
        
        return {
            "status": "feedback_received",
            "message": "Feedback submitted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/trends", response_model=Dict[str, Any])
async def get_style_trends(
    season: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = 20
):
    """
    Get current style trends
    
    Args:
        season: Optional season filter
        category: Optional category filter
        limit: Maximum number of trends
    
    Returns:
        Current style trends
    """
    try:
        logger.info("Getting style trends")
        
        trends = await recommendation_engine.get_style_trends(
            season=season,
            category=category,
            limit=limit
        )
        
        return {
            "trends": trends,
            "filters": {
                "season": season,
                "category": category
            },
            "updated_at": "2024-01-01T00:00:00Z"  # TODO: Use actual timestamp
        }
        
    except Exception as e:
        logger.error(f"Error getting style trends: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
