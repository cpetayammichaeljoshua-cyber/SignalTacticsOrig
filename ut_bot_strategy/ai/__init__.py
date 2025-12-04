"""
AI Insights Module for UT Bot + STC Trading Strategy

Provides intelligent signal analysis using OpenAI GPT-5 for:
- Signal confidence scoring
- Market sentiment analysis
- AI-adjusted leverage recommendations
"""

from .feature_extractor import FeatureExtractor
from .ai_insights_service import AIInsightsService, AIInsights

__all__ = ['FeatureExtractor', 'AIInsightsService', 'AIInsights']
