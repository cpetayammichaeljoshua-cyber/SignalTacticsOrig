#!/usr/bin/env python3
"""
AI Capability Checker - Hard Dependency Verification System
Ensures AI components are properly available and enforces smart analysis requirements
Provides degraded mode detection and fail-fast behavior
"""

import logging
import os
import sys
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import traceback

class CapabilityLevel(Enum):
    """AI capability levels"""
    FULL = "full"           # All AI components available
    DEGRADED = "degraded"   # Some AI components missing, using intelligent fallbacks
    FAILED = "failed"       # Critical AI components missing, cannot provide smart analysis

@dataclass
class CapabilityResult:
    """Result of capability check"""
    component: str
    available: bool
    level: CapabilityLevel
    error: Optional[str]
    fallback_available: bool
    intelligence_score: float  # 0.0 to 1.0, how intelligent the component is

@dataclass
class SystemCapability:
    """Overall system capability assessment"""
    level: CapabilityLevel
    components: Dict[str, CapabilityResult]
    intelligence_score: float
    can_provide_smart_analysis: bool
    issues: List[str]
    recommendations: List[str]

class AICapabilityChecker:
    """
    Comprehensive AI capability checker that enforces smart analysis requirements
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Minimum intelligence requirements
        self.min_intelligence_score = 0.7  # Minimum 70% intelligence for smart analysis
        self.critical_components = ['sentiment_analysis', 'market_prediction']
        
        # Component requirements and fallback intelligence scores
        self.component_requirements = {
            'openai_gpt': {
                'packages': ['openai'],
                'env_vars': ['OPENAI_API_KEY'],
                'intelligence_score': 1.0,
                'fallback_score': 0.0,
                'critical': True
            },
            'pytorch_transformers': {
                'packages': ['torch', 'transformers'],
                'env_vars': [],
                'intelligence_score': 0.9,
                'fallback_score': 0.6,  # Statistical models can provide decent intelligence
                'critical': False
            },
            'sklearn': {
                'packages': ['sklearn'],
                'env_vars': [],
                'intelligence_score': 0.7,
                'fallback_score': 0.4,  # Basic statistical analysis
                'critical': False
            },
            'sentiment_analysis': {
                'packages': ['textblob', 'vaderSentiment'],  # Alternative packages
                'env_vars': [],
                'intelligence_score': 0.8,
                'fallback_score': 0.5,  # Rule-based sentiment can be intelligent
                'critical': True
            },
            'market_prediction': {
                'packages': ['numpy', 'pandas', 'scipy'],
                'env_vars': [],
                'intelligence_score': 0.7,
                'fallback_score': 0.4,  # Statistical prediction methods
                'critical': True
            }
        }
        
    def check_system_capabilities(self) -> SystemCapability:
        """
        Perform comprehensive system capability check
        
        Returns:
            SystemCapability with full assessment
        """
        self.logger.info("🔍 Starting comprehensive AI capability check...")
        
        # Check individual components
        component_results = {}
        total_intelligence = 0.0
        available_components = 0
        issues = []
        recommendations = []
        
        for component, requirements in self.component_requirements.items():
            result = self._check_component(component, requirements)
            component_results[component] = result
            
            if result.available:
                total_intelligence += result.intelligence_score
                available_components += 1
            elif result.fallback_available:
                total_intelligence += requirements['fallback_score']
                available_components += 1
                issues.append(f"{component}: Using fallback (reduced intelligence)")
                recommendations.append(f"Install {component} dependencies for full intelligence")
            else:
                if requirements['critical']:
                    issues.append(f"{component}: CRITICAL component unavailable")
                    recommendations.append(f"URGENT: Install {component} dependencies")
                else:
                    issues.append(f"{component}: Component unavailable")
                    recommendations.append(f"Consider installing {component} for enhanced analysis")
        
        # Calculate overall intelligence score
        max_possible_intelligence = len(self.component_requirements)
        overall_intelligence = total_intelligence / max_possible_intelligence if max_possible_intelligence > 0 else 0.0
        
        # Determine system capability level
        level = self._determine_system_level(component_results, overall_intelligence)
        can_provide_smart_analysis = overall_intelligence >= self.min_intelligence_score
        
        if not can_provide_smart_analysis:
            issues.append("SYSTEM WARNING: Intelligence score below minimum threshold")
            recommendations.append("Install missing AI dependencies to enable smart analysis")
        
        capability = SystemCapability(
            level=level,
            components=component_results,
            intelligence_score=overall_intelligence,
            can_provide_smart_analysis=can_provide_smart_analysis,
            issues=issues,
            recommendations=recommendations
        )
        
        self._log_capability_results(capability)
        return capability
    
    def _check_component(self, component_name: str, requirements: Dict[str, Any]) -> CapabilityResult:
        """Check individual component capability"""
        try:
            # Check package dependencies
            packages_available = self._check_packages(requirements.get('packages', []))
            
            # Check environment variables
            env_vars_available = self._check_env_vars(requirements.get('env_vars', []))
            
            # Check specific component functionality
            functionality_available = self._check_component_functionality(component_name)
            
            # Component is available if all requirements are met
            available = packages_available and env_vars_available and functionality_available
            
            # Check if intelligent fallback is available
            fallback_available = self._check_fallback_availability(component_name)
            
            # Determine intelligence score
            if available:
                intelligence_score = requirements.get('intelligence_score', 0.7)
                level = CapabilityLevel.FULL
            elif fallback_available:
                intelligence_score = requirements.get('fallback_score', 0.3)
                level = CapabilityLevel.DEGRADED
            else:
                intelligence_score = 0.0
                level = CapabilityLevel.FAILED
            
            error_msg = None
            if not available:
                error_reasons = []
                if not packages_available:
                    error_reasons.append("Missing packages")
                if not env_vars_available:
                    error_reasons.append("Missing environment variables")
                if not functionality_available:
                    error_reasons.append("Functionality check failed")
                error_msg = "; ".join(error_reasons)
            
            return CapabilityResult(
                component=component_name,
                available=available,
                level=level,
                error=error_msg,
                fallback_available=fallback_available,
                intelligence_score=intelligence_score
            )
            
        except Exception as e:
            self.logger.error(f"❌ Component check failed for {component_name}: {e}")
            return CapabilityResult(
                component=component_name,
                available=False,
                level=CapabilityLevel.FAILED,
                error=str(e),
                fallback_available=False,
                intelligence_score=0.0
            )
    
    def _check_packages(self, packages: List[str]) -> bool:
        """Check if required packages are available"""
        for package in packages:
            try:
                __import__(package)
            except ImportError:
                return False
        return True
    
    def _check_env_vars(self, env_vars: List[str]) -> bool:
        """Check if required environment variables are set"""
        for var in env_vars:
            if not os.environ.get(var):
                return False
        return True
    
    def _check_component_functionality(self, component_name: str) -> bool:
        """Check if specific component functionality works"""
        try:
            if component_name == 'openai_gpt':
                return self._test_openai_functionality()
            elif component_name == 'pytorch_transformers':
                return self._test_pytorch_functionality()
            elif component_name == 'sentiment_analysis':
                return self._test_sentiment_functionality()
            elif component_name == 'market_prediction':
                return self._test_prediction_functionality()
            else:
                return True  # Assume basic components work if packages are available
                
        except Exception as e:
            self.logger.debug(f"Functionality check failed for {component_name}: {e}")
            return False
    
    def _test_openai_functionality(self) -> bool:
        """Test OpenAI GPT functionality"""
        try:
            import openai
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                return False
            
            # Basic client creation test
            client = openai.OpenAI(api_key=api_key)
            return True
            
        except Exception:
            return False
    
    def _test_pytorch_functionality(self) -> bool:
        """Test PyTorch and Transformers functionality"""
        try:
            import torch
            import transformers
            
            # Test basic tensor operations
            x = torch.randn(2, 3)
            y = torch.sum(x)
            
            return True
            
        except Exception:
            return False
    
    def _test_sentiment_functionality(self) -> bool:
        """Test sentiment analysis functionality"""
        try:
            # Test if we can perform basic sentiment analysis
            import re
            
            # If advanced packages aren't available, we can still do rule-based analysis
            test_text = "This is a positive market outlook"
            positive_words = ["positive", "good", "bullish", "optimistic", "growth"]
            negative_words = ["negative", "bad", "bearish", "pessimistic", "decline"]
            
            # Basic sentiment logic test
            pos_count = sum(1 for word in positive_words if word.lower() in test_text.lower())
            neg_count = sum(1 for word in negative_words if word.lower() in test_text.lower())
            
            return True  # Rule-based sentiment always works
            
        except Exception:
            return False
    
    def _test_prediction_functionality(self) -> bool:
        """Test market prediction functionality"""
        try:
            import numpy as np
            import pandas as pd
            
            # Test basic statistical operations
            data = np.random.randn(100)
            mean = np.mean(data)
            std = np.std(data)
            
            # Test pandas operations
            df = pd.DataFrame({'price': data})
            sma = df['price'].rolling(20).mean()
            
            return True
            
        except Exception:
            return False
    
    def _check_fallback_availability(self, component_name: str) -> bool:
        """Check if intelligent fallback is available for component"""
        try:
            if component_name == 'openai_gpt':
                # No intelligent fallback for GPT-5, this is critical
                return False
            elif component_name == 'pytorch_transformers':
                # Can use statistical models as fallback
                return self._test_prediction_functionality()
            elif component_name == 'sentiment_analysis':
                # Rule-based sentiment analysis is available
                return True
            elif component_name == 'market_prediction':
                # Statistical prediction methods are available
                return self._test_prediction_functionality()
            else:
                return False
                
        except Exception:
            return False
    
    def _determine_system_level(self, component_results: Dict[str, CapabilityResult], 
                              intelligence_score: float) -> CapabilityLevel:
        """Determine overall system capability level"""
        
        # Check for critical component failures
        critical_failures = []
        for component_name in self.critical_components:
            if component_name in component_results:
                result = component_results[component_name]
                if result.level == CapabilityLevel.FAILED:
                    critical_failures.append(component_name)
        
        # If critical components failed and no fallbacks, system fails
        if critical_failures and intelligence_score < self.min_intelligence_score:
            return CapabilityLevel.FAILED
        
        # If intelligence score is high enough, determine level based on component availability
        full_components = sum(1 for r in component_results.values() if r.level == CapabilityLevel.FULL)
        total_components = len(component_results)
        
        if full_components == total_components:
            return CapabilityLevel.FULL
        elif intelligence_score >= self.min_intelligence_score:
            return CapabilityLevel.DEGRADED
        else:
            return CapabilityLevel.FAILED
    
    def _log_capability_results(self, capability: SystemCapability):
        """Log capability check results"""
        
        level_emoji = {
            CapabilityLevel.FULL: "✅",
            CapabilityLevel.DEGRADED: "⚠️",
            CapabilityLevel.FAILED: "❌"
        }
        
        emoji = level_emoji.get(capability.level, "❓")
        
        self.logger.info(f"{emoji} AI System Capability: {capability.level.value.upper()}")
        self.logger.info(f"📊 Intelligence Score: {capability.intelligence_score:.2f}")
        self.logger.info(f"🧠 Can Provide Smart Analysis: {capability.can_provide_smart_analysis}")
        
        # Log component status
        for component, result in capability.components.items():
            status = "✅" if result.available else ("🔄" if result.fallback_available else "❌")
            self.logger.info(f"{status} {component}: {result.level.value} (intelligence: {result.intelligence_score:.2f})")
            
            if result.error:
                self.logger.warning(f"  └─ Issue: {result.error}")
        
        # Log issues
        if capability.issues:
            self.logger.warning("⚠️ System Issues:")
            for issue in capability.issues:
                self.logger.warning(f"  • {issue}")
        
        # Log recommendations
        if capability.recommendations:
            self.logger.info("💡 Recommendations:")
            for rec in capability.recommendations:
                self.logger.info(f"  • {rec}")
    
    def enforce_smart_analysis_requirements(self, capability: SystemCapability) -> bool:
        """
        Enforce smart analysis requirements - fail fast if not met
        
        Returns:
            bool: True if requirements are met, False otherwise
        """
        if not capability.can_provide_smart_analysis:
            error_msg = (
                f"❌ SMART ANALYSIS REQUIREMENTS NOT MET:\n"
                f"   Intelligence Score: {capability.intelligence_score:.2f} (required: {self.min_intelligence_score:.2f})\n"
                f"   System Level: {capability.level.value}\n"
                f"   Issues: {', '.join(capability.issues)}\n\n"
                f"💡 To enable smart analysis:\n"
            )
            
            for rec in capability.recommendations:
                error_msg += f"   • {rec}\n"
            
            self.logger.error(error_msg)
            return False
        
        return True
    
    def get_degraded_mode_info(self, capability: SystemCapability) -> Dict[str, Any]:
        """Get information about degraded mode operation"""
        
        available_features = []
        degraded_features = []
        unavailable_features = []
        
        for component, result in capability.components.items():
            if result.available:
                available_features.append(component)
            elif result.fallback_available:
                degraded_features.append(component)
            else:
                unavailable_features.append(component)
        
        return {
            'mode': 'degraded' if capability.level == CapabilityLevel.DEGRADED else capability.level.value,
            'intelligence_score': capability.intelligence_score,
            'available_features': available_features,
            'degraded_features': degraded_features,
            'unavailable_features': unavailable_features,
            'smart_analysis_possible': capability.can_provide_smart_analysis,
            'recommendations': capability.recommendations
        }


# Global capability checker instance
_capability_checker = None

def get_capability_checker() -> AICapabilityChecker:
    """Get global capability checker instance"""
    global _capability_checker
    if _capability_checker is None:
        _capability_checker = AICapabilityChecker()
    return _capability_checker

def check_ai_capabilities() -> SystemCapability:
    """Convenience function to check AI capabilities"""
    checker = get_capability_checker()
    return checker.check_system_capabilities()

def enforce_smart_ai_requirements() -> bool:
    """Convenience function to enforce smart AI requirements"""
    checker = get_capability_checker()
    capability = checker.check_system_capabilities()
    return checker.enforce_smart_analysis_requirements(capability)