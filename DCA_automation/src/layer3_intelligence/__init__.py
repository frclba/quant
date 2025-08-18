"""
Layer 3: Intelligence & Scoring

Multi-factor asset scoring and market intelligence for the DCA system.
Provides asset classification, regime detection, and predictive signals.

Main Components:
- MultiFactorScoringEngine: Asset scoring and ranking
- MarketRegimeDetector: Bull/bear/sideways classification
- AssetClassificationSystem: Tier-based asset categorization
- PredictiveSignalEngine: Forward-looking indicators

Integrates with Layer 2 (data processing) and feeds Layer 4 (optimization).
"""

from .scoring_engine import MultiFactorScoringEngine, ScoringConfig, AssetScore
from .regime_detector import MarketRegimeDetector, RegimeConfig, MarketRegime
from .asset_classifier import AssetClassificationSystem, ClassificationConfig, AssetTier
from .signal_engine import PredictiveSignalEngine, SignalConfig, PredictiveSignal
from .layer3_processor import Layer3IntelligenceProcessor, Layer3Config, IntelligenceResult

__all__ = [
    # Main processor
    'Layer3IntelligenceProcessor',
    'Layer3Config',
    'IntelligenceResult',
    
    # Core engines
    'MultiFactorScoringEngine',
    'MarketRegimeDetector', 
    'AssetClassificationSystem',
    'PredictiveSignalEngine',
    
    # Configuration classes
    'ScoringConfig',
    'RegimeConfig',
    'ClassificationConfig',
    'SignalConfig',
    
    # Data structures
    'AssetScore',
    'MarketRegime',
    'AssetTier', 
    'PredictiveSignal'
]
