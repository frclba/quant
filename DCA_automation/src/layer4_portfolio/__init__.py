"""
Layer 4: Portfolio Optimization

Modern Portfolio Theory implementation for optimal asset allocation.
Provides Markowitz optimization, risk-adjusted allocation, and rebalancing.

Main Components:
- MarkowitzOptimizer: Classic MPT optimization
- RiskAdjustedAllocator: Dynamic position sizing
- RebalancingEngine: Portfolio maintenance
- Layer4PortfolioProcessor: Main orchestrator
"""

from .markowitz_optimizer import MarkowitzPortfolioOptimizer, MarkowitzConfig, OptimizationResult as MarkowitzResult
from .risk_allocator import RiskAdjustedAllocator, RiskConfig, AllocationResult
from .rebalancing_engine import RebalancingEngine, RebalanceConfig, RebalanceAction
from .layer4_processor import Layer4PortfolioProcessor, Layer4Config, OptimizationResult

__all__ = [
    # Main processor
    'Layer4PortfolioProcessor',
    'Layer4Config',
    'OptimizationResult',
    
    # Core engines
    'MarkowitzPortfolioOptimizer',
    'RiskAdjustedAllocator',
    'RebalancingEngine',
    
    # Configuration classes
    'MarkowitzConfig',
    'RiskConfig', 
    'RebalanceConfig',
    
    # Data structures
    'MarkowitzResult',
    'AllocationResult',
    'RebalanceAction'
]
