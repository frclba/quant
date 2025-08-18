"""
Layer 5: Execution & Order Management

Professional order execution with risk controls and position management.
Provides intelligent order routing, slippage control, and trade reconciliation.

Main Components:
- OrderManagementSystem: Intelligent order execution
- RiskControlSystem: Real-time risk monitoring
- PositionManager: Portfolio position tracking
- Layer5ExecutionProcessor: Main orchestrator
"""

from .order_management import OrderManagementSystem, OMSConfig, OrderResult
from .risk_controls import RiskControlSystem, RiskControlConfig, RiskCheck
from .position_manager import PositionManager, PositionConfig, Position
from .layer5_processor import Layer5ExecutionProcessor, Layer5Config, ExecutionResult

__all__ = [
    # Main processor
    'Layer5ExecutionProcessor',
    'Layer5Config', 
    'ExecutionResult',
    
    # Core systems
    'OrderManagementSystem',
    'RiskControlSystem',
    'PositionManager',
    
    # Configuration classes
    'OMSConfig',
    'RiskControlConfig',
    'PositionConfig',
    
    # Data structures
    'OrderResult',
    'RiskCheck',
    'Position'
]
