"""
Visualization Module

Comprehensive visualization system for DCA portfolio analysis.
Provides charts, graphs, and dashboards to visualize:
- Portfolio correlation matrices
- Returns and volatility analysis
- Asset scoring and classification
- Markowitz efficient frontier
- Portfolio performance metrics
"""

from .portfolio_visualizer import (
    PortfolioVisualizer,
    VisualizationConfig
)

__all__ = [
    'PortfolioVisualizer',
    'VisualizationConfig'
]
