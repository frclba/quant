#!/usr/bin/env python3
"""
Layer 2: Data Processing & Analytics

This layer transforms raw market data from Layer 1 into processed features
and analytics required for Layer 3 intelligence systems.

Modules:
- technical_indicators: Technical analysis calculations (RSI, MACD, Bollinger Bands)
- statistical_analysis: Advanced statistical calculations (returns, correlations, risk metrics)
- data_pipeline: Data transformation, normalization, and feature engineering

Based on PFS Theory Notebooks:
- Notebook 1: Price modeling, returns, correlations
- Notebook 3: Technical indicators and frequency response  
- Notebook 5: Long/short operations and cointegration
"""

from .technical_indicators import TechnicalIndicatorsEngine, IndicatorConfig
from .statistical_analysis import StatisticalAnalysisEngine, StatisticalConfig
from .data_pipeline import DataTransformationPipeline, PipelineConfig
from .layer2_processor import Layer2DataProcessor, Layer2Config, process_layer1_data

__all__ = [
    'TechnicalIndicatorsEngine',
    'IndicatorConfig', 
    'StatisticalAnalysisEngine',
    'StatisticalConfig',
    'DataTransformationPipeline', 
    'PipelineConfig',
    'Layer2DataProcessor',
    'Layer2Config',
    'process_layer1_data'
]

__version__ = '1.0.0'
