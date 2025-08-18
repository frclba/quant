"""
Layer 1: Data Ingestion & Storage

Core data collection functionality for the DCA system.
Provides market data collection with incremental updates.

Main Components:
- Layer1DataProcessor: Main orchestrator for Layer 1 operations
- ProductionDataCollector: Production-ready data collector
- MercadoBitcoinConnector: Market data API connector  
- DataStorageEngine: Data storage management
- DataQualityManager: Data quality validation
"""

# Main processor (recommended)
from .layer1_processor import Layer1DataProcessor, Layer1Config, IngestionResult

# Production data collector (integrated with system)
from .data_collector import ProductionDataCollector, CollectionMode, CollectionStats

# Core components for advanced usage
from .market_data_collector import MercadoBitcoinConnector, MarketDataPoint
from .data_storage import DataStorageEngine, StorageConfig  
from .data_quality import DataQualityManager, DataQualityReport
from .config import get_config, ProductionConfig

__all__ = [
    # Main processor
    'Layer1DataProcessor',
    'Layer1Config',
    'IngestionResult',
    
    # Production interface
    'ProductionDataCollector',
    'CollectionMode',
    'CollectionStats',
    
    # Core components  
    'MercadoBitcoinConnector',
    'DataStorageEngine', 
    'DataQualityManager',
    
    # Configuration
    'get_config',
    'ProductionConfig',
    
    # Data structures
    'MarketDataPoint',
    'StorageConfig',
    'DataQualityReport'
]
