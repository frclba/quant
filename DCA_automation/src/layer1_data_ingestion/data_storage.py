"""
Module 1.2: Data Storage Engine

Persistent storage system for the DCA system with efficient retrieval.
Handles time-series data storage using Parquet format with partitioning.

Storage Architecture:
- Raw market data (OHLCV, orderbook, trades)
- Processed features (technical indicators, scores)
- Portfolio data (positions, execution logs)
- Metadata (asset information, system state)
"""

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
import hashlib
import pickle
from abc import ABC, abstractmethod

from .market_data_collector import MarketDataPoint, AssetMetadata

logger = logging.getLogger(__name__)


@dataclass
class StorageConfig:
    """Configuration for data storage system"""
    base_path: str
    raw_data_retention_days: int = 365
    processed_data_retention_days: int = 180
    partition_by_date: bool = True
    enable_compression: bool = True
    backup_enabled: bool = True
    max_file_size_mb: int = 100


class DataStorageEngine:
    """
    Main data storage orchestrator
    Provides unified interface for storing and retrieving market data
    """
    
    def __init__(self, config: StorageConfig):
        self.config = config
        self.base_path = Path(config.base_path)
        
        # Initialize storage structure
        self._create_storage_structure()
        
        # Storage managers for different data types
        self.ohlcv_storage = OHLCVStorage(self.base_path / "market_data" / "ohlcv", config)
        self.metadata_storage = MetadataStorage(self.base_path / "metadata", config)
        self.features_storage = FeaturesStorage(self.base_path / "features", config)
        self.portfolio_storage = PortfolioStorage(self.base_path / "portfolio", config)
        self.system_storage = SystemStateStorage(self.base_path / "system", config)
        
        logger.info(f"Data storage engine initialized at: {self.base_path}")
    
    def _create_storage_structure(self):
        """Create directory structure for data storage"""
        directories = [
            "market_data/ohlcv",
            "market_data/orderbook",
            "market_data/trades",
            "metadata",
            "features/technical_indicators",
            "features/scores",
            "portfolio/positions",
            "portfolio/execution_logs",
            "portfolio/performance",
            "system/config",
            "system/logs",
            "system/state",
            "backups"
        ]
        
        for directory in directories:
            (self.base_path / directory).mkdir(parents=True, exist_ok=True)
    
    async def store_ohlcv_data(self, ohlcv_data: Dict[str, List[MarketDataPoint]], 
                              timeframe: str = "1d") -> bool:
        """Store OHLCV data with efficient partitioning"""
        return await self.ohlcv_storage.store_data(ohlcv_data, timeframe)
    
    async def retrieve_ohlcv_data(self, symbols: List[str], start_date: datetime, 
                                 end_date: datetime, timeframe: str = "1d") -> pd.DataFrame:
        """Retrieve OHLCV data for specified symbols and date range"""
        return await self.ohlcv_storage.retrieve_data(symbols, start_date, end_date, timeframe)
    
    async def store_metadata(self, metadata: Dict[str, AssetMetadata]) -> bool:
        """Store asset metadata"""
        return await self.metadata_storage.store_metadata(metadata)
    
    async def retrieve_metadata(self, symbols: Optional[List[str]] = None) -> Dict[str, AssetMetadata]:
        """Retrieve asset metadata"""
        return await self.metadata_storage.retrieve_metadata(symbols)
    
    async def store_features(self, features: pd.DataFrame, feature_type: str) -> bool:
        """Store calculated features (technical indicators, scores, etc.)"""
        return await self.features_storage.store_features(features, feature_type)
    
    async def retrieve_features(self, symbols: List[str], start_date: datetime, 
                               end_date: datetime, feature_type: str) -> pd.DataFrame:
        """Retrieve stored features"""
        return await self.features_storage.retrieve_features(symbols, start_date, end_date, feature_type)
    
    def get_storage_statistics(self) -> Dict[str, Dict]:
        """Get storage statistics and health information"""
        stats = {}
        
        for storage_name, storage in [
            ("ohlcv", self.ohlcv_storage),
            ("metadata", self.metadata_storage),
            ("features", self.features_storage),
            ("portfolio", self.portfolio_storage),
        ]:
            stats[storage_name] = storage.get_statistics()
        
        return stats


class OHLCVStorage:
    """Specialized storage for OHLCV market data"""
    
    def __init__(self, base_path: Path, config: StorageConfig):
        self.base_path = base_path
        self.config = config
        base_path.mkdir(parents=True, exist_ok=True)
    
    async def store_data(self, ohlcv_data: Dict[str, List[MarketDataPoint]], timeframe: str) -> bool:
        """Store OHLCV data with date partitioning"""
        try:
            for symbol, data_points in ohlcv_data.items():
                if not data_points:
                    continue
                
                # Convert to DataFrame
                df = pd.DataFrame([
                    {
                        'timestamp': point.timestamp,
                        'open': point.open_price,
                        'high': point.high_price,
                        'low': point.low_price,
                        'close': point.close_price,
                        'volume': point.volume,
                        'quote_volume': point.quote_volume,
                        'source': point.source
                    }
                    for point in data_points
                ])
                
                if df.empty:
                    continue
                
                df['date'] = df['timestamp'].dt.date
                df.set_index('timestamp', inplace=True)
                
                # Store with date partitioning
                if self.config.partition_by_date:
                    for date, group in df.groupby('date'):
                        partition_path = self.base_path / timeframe / f"symbol={symbol}" / f"date={date}"
                        partition_path.mkdir(parents=True, exist_ok=True)
                        
                        file_path = partition_path / f"{symbol}_{date}_{timeframe}.parquet"
                        
                        # Use compression if enabled
                        compression = 'snappy' if self.config.enable_compression else None
                        group.drop('date', axis=1).to_parquet(file_path, compression=compression)
                else:
                    # Store as single file per symbol
                    file_path = self.base_path / timeframe / f"{symbol}_{timeframe}.parquet"
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    compression = 'snappy' if self.config.enable_compression else None
                    df.drop('date', axis=1).to_parquet(file_path, compression=compression)
            
            logger.info(f"Stored OHLCV data for {len(ohlcv_data)} symbols, timeframe={timeframe}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing OHLCV data: {e}")
            return False
    
    async def retrieve_data(self, symbols: List[str], start_date: datetime, 
                           end_date: datetime, timeframe: str) -> pd.DataFrame:
        """Retrieve OHLCV data for specified parameters"""
        try:
            dfs = []
            
            for symbol in symbols:
                symbol_dfs = []
                
                if self.config.partition_by_date:
                    # Read from date partitions
                    current_date = start_date.date()
                    end_date_only = end_date.date()
                    
                    while current_date <= end_date_only:
                        partition_path = self.base_path / timeframe / f"symbol={symbol}" / f"date={current_date}"
                        file_path = partition_path / f"{symbol}_{current_date}_{timeframe}.parquet"
                        
                        if file_path.exists():
                            df_partition = pd.read_parquet(file_path)
                            symbol_dfs.append(df_partition)
                        
                        current_date += timedelta(days=1)
                else:
                    # Read from single file
                    file_path = self.base_path / timeframe / f"{symbol}_{timeframe}.parquet"
                    if file_path.exists():
                        df = pd.read_parquet(file_path)
                        # Filter by date range
                        mask = (df.index >= start_date) & (df.index <= end_date)
                        symbol_dfs.append(df[mask])
                
                # Combine symbol data
                if symbol_dfs:
                    symbol_df = pd.concat(symbol_dfs, axis=0).sort_index()
                    # Add symbol to column names
                    symbol_df.columns = [(symbol, col) for col in symbol_df.columns]
                    dfs.append(symbol_df)
            
            # Combine all symbols
            if dfs:
                result_df = pd.concat(dfs, axis=1, sort=True)
                result_df = result_df.loc[start_date:end_date]
                logger.info(f"Retrieved OHLCV data: {len(symbols)} symbols, {len(result_df)} data points")
                return result_df
            else:
                logger.warning(f"No OHLCV data found for symbols: {symbols}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error retrieving OHLCV data: {e}")
            return pd.DataFrame()
    
    def get_statistics(self) -> Dict:
        """Get OHLCV storage statistics"""
        stats = {
            'total_files': 0,
            'total_size_mb': 0,
            'symbols_count': 0,
            'date_range': None
        }
        
        try:
            # Count files and calculate size
            for file_path in self.base_path.rglob("*.parquet"):
                stats['total_files'] += 1
                stats['total_size_mb'] += file_path.stat().st_size / (1024 * 1024)
            
            # Count unique symbols
            symbols = set()
            for symbol_dir in self.base_path.rglob("symbol=*"):
                symbol = symbol_dir.name.replace("symbol=", "")
                symbols.add(symbol)
            stats['symbols_count'] = len(symbols)
            
            logger.debug(f"OHLCV storage stats: {stats}")
            
        except Exception as e:
            logger.error(f"Error getting OHLCV storage statistics: {e}")
        
        return stats


class MetadataStorage:
    """Specialized storage for asset metadata"""
    
    def __init__(self, base_path: Path, config: StorageConfig):
        self.base_path = base_path
        self.config = config
        base_path.mkdir(parents=True, exist_ok=True)
        
        # SQLite database for metadata
        self.db_path = base_path / "metadata.db"
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize SQLite database for metadata"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS asset_metadata (
                    symbol TEXT PRIMARY KEY,
                    name TEXT,
                    market_cap_usd REAL,
                    market_cap_brl REAL,
                    daily_volume_brl REAL,
                    listing_date TEXT,
                    is_active BOOLEAN,
                    exchange TEXT,
                    sector TEXT,
                    source TEXT,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    data_hash TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_market_cap ON asset_metadata(market_cap_usd);
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_volume ON asset_metadata(daily_volume_brl);
            """)
    
    async def store_metadata(self, metadata: Dict[str, AssetMetadata]) -> bool:
        """Store asset metadata in SQLite database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                for symbol, meta in metadata.items():
                    # Calculate hash for change detection
                    meta_dict = asdict(meta)
                    meta_hash = hashlib.md5(json.dumps(meta_dict, sort_keys=True).encode()).hexdigest()
                    
                    conn.execute("""
                        INSERT OR REPLACE INTO asset_metadata 
                        (symbol, name, market_cap_usd, market_cap_brl, daily_volume_brl,
                         listing_date, is_active, exchange, sector, source, data_hash)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        meta.symbol,
                        meta.name,
                        meta.market_cap_usd,
                        meta.market_cap_brl,
                        meta.daily_volume_brl,
                        meta.listing_date.isoformat() if meta.listing_date else None,
                        meta.is_active,
                        meta.exchange,
                        meta.sector,
                        meta.source,
                        meta_hash
                    ))
                
                conn.commit()
            
            logger.info(f"Stored metadata for {len(metadata)} assets")
            return True
            
        except Exception as e:
            logger.error(f"Error storing metadata: {e}")
            return False
    
    async def retrieve_metadata(self, symbols: Optional[List[str]] = None) -> Dict[str, AssetMetadata]:
        """Retrieve asset metadata from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                if symbols:
                    placeholders = ','.join('?' * len(symbols))
                    query = f"SELECT * FROM asset_metadata WHERE symbol IN ({placeholders})"
                    cursor = conn.execute(query, symbols)
                else:
                    cursor = conn.execute("SELECT * FROM asset_metadata")
                
                metadata = {}
                for row in cursor:
                    metadata[row['symbol']] = AssetMetadata(
                        symbol=row['symbol'],
                        name=row['name'],
                        market_cap_usd=row['market_cap_usd'],
                        market_cap_brl=row['market_cap_brl'],
                        daily_volume_brl=row['daily_volume_brl'],
                        listing_date=datetime.fromisoformat(row['listing_date']) if row['listing_date'] else None,
                        is_active=bool(row['is_active']),
                        exchange=row['exchange'],
                        sector=row['sector'],
                        source=row['source']
                    )
                
                logger.info(f"Retrieved metadata for {len(metadata)} assets")
                return metadata
                
        except Exception as e:
            logger.error(f"Error retrieving metadata: {e}")
            return {}
    
    def get_statistics(self) -> Dict:
        """Get metadata storage statistics"""
        stats = {
            'total_assets': 0,
            'active_assets': 0,
            'exchanges': [],
            'sectors': [],
            'avg_market_cap_usd': 0
        }
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Total assets
                cursor = conn.execute("SELECT COUNT(*) FROM asset_metadata")
                stats['total_assets'] = cursor.fetchone()[0]
                
                # Active assets
                cursor = conn.execute("SELECT COUNT(*) FROM asset_metadata WHERE is_active = 1")
                stats['active_assets'] = cursor.fetchone()[0]
                
                # Exchanges
                cursor = conn.execute("SELECT DISTINCT exchange FROM asset_metadata")
                stats['exchanges'] = [row[0] for row in cursor.fetchall()]
                
                # Sectors  
                cursor = conn.execute("SELECT DISTINCT sector FROM asset_metadata WHERE sector IS NOT NULL")
                stats['sectors'] = [row[0] for row in cursor.fetchall()]
                
                # Average market cap
                cursor = conn.execute("SELECT AVG(market_cap_usd) FROM asset_metadata WHERE market_cap_usd IS NOT NULL")
                result = cursor.fetchone()[0]
                stats['avg_market_cap_usd'] = result if result else 0
                
        except Exception as e:
            logger.error(f"Error getting metadata statistics: {e}")
        
        return stats


class FeaturesStorage:
    """Storage for calculated features (technical indicators, scores)"""
    
    def __init__(self, base_path: Path, config: StorageConfig):
        self.base_path = base_path
        self.config = config
        base_path.mkdir(parents=True, exist_ok=True)
    
    async def store_features(self, features: pd.DataFrame, feature_type: str) -> bool:
        """Store calculated features"""
        try:
            if features.empty:
                logger.warning(f"No features to store for type: {feature_type}")
                return False
            
            # Create feature type directory
            feature_dir = self.base_path / feature_type
            feature_dir.mkdir(exist_ok=True)
            
            # Store with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = feature_dir / f"features_{feature_type}_{timestamp}.parquet"
            
            compression = 'snappy' if self.config.enable_compression else None
            features.to_parquet(file_path, compression=compression)
            
            logger.info(f"Stored {len(features)} feature records of type: {feature_type}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing features: {e}")
            return False
    
    async def retrieve_features(self, symbols: List[str], start_date: datetime, 
                               end_date: datetime, feature_type: str) -> pd.DataFrame:
        """Retrieve stored features"""
        try:
            feature_dir = self.base_path / feature_type
            
            if not feature_dir.exists():
                logger.warning(f"No features directory for type: {feature_type}")
                return pd.DataFrame()
            
            # Find latest feature file
            feature_files = list(feature_dir.glob("features_*.parquet"))
            if not feature_files:
                logger.warning(f"No feature files found for type: {feature_type}")
                return pd.DataFrame()
            
            # Get most recent file
            latest_file = max(feature_files, key=lambda p: p.stat().st_mtime)
            
            df = pd.read_parquet(latest_file)
            
            # Filter by symbols and date range
            if symbols:
                # Assuming MultiIndex columns with (symbol, feature) format
                symbol_columns = [col for col in df.columns if col[0] in symbols]
                df = df[symbol_columns]
            
            if start_date and end_date:
                df = df.loc[start_date:end_date]
            
            logger.info(f"Retrieved features: {len(df)} rows, {len(df.columns)} columns")
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving features: {e}")
            return pd.DataFrame()
    
    def get_statistics(self) -> Dict:
        """Get features storage statistics"""
        stats = {
            'feature_types': 0,
            'total_files': 0,
            'total_size_mb': 0
        }
        
        try:
            # Count feature types (subdirectories)
            feature_dirs = [d for d in self.base_path.iterdir() if d.is_dir()]
            stats['feature_types'] = len(feature_dirs)
            
            # Count files and size
            for file_path in self.base_path.rglob("*.parquet"):
                stats['total_files'] += 1
                stats['total_size_mb'] += file_path.stat().st_size / (1024 * 1024)
                
        except Exception as e:
            logger.error(f"Error getting features storage statistics: {e}")
        
        return stats


class PortfolioStorage:
    """Storage for portfolio data (positions, trades, performance)"""
    
    def __init__(self, base_path: Path, config: StorageConfig):
        self.base_path = base_path
        self.config = config
        base_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize portfolio database
        self.db_path = base_path / "portfolio.db"
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize portfolio database"""
        with sqlite3.connect(self.db_path) as conn:
            # Positions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    avg_cost REAL NOT NULL,
                    current_price REAL,
                    market_value REAL,
                    unrealized_pnl REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Trades table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    amount REAL NOT NULL,
                    fees REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    order_id TEXT,
                    execution_status TEXT
                )
            """)
            
            # Performance table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE NOT NULL,
                    total_value REAL,
                    total_cost REAL,
                    unrealized_pnl REAL,
                    realized_pnl REAL,
                    daily_return REAL,
                    cumulative_return REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL
                )
            """)
    
    def get_statistics(self) -> Dict:
        """Get portfolio storage statistics"""
        stats = {
            'total_positions': 0,
            'total_trades': 0,
            'performance_records': 0
        }
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM positions")
                stats['total_positions'] = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(*) FROM trades")
                stats['total_trades'] = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(*) FROM performance")
                stats['performance_records'] = cursor.fetchone()[0]
                
        except Exception as e:
            logger.error(f"Error getting portfolio statistics: {e}")
        
        return stats


class SystemStateStorage:
    """Storage for system state and configuration"""
    
    def __init__(self, base_path: Path, config: StorageConfig):
        self.base_path = base_path
        self.config = config
        base_path.mkdir(parents=True, exist_ok=True)
    
    def save_system_state(self, state: Dict) -> bool:
        """Save system state to disk"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            state_file = self.base_path / "state" / f"system_state_{timestamp}.json"
            state_file.parent.mkdir(exist_ok=True)
            
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
            # Keep only latest 10 state files
            state_files = sorted(self.base_path.glob("state/system_state_*.json"))
            for old_file in state_files[:-10]:
                old_file.unlink()
            
            logger.info("System state saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error saving system state: {e}")
            return False
    
    def load_system_state(self) -> Dict:
        """Load latest system state"""
        try:
            state_files = sorted(self.base_path.glob("state/system_state_*.json"))
            if not state_files:
                logger.warning("No system state files found")
                return {}
            
            latest_file = state_files[-1]
            with open(latest_file, 'r') as f:
                state = json.load(f)
            
            logger.info(f"Loaded system state from: {latest_file.name}")
            return state
            
        except Exception as e:
            logger.error(f"Error loading system state: {e}")
            return {}

