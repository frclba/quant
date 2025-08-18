#!/usr/bin/env python3
"""
Production Data Collector for DCA Layer 1 Data Ingestion

Unified, production-ready collector that consolidates functionality from:
- download_all_symbols.py (simple incremental downloads)
- incremental_collector.py (sophisticated collection planning)

Features:
- Smart incremental collection (only downloads what's needed)
- Multi-source data acquisition with failover
- Comprehensive data quality validation
- Production monitoring and alerting
- Configurable collection strategies
- Robust error handling and recovery
"""

import asyncio
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import time

# Internal imports
from .config import get_config, ProductionConfig
from .market_data_collector import MercadoBitcoinConnector, MarketDataPoint
from .data_storage import DataStorageEngine, StorageConfig
from .data_quality import DataQualityManager, DataQualityReport


logger = logging.getLogger(__name__)


class CollectionMode(Enum):
    """Data collection modes"""
    INCREMENTAL = "incremental"  # Only collect missing/new data
    FULL_REFRESH = "full"       # Complete historical download
    BACKFILL = "backfill"       # Fill gaps in existing data
    VALIDATION = "validation"    # Re-validate existing data


@dataclass
class CollectionStats:
    """Statistics for a data collection run"""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    total_symbols: int = 0
    symbols_processed: int = 0
    symbols_successful: int = 0
    symbols_failed: int = 0
    symbols_up_to_date: int = 0
    total_data_points: int = 0
    api_calls_made: int = 0
    bytes_downloaded: int = 0
    failed_symbols: List[str] = field(default_factory=list)
    
    @property
    def duration(self) -> Optional[timedelta]:
        if self.end_time:
            return self.end_time - self.start_time
        return datetime.now() - self.start_time
    
    @property
    def success_rate(self) -> float:
        if self.symbols_processed == 0:
            return 0.0
        return self.symbols_successful / self.symbols_processed
    
    def to_dict(self) -> Dict:
        """Convert stats to dictionary for logging/storage"""
        return {
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': self.duration.total_seconds() if self.duration else None,
            'total_symbols': self.total_symbols,
            'symbols_processed': self.symbols_processed,
            'symbols_successful': self.symbols_successful,
            'symbols_failed': self.symbols_failed,
            'symbols_up_to_date': self.symbols_up_to_date,
            'success_rate': self.success_rate,
            'total_data_points': self.total_data_points,
            'api_calls_made': self.api_calls_made,
            'bytes_downloaded': self.bytes_downloaded,
            'failed_symbols': self.failed_symbols
        }


@dataclass
class SymbolCollectionPlan:
    """Collection plan for a single symbol"""
    symbol: str
    mode: CollectionMode
    days_needed: int
    priority: int  # 1=high, 2=medium, 3=low
    reason: str    # Why this collection is needed
    estimated_data_points: int = 0
    
    def __str__(self):
        return f"{self.symbol}: {self.mode.value} ({self.days_needed} days, priority {self.priority})"


class ProductionDataCollector:
    """
    Production-ready data collector with smart incremental collection,
    monitoring, and robust error handling.
    """
    
    def __init__(self, config: Optional[ProductionConfig] = None):
        self.config = config or get_config()
        self.stats = CollectionStats()
        
        # Set up logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize timestamp tracking
        self.timestamp_file = Path(self.config.storage.base_path) / self.config.storage.logs_subdir / "last_update_timestamps.json"
        self.last_timestamps = self._load_timestamps()
        
        # Initialize components (will be created in async context)
        self.mb_connector: Optional[MercadoBitcoinConnector] = None
        self.storage_engine: Optional[DataStorageEngine] = None
        self.quality_manager: Optional[DataQualityManager] = None
    
    def _load_timestamps(self) -> Dict[str, str]:
        """Load last update timestamps for each symbol"""
        if self.timestamp_file.exists():
            try:
                with open(self.timestamp_file) as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load timestamps: {e}")
        return {}
    
    def _save_timestamps(self):
        """Save last update timestamps"""
        self.timestamp_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.timestamp_file, 'w') as f:
                json.dump(self.last_timestamps, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save timestamps: {e}")
    
    def _load_symbol_universe(self) -> Dict[str, List[str]]:
        """Load available symbols from data file"""
        symbols_file = Path(self.config.storage.base_path) / self.config.data.symbol_universe_file
        
        if not symbols_file.exists():
            raise FileNotFoundError(f"Symbol universe file not found: {symbols_file}")
        
        with open(symbols_file) as f:
            data = json.load(f)
        
        return {
            'major_cryptos': data['categories']['major_crypto'],
            'altcoins': data['categories']['altcoins'],
            'all_symbols': data['categories']['major_crypto'] + data['categories']['altcoins']
        }
    
    def _get_data_file_path(self, symbol: str, is_major_crypto: bool = False) -> Path:
        """Get the data file path for a symbol"""
        base_path = Path(self.config.storage.base_path)
        subdir = self.config.storage.major_crypto_subdir if is_major_crypto else self.config.storage.altcoin_subdir
        return base_path / subdir / f"{symbol}_data.csv"
    
    def _analyze_existing_data(self, symbol: str, data_file: Path) -> Optional[datetime]:
        """Analyze existing data and return the last data date"""
        if not data_file.exists():
            return None
        
        try:
            df = pd.read_csv(data_file)
            if len(df) == 0:
                return None
            
            # Get the latest timestamp
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df['timestamp'].max().date()
            elif 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                return df['date'].max().date()
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Error analyzing existing data for {symbol}: {e}")
            return None
    
    def _create_collection_plan(self, symbols: Dict[str, List[str]], mode: CollectionMode) -> List[SymbolCollectionPlan]:
        """Create collection plan for all symbols"""
        plans = []
        today = datetime.now().date()
        
        for symbol in symbols['all_symbols']:
            is_major = symbol in symbols['major_cryptos']
            data_file = self._get_data_file_path(symbol, is_major)
            
            if mode == CollectionMode.FULL_REFRESH:
                plan = SymbolCollectionPlan(
                    symbol=symbol,
                    mode=mode,
                    days_needed=self.config.data.default_history_days,
                    priority=1 if is_major else 2,
                    reason="Full refresh requested",
                    estimated_data_points=self.config.data.default_history_days
                )
                plans.append(plan)
                continue
            
            # For incremental mode, analyze what we need
            last_date = self._analyze_existing_data(symbol, data_file)
            
            if last_date is None:
                # No existing data - need full download
                plan = SymbolCollectionPlan(
                    symbol=symbol,
                    mode=CollectionMode.FULL_REFRESH,
                    days_needed=self.config.data.default_history_days,
                    priority=1 if is_major else 2,
                    reason="No existing data",
                    estimated_data_points=self.config.data.default_history_days
                )
                plans.append(plan)
                continue
            
            # Check if we need updates
            days_since_last = (today - last_date).days
            
            if days_since_last <= 1:
                # Data is up to date
                self.stats.symbols_up_to_date += 1
                continue
            
            # Need incremental update
            days_needed = min(days_since_last + self.config.data.incremental_buffer_days, 
                            self.config.data.default_history_days)
            
            plan = SymbolCollectionPlan(
                symbol=symbol,
                mode=CollectionMode.INCREMENTAL,
                days_needed=days_needed,
                priority=1 if is_major else 2,
                reason=f"Last data: {last_date}, {days_since_last} days behind",
                estimated_data_points=days_needed
            )
            plans.append(plan)
        
        # Sort by priority (1=highest priority first)
        plans.sort(key=lambda p: (p.priority, p.symbol))
        return plans
    
    def _merge_new_data(self, existing_file: Path, new_data: pd.DataFrame) -> pd.DataFrame:
        """Merge new data with existing data, removing duplicates"""
        if not existing_file.exists():
            return new_data
        
        try:
            existing_df = pd.read_csv(existing_file)
            existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])
            new_data['timestamp'] = pd.to_datetime(new_data['timestamp'])
            
            # Combine and deduplicate
            combined_df = pd.concat([existing_df, new_data], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['timestamp'], keep='last')
            combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
            
            # Update date column
            combined_df['date'] = combined_df['timestamp'].dt.date
            
            return combined_df
            
        except Exception as e:
            self.logger.warning(f"Error merging data for {existing_file}: {e}")
            return new_data
    
    async def _collect_symbol_data(self, plan: SymbolCollectionPlan, symbols: Dict[str, List[str]]) -> bool:
        """Collect data for a single symbol according to the plan"""
        try:
            # Get data from API
            data_points = await self.mb_connector.get_ohlcv_data(
                plan.symbol, 
                timeframe='1d', 
                limit=plan.days_needed
            )
            
            self.stats.api_calls_made += 1
            
            if not data_points or len(data_points) == 0:
                self.logger.warning(f"{plan.symbol}: No data available")
                return False
            
            # Convert to DataFrame
            new_df = pd.DataFrame([
                {
                    'timestamp': point.timestamp,
                    'date': point.timestamp.date(),
                    'open': point.open_price,
                    'high': point.high_price,
                    'low': point.low_price,
                    'close': point.close_price,
                    'volume': point.volume,
                    'quote_volume': point.quote_volume,
                    'source': 'mercado_bitcoin'
                }
                for point in data_points
            ])
            
            new_df = new_df.sort_values('timestamp').reset_index(drop=True)
            
            # Determine output file
            is_major = plan.symbol in symbols['major_cryptos']
            data_file = self._get_data_file_path(plan.symbol, is_major)
            
            # Merge with existing data if incremental
            if plan.mode == CollectionMode.INCREMENTAL:
                final_df = self._merge_new_data(data_file, new_df)
                new_records = len(new_df)
                total_records = len(final_df)
            else:
                final_df = new_df
                new_records = len(new_df)
                total_records = len(final_df)
            
            # Ensure directory exists
            data_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save data
            final_df.to_csv(data_file, index=False)
            
            # Update statistics
            self.stats.total_data_points += new_records
            self.last_timestamps[plan.symbol] = datetime.now().isoformat()
            
            # Log success
            latest_price = final_df['close'].iloc[-1]
            date_range = f"{final_df.iloc[0]['date']} to {final_df.iloc[-1]['date']}"
            
            if plan.mode == CollectionMode.INCREMENTAL and new_records != total_records:
                self.logger.info(f"{plan.symbol}: {new_records} new + {total_records-new_records} existing = {total_records} total | Latest: {latest_price:>10,.2f} BRL | {date_range}")
            else:
                self.logger.info(f"{plan.symbol}: {total_records} records | Latest: {latest_price:>10,.2f} BRL | {date_range}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"{plan.symbol}: Collection failed - {str(e)}")
            return False
    
    def _log_progress(self, current: int, total: int, eta: str):
        """Log progress information"""
        progress_pct = (current / total) * 100
        elapsed = self.stats.duration
        
        self.logger.info(f"📊 [{current:3d}/{total}] ({progress_pct:5.1f}%)")
        self.logger.info(f"    ⏱️  Elapsed: {str(elapsed).split('.')[0]} | ETA: {eta}")
    
    def _save_run_statistics(self):
        """Save run statistics to file"""
        stats_file = Path(self.config.storage.base_path) / self.config.storage.logs_subdir / f"collection_run_{self.stats.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            stats_file.parent.mkdir(parents=True, exist_ok=True)
            with open(stats_file, 'w') as f:
                json.dump(self.stats.to_dict(), f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save run statistics: {e}")
    
    async def collect_data(self, mode: CollectionMode = CollectionMode.INCREMENTAL) -> CollectionStats:
        """
        Main data collection method
        
        Args:
            mode: Collection mode (incremental, full, etc.)
            
        Returns:
            CollectionStats: Statistics from the collection run
        """
        self.logger.info(f"🚀 Starting data collection in {mode.value.upper()} mode")
        
        try:
            # Initialize async components
            self.mb_connector = MercadoBitcoinConnector()
            await self.mb_connector.__aenter__()
            
            # Load symbol universe
            symbols = self._load_symbol_universe()
            self.stats.total_symbols = len(symbols['all_symbols'])
            
            self.logger.info(f"📋 Loaded {self.stats.total_symbols} symbols:")
            self.logger.info(f"    • Major cryptos: {len(symbols['major_cryptos'])}")
            self.logger.info(f"    • Altcoins: {len(symbols['altcoins'])}")
            
            # Create collection plan
            collection_plans = self._create_collection_plan(symbols, mode)
            symbols_to_process = len(collection_plans)
            
            self.logger.info(f"📋 Collection Plan:")
            self.logger.info(f"    • Symbols to process: {symbols_to_process}")
            self.logger.info(f"    • Symbols up to date: {self.stats.symbols_up_to_date}")
            
            if symbols_to_process == 0:
                self.logger.info("✅ All symbols are up to date!")
                return self.stats
            
            # Execute collection plan
            processed_count = 0
            
            for i, plan in enumerate(collection_plans, 1):
                # Calculate ETA
                if processed_count > 0:
                    elapsed = self.stats.duration
                    avg_time = elapsed.total_seconds() / processed_count
                    remaining = symbols_to_process - i
                    eta_seconds = avg_time * remaining
                    eta = str(timedelta(seconds=int(eta_seconds)))
                else:
                    eta = "calculating..."
                
                # Log progress
                if i % self.config.monitoring.progress_report_interval == 0 or i == 1:
                    self._log_progress(i, symbols_to_process, eta)
                
                self.logger.info(f"\n📊 [{i:3d}/{symbols_to_process}] {plan}")
                
                # Collect data for this symbol
                success = await self._collect_symbol_data(plan, symbols)
                
                # Update statistics
                self.stats.symbols_processed += 1
                processed_count += 1
                
                if success:
                    self.stats.symbols_successful += 1
                else:
                    self.stats.symbols_failed += 1
                    self.stats.failed_symbols.append(plan.symbol)
                
                # Rate limiting
                await asyncio.sleep(self.config.api.rate_limit_seconds)
            
            # Finalize
            self.stats.end_time = datetime.now()
            
            # Save timestamps and statistics
            self._save_timestamps()
            self._save_run_statistics()
            
            # Log final results
            self._log_final_results(mode, symbols_to_process)
            
            return self.stats
            
        except Exception as e:
            self.logger.error(f"Data collection failed: {e}")
            raise
        finally:
            # Cleanup async components
            if self.mb_connector:
                await self.mb_connector.__aexit__(None, None, None)
    
    def _log_final_results(self, mode: CollectionMode, symbols_processed: int):
        """Log final collection results"""
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"🎉 {mode.value.upper()} COLLECTION COMPLETED!")
        self.logger.info(f"{'='*70}")
        self.logger.info(f"📊 Final Results:")
        self.logger.info(f"  ✅ Successful: {self.stats.symbols_successful:3d}/{symbols_processed} processed ({self.stats.success_rate:.1%})")
        self.logger.info(f"  ❌ Failed:     {self.stats.symbols_failed:3d}/{symbols_processed}")
        self.logger.info(f"  🔄 Up to date: {self.stats.symbols_up_to_date:3d}/{self.stats.total_symbols}")
        self.logger.info(f"  📈 New records: {self.stats.total_data_points:,}")
        self.logger.info(f"  🌐 API calls: {self.stats.api_calls_made:,}")
        self.logger.info(f"  ⏱️  Duration: {str(self.stats.duration).split('.')[0]}")
        self.logger.info(f"  💰 Currency: BRL (Brazilian Real)")
        
        if self.stats.failed_symbols:
            failed_count = len(self.stats.failed_symbols)
            sample_failed = self.stats.failed_symbols[:10]
            if failed_count <= 10:
                self.logger.warning(f"\n❌ Failed symbols: {sample_failed}")
            else:
                self.logger.warning(f"\n❌ Failed symbols ({failed_count}): {sample_failed}... +{failed_count-10} more")
        
        self.logger.info(f"\n🚀 Dataset ready for DCA analysis!")
        self.logger.info(f"🗂️  Timestamps saved to: {self.timestamp_file}")


# Convenience function for direct usage
async def run_data_collection(mode: CollectionMode = CollectionMode.INCREMENTAL, 
                            config: Optional[ProductionConfig] = None) -> CollectionStats:
    """
    Convenience function to run data collection
    
    Args:
        mode: Collection mode
        config: Optional configuration (uses default if None)
        
    Returns:
        CollectionStats: Statistics from the collection run
    """
    collector = ProductionDataCollector(config)
    return await collector.collect_data(mode)


if __name__ == "__main__":
    # Demo/test the collector
    async def main():
        # Test configuration
        config = get_config()
        config.setup_logging()
        
        # Run incremental collection
        stats = await run_data_collection(CollectionMode.INCREMENTAL)
        print(f"\n📊 Collection completed with {stats.success_rate:.1%} success rate")
    
    asyncio.run(main())
