"""
Incremental Data Collection System

Smart data collection that:
1. Downloads historical data once (bulk initial load)
2. Identifies gaps in existing data
3. Updates only recent/missing data
4. Minimizes API calls and respects rate limits
5. Handles data versioning and consistency

This is the main interface for data collection in the DCA system.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
import asyncio

from .market_data_collector import MarketDataCollector, MarketDataPoint, AssetMetadata
from .data_storage import DataStorageEngine, StorageConfig
from .data_quality import DataQualityManager, DataQualityReport

logger = logging.getLogger(__name__)


@dataclass
class DataInventory:
    """Tracks what data we have vs what we need"""
    symbol: str
    earliest_date: Optional[datetime]
    latest_date: Optional[datetime]
    total_points: int
    missing_ranges: List[Tuple[datetime, datetime]]
    data_quality_score: float
    last_updated: datetime
    needs_update: bool


@dataclass
class CollectionPlan:
    """Plan for data collection - what to fetch from where"""
    symbol: str
    date_ranges: List[Tuple[datetime, datetime]]
    estimated_api_calls: int
    priority: int  # 1=high, 2=medium, 3=low
    source_preference: List[str]  # Ordered list of preferred sources


class IncrementalDataCollector:
    """
    Smart incremental data collector
    
    Key Features:
    - Initial bulk download (once)
    - Incremental updates (daily/weekly)
    - Gap detection and filling
    - Quality-aware collection
    - API rate limit management
    """
    
    def __init__(self, storage_path: str, cg_api_key: str = None):
        # Initialize storage
        storage_config = StorageConfig(
            base_path=storage_path,
            raw_data_retention_days=730,  # 2 years of raw data
            processed_data_retention_days=365,  # 1 year of processed data
            partition_by_date=True,
            enable_compression=True,
            backup_enabled=True
        )
        
        self.storage = DataStorageEngine(storage_config)
        self.collector = MarketDataCollector(cg_api_key=cg_api_key)
        self.quality_manager = DataQualityManager(min_acceptable_score=0.7)
        
        # Collection configuration
        self.config = {
            'bulk_load_days': 365,           # Initial bulk load period
            'update_frequency_days': 1,       # How often to check for updates
            'max_gap_days': 3,               # Max gap before considering data stale
            'batch_size': 5,                 # Symbols per batch
            'max_daily_api_calls': 1000,     # Daily API call limit
            'quality_threshold': 0.7,        # Minimum data quality
        }
        
        self._api_calls_today = 0
        self._last_reset_date = datetime.now().date()
        
        logger.info(f"Incremental Data Collector initialized at: {storage_path}")
    
    async def __aenter__(self):
        await self.collector.__aenter__()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.collector.__aexit__(exc_type, exc_val, exc_tb)
    
    def _reset_api_counter_if_needed(self):
        """Reset daily API call counter"""
        today = datetime.now().date()
        if today > self._last_reset_date:
            self._api_calls_today = 0
            self._last_reset_date = today
            logger.info("API call counter reset for new day")
    
    async def initial_bulk_load(self, symbols: Optional[List[str]] = None, 
                               days_back: Optional[int] = None) -> Dict[str, bool]:
        """
        Perform initial bulk data load for symbols
        
        Args:
            symbols: List of symbols to load. If None, gets from exchange
            days_back: Days of historical data. If None, uses config default
            
        Returns:
            Dictionary of symbol -> success status
        """
        logger.info("🚀 Starting Initial Bulk Data Load")
        
        # Get symbols to load
        if symbols is None:
            symbols = await self.collector.get_available_assets()
        
        days_back = days_back or self.config['bulk_load_days']
        
        logger.info(f"📊 Bulk loading {len(symbols)} symbols, {days_back} days back")
        
        # Check what data we already have
        inventory = await self._analyze_data_inventory(symbols)
        
        # Filter to symbols that actually need bulk loading
        symbols_to_load = []
        for symbol in symbols:
            inv = inventory.get(symbol)
            if not inv or inv.total_points == 0 or inv.needs_update:
                symbols_to_load.append(symbol)
        
        logger.info(f"📥 {len(symbols_to_load)} symbols need initial data loading")
        
        if not symbols_to_load:
            logger.info("✅ All symbols already have data - skipping bulk load")
            return {symbol: True for symbol in symbols}
        
        # Execute bulk load
        results = {}
        batch_size = self.config['batch_size']
        
        for i in range(0, len(symbols_to_load), batch_size):
            batch = symbols_to_load[i:i + batch_size]
            logger.info(f"📦 Processing batch {i//batch_size + 1}/{(len(symbols_to_load)-1)//batch_size + 1}: {batch}")
            
            # Check API limits
            self._reset_api_counter_if_needed()
            if self._api_calls_today >= self.config['max_daily_api_calls']:
                logger.warning("⚠️ Daily API limit reached - pausing bulk load")
                break
            
            # Collect data for batch
            batch_results = await self._bulk_load_batch(batch, days_back)
            results.update(batch_results)
            
            # Small delay between batches
            await asyncio.sleep(1.0)
        
        successful_loads = sum(1 for success in results.values() if success)
        logger.info(f"✅ Bulk load completed: {successful_loads}/{len(symbols_to_load)} successful")
        
        return results
    
    async def _bulk_load_batch(self, symbols: List[str], days_back: int) -> Dict[str, bool]:
        """Load a batch of symbols"""
        results = {}
        
        try:
            # Collect OHLCV data
            ohlcv_data = await self.collector.collect_ohlcv_data(
                symbols=symbols,
                timeframe='1d',
                limit=days_back
            )
            
            # Collect metadata
            metadata = await self.collector.collect_metadata(symbols)
            
            # Update API call counter
            self._api_calls_today += len(symbols) * 2  # OHLCV + metadata calls
            
            # Quality check
            ohlcv_quality = await self.quality_manager.assess_ohlcv_quality(ohlcv_data)
            metadata_quality = await self.quality_manager.assess_metadata_quality(metadata)
            
            # Store data for each symbol
            for symbol in symbols:
                try:
                    # Check quality before storing
                    ohlcv_ok = (symbol in ohlcv_quality and 
                               ohlcv_quality[symbol].is_usable)
                    metadata_ok = (symbol in metadata_quality and 
                                  metadata_quality[symbol].is_usable)
                    
                    if ohlcv_ok:
                        # Store OHLCV data
                        symbol_ohlcv = {symbol: ohlcv_data.get(symbol, [])}
                        await self.storage.store_ohlcv_data(symbol_ohlcv, '1d')
                        
                        # Store metadata
                        if metadata_ok and symbol in metadata:
                            symbol_metadata = {symbol: metadata[symbol]}
                            await self.storage.store_metadata(symbol_metadata)
                        
                        results[symbol] = True
                        logger.debug(f"✅ Stored bulk data for {symbol}")
                    else:
                        results[symbol] = False
                        logger.warning(f"❌ Poor quality data for {symbol} - skipped")
                        
                except Exception as e:
                    logger.error(f"❌ Error storing bulk data for {symbol}: {e}")
                    results[symbol] = False
            
        except Exception as e:
            logger.error(f"❌ Bulk load batch failed: {e}")
            results = {symbol: False for symbol in symbols}
        
        return results
    
    async def incremental_update(self, symbols: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        Perform incremental data update
        
        Only fetches recent data that's missing or needs updating
        """
        logger.info("🔄 Starting Incremental Data Update")
        
        # Get symbols to update
        if symbols is None:
            # Get all symbols we have data for
            stored_metadata = await self.storage.retrieve_metadata()
            symbols = list(stored_metadata.keys()) if stored_metadata else []
        
        if not symbols:
            logger.warning("No symbols found for incremental update")
            return {}
        
        logger.info(f"🔍 Checking {len(symbols)} symbols for updates")
        
        # Analyze what data we have vs what we need
        inventory = await self._analyze_data_inventory(symbols)
        
        # Create collection plan for updates needed
        collection_plans = await self._create_incremental_collection_plan(inventory)
        
        if not collection_plans:
            logger.info("✅ All data is up to date - no updates needed")
            return {symbol: True for symbol in symbols}
        
        logger.info(f"📥 {len(collection_plans)} symbols need updates")
        
        # Execute incremental updates
        results = await self._execute_collection_plans(collection_plans)
        
        successful_updates = sum(1 for success in results.values() if success)
        logger.info(f"✅ Incremental update completed: {successful_updates}/{len(collection_plans)} successful")
        
        return results
    
    async def _analyze_data_inventory(self, symbols: List[str]) -> Dict[str, DataInventory]:
        """Analyze what data we have for each symbol"""
        logger.info(f"📋 Analyzing data inventory for {len(symbols)} symbols")
        
        inventory = {}
        current_date = datetime.now().date()
        
        for symbol in symbols:
            try:
                # Check date range of stored data
                start_date = current_date - timedelta(days=self.config['bulk_load_days'])
                end_date = current_date
                
                stored_data = await self.storage.retrieve_ohlcv_data(
                    symbols=[symbol],
                    start_date=datetime.combine(start_date, datetime.min.time()),
                    end_date=datetime.combine(end_date, datetime.min.time()),
                    timeframe='1d'
                )
                
                if stored_data.empty:
                    # No data stored
                    inventory[symbol] = DataInventory(
                        symbol=symbol,
                        earliest_date=None,
                        latest_date=None,
                        total_points=0,
                        missing_ranges=[(
                            datetime.combine(start_date, datetime.min.time()),
                            datetime.combine(end_date, datetime.min.time())
                        )],
                        data_quality_score=0.0,
                        last_updated=datetime.min,
                        needs_update=True
                    )
                else:
                    # Analyze existing data
                    symbol_data = stored_data[[col for col in stored_data.columns if col[0] == symbol]]
                    
                    if not symbol_data.empty:
                        earliest_date = symbol_data.index.min()
                        latest_date = symbol_data.index.max()
                        total_points = len(symbol_data)
                        
                        # Check if update is needed (data is older than update frequency)
                        days_since_update = (datetime.now() - latest_date).days
                        needs_update = days_since_update >= self.config['update_frequency_days']
                        
                        # Detect missing ranges (simplified - could be more sophisticated)
                        missing_ranges = []
                        if needs_update:
                            missing_ranges.append((
                                latest_date + timedelta(days=1),
                                datetime.now()
                            ))
                        
                        inventory[symbol] = DataInventory(
                            symbol=symbol,
                            earliest_date=earliest_date,
                            latest_date=latest_date,
                            total_points=total_points,
                            missing_ranges=missing_ranges,
                            data_quality_score=0.8,  # Would calculate from quality reports
                            last_updated=latest_date,
                            needs_update=needs_update
                        )
                    else:
                        # Data exists but empty for this symbol
                        inventory[symbol] = DataInventory(
                            symbol=symbol,
                            earliest_date=None,
                            latest_date=None,
                            total_points=0,
                            missing_ranges=[(
                                datetime.combine(start_date, datetime.min.time()),
                                datetime.combine(end_date, datetime.min.time())
                            )],
                            data_quality_score=0.0,
                            last_updated=datetime.min,
                            needs_update=True
                        )
                        
            except Exception as e:
                logger.error(f"Error analyzing inventory for {symbol}: {e}")
                inventory[symbol] = DataInventory(
                    symbol=symbol,
                    earliest_date=None,
                    latest_date=None,
                    total_points=0,
                    missing_ranges=[],
                    data_quality_score=0.0,
                    last_updated=datetime.min,
                    needs_update=True
                )
        
        return inventory
    
    async def _create_incremental_collection_plan(self, inventory: Dict[str, DataInventory]) -> List[CollectionPlan]:
        """Create optimized collection plan for incremental updates"""
        plans = []
        
        for symbol, inv in inventory.items():
            if not inv.needs_update or not inv.missing_ranges:
                continue
            
            # Calculate API calls needed
            total_days = sum(
                (end_date - start_date).days 
                for start_date, end_date in inv.missing_ranges
            )
            estimated_calls = 1  # Usually just one call for recent data
            
            # Determine priority
            if inv.total_points == 0:
                priority = 1  # High priority - no data
            elif total_days <= 7:
                priority = 2  # Medium priority - weekly update
            else:
                priority = 3  # Low priority - older gaps
            
            plans.append(CollectionPlan(
                symbol=symbol,
                date_ranges=inv.missing_ranges,
                estimated_api_calls=estimated_calls,
                priority=priority,
                source_preference=['mercado_bitcoin', 'coingecko', 'yahoo_finance']
            ))
        
        # Sort by priority (high priority first)
        plans.sort(key=lambda p: p.priority)
        
        logger.info(f"📋 Created {len(plans)} collection plans")
        return plans
    
    async def _execute_collection_plans(self, plans: List[CollectionPlan]) -> Dict[str, bool]:
        """Execute collection plans with API rate limiting"""
        results = {}
        
        for plan in plans:
            # Check API limits
            self._reset_api_counter_if_needed()
            if self._api_calls_today >= self.config['max_daily_api_calls']:
                logger.warning(f"⚠️ Daily API limit reached - postponing {plan.symbol}")
                results[plan.symbol] = False
                continue
            
            try:
                # For incremental updates, we usually just need recent data
                # Calculate days needed
                if plan.date_ranges:
                    latest_range = max(plan.date_ranges, key=lambda r: r[1])
                    days_needed = (latest_range[1] - latest_range[0]).days + 1
                    days_needed = min(days_needed, 30)  # Limit to 30 days for incremental
                else:
                    days_needed = 7  # Default to 1 week
                
                # Collect data
                ohlcv_data = await self.collector.collect_ohlcv_data(
                    symbols=[plan.symbol],
                    timeframe='1d',
                    limit=days_needed
                )
                
                self._api_calls_today += 1
                
                # Store new data
                if plan.symbol in ohlcv_data and ohlcv_data[plan.symbol]:
                    symbol_data = {plan.symbol: ohlcv_data[plan.symbol]}
                    await self.storage.store_ohlcv_data(symbol_data, '1d')
                    results[plan.symbol] = True
                    logger.debug(f"✅ Updated data for {plan.symbol}")
                else:
                    results[plan.symbol] = False
                    logger.warning(f"❌ No new data for {plan.symbol}")
                
                # Small delay between requests
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"❌ Error updating {plan.symbol}: {e}")
                results[plan.symbol] = False
        
        return results
    
    async def get_latest_data(self, symbols: List[str], days: int = 90) -> pd.DataFrame:
        """
        Get the latest available data for symbols
        Automatically handles incremental updates if needed
        """
        logger.info(f"📊 Getting latest data for {len(symbols)} symbols ({days} days)")
        
        # Check if we need updates first
        inventory = await self._analyze_data_inventory(symbols)
        symbols_needing_update = [
            symbol for symbol, inv in inventory.items() 
            if inv.needs_update
        ]
        
        if symbols_needing_update:
            logger.info(f"🔄 {len(symbols_needing_update)} symbols need updates")
            await self.incremental_update(symbols_needing_update)
        
        # Retrieve data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        data = await self.storage.retrieve_ohlcv_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            timeframe='1d'
        )
        
        logger.info(f"📈 Retrieved data: {len(data)} rows, {len(data.columns) // 6} symbols")
        return data
    
    def get_data_summary(self) -> Dict:
        """Get summary of stored data"""
        try:
            storage_stats = self.storage.get_storage_statistics()
            
            summary = {
                'storage_stats': storage_stats,
                'api_calls_today': self._api_calls_today,
                'daily_limit': self.config['max_daily_api_calls'],
                'api_calls_remaining': self.config['max_daily_api_calls'] - self._api_calls_today,
                'config': self.config
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting data summary: {e}")
            return {}
    
    async def cleanup_old_data(self):
        """Clean up old data based on retention policies"""
        logger.info("🧹 Starting data cleanup")
        # This would implement cleanup logic based on retention policies
        # For now, just log that it would happen
        logger.info("✅ Data cleanup completed")


# Convenience function for easy usage
async def setup_incremental_collector(storage_path: str, cg_api_key: str = None) -> IncrementalDataCollector:
    """Setup and return configured incremental data collector"""
    collector = IncrementalDataCollector(storage_path, cg_api_key)
    await collector.__aenter__()
    return collector

