"""
Module 1.1: Market Data Collector

Primary data acquisition engine for the DCA system.
Uses proven data sources:
- Mercado Bitcoin API (Brazilian crypto exchange) - https://api.mercadobitcoin.net/api/v4/docs
- CoinGecko API (market data and metadata)
- Yahoo Finance (fallback price data)

Based on Notebook 1 concepts: Price modeling, returns, correlation data acquisition.
"""

import pandas as pd
import numpy as np
import requests
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import asyncio
import aiohttp
from urllib.parse import urljoin
import yfinance as yf
from pycoingecko import CoinGeckoAPI

logger = logging.getLogger(__name__)


@dataclass
class MarketDataPoint:
    """Single market data point structure"""
    symbol: str
    timestamp: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    quote_volume: float
    source: str


@dataclass
class AssetMetadata:
    """Asset metadata structure"""
    symbol: str
    name: str
    market_cap_usd: Optional[float]
    market_cap_brl: Optional[float] 
    daily_volume_brl: Optional[float]
    listing_date: Optional[datetime]
    is_active: bool
    exchange: str
    sector: Optional[str]
    source: str


class DataConnector(ABC):
    """Abstract base class for data source connectors"""
    
    @abstractmethod
    async def get_ohlcv_data(self, symbol: str, timeframe: str, limit: int) -> List[MarketDataPoint]:
        """Fetch OHLCV data for a symbol"""
        pass
    
    @abstractmethod
    async def get_asset_metadata(self, symbol: str) -> AssetMetadata:
        """Fetch metadata for an asset"""
        pass
    
    @abstractmethod
    async def get_available_symbols(self) -> List[str]:
        """Get list of available trading symbols"""
        pass


class MercadoBitcoinConnector(DataConnector):
    """
    Mercado Bitcoin API connector - Primary Brazilian exchange
    API Documentation: https://api.mercadobitcoin.net/api/v4/docs
    """
    
    def __init__(self):
        self.base_url = "https://api.mercadobitcoin.net"
        self.session = None
        self.rate_limit_delay = 1.1  # 1 request per second = 1.1 seconds to be safe
        self.last_request_time = 0
        
    async def __aenter__(self):
        # Configure SSL handling for aiohttp
        import ssl
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # Create session with SSL context
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        self.session = aiohttp.ClientSession(connector=connector)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def _rate_limit(self):
        """Enforce rate limiting for public API"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()
    
    async def _make_request(self, endpoint: str, params: Dict = None) -> Union[Dict, List]:
        """Make API request with error handling and rate limiting"""
        await self._rate_limit()
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.debug(f"MB API: Fetched {endpoint} successfully")
                    return data
                elif response.status == 429:
                    logger.warning("MB API: Rate limited, increasing delay")
                    self.rate_limit_delay *= 2
                    await asyncio.sleep(self.rate_limit_delay)
                    return await self._make_request(endpoint, params)  # Retry
                else:
                    logger.error(f"MB API: Error {response.status} for {endpoint}")
                    return {}
                    
        except Exception as e:
            logger.error(f"MB API: Connection error for {endpoint}: {e}")
            return {}
    
    async def get_available_symbols(self) -> List[str]:
        """Get all available trading symbols from Mercado Bitcoin"""
        try:
            # Use the symbols endpoint to get available trading pairs
            data = await self._make_request("/api/v4/symbols")
            
            if not data:
                logger.warning("MB API: No symbols data received")
                return []
            
            symbols = []
            # The API returns a list of symbol objects
            for symbol_info in data:
                if isinstance(symbol_info, dict):
                    symbol = symbol_info.get('symbol', '')
                    # Focus on BRL pairs and extract base currency
                    if symbol and symbol.endswith('-BRL'):
                        base_currency = symbol.replace('-BRL', '')
                        symbols.append(base_currency)
            
            logger.info(f"MB API: Found {len(symbols)} available symbols")
            return symbols
            
        except Exception as e:
            logger.error(f"MB API: Error fetching symbols: {e}")
            return []
    
    async def get_ohlcv_data(self, symbol: str, timeframe: str = "1d", limit: int = 100) -> List[MarketDataPoint]:
        """
        Fetch OHLCV data from Mercado Bitcoin
        
        Args:
            symbol: Trading symbol (e.g., 'BTC', 'ETH')  
            timeframe: Timeframe ('1m', '5m', '15m', '30m', '1h', '4h', '1d')
            limit: Number of data points to fetch
        """
        try:
            # MB uses SYMBOL-BRL format
            mb_symbol = f"{symbol}-BRL"
            
            # Map timeframe to MB format
            timeframe_map = {
                '1m': '1m',
                '5m': '5m', 
                '15m': '15m',
                '30m': '30m',
                '1h': '1h',
                '4h': '4h',
                '1d': '1d'
            }
            
            mb_timeframe = timeframe_map.get(timeframe, '1d')
            
            # Use the /api/v4/candles endpoint as recommended
            endpoint = "/api/v4/candles"
            params = {
                'symbol': mb_symbol,
                'resolution': mb_timeframe,
                'from': int((datetime.now() - timedelta(days=limit)).timestamp()),
                'to': int(datetime.now().timestamp())
            }
            
            data = await self._make_request(endpoint, params)
            
            if not data or not isinstance(data, dict):
                logger.warning(f"MB API: No OHLCV data for {symbol}")
                return []
            
            # MB API returns dictionary format: {'t': [...], 'o': [...], 'h': [...], 'l': [...], 'c': [...], 'v': [...]}
            if not all(key in data for key in ['t', 'o', 'h', 'l', 'c', 'v']):
                logger.warning(f"MB API: Invalid data format for {symbol}")
                return []
            
            timestamps = data['t']
            opens = data['o']
            highs = data['h']
            lows = data['l']
            closes = data['c']
            volumes = data['v']
            
            if not all(len(arr) == len(timestamps) for arr in [opens, highs, lows, closes, volumes]):
                logger.warning(f"MB API: Mismatched data arrays for {symbol}")
                return []
            
            market_data = []
            for i in range(len(timestamps)):
                try:
                    market_data.append(MarketDataPoint(
                        symbol=symbol,
                        timestamp=datetime.fromtimestamp(timestamps[i]),
                        open_price=float(opens[i]),
                        high_price=float(highs[i]), 
                        low_price=float(lows[i]),
                        close_price=float(closes[i]),
                        volume=float(volumes[i]),
                        quote_volume=float(volumes[i]) * float(closes[i]),
                        source='mercado_bitcoin'
                    ))
                except (ValueError, TypeError) as e:
                    logger.warning(f"MB API: Invalid data point for {symbol} at index {i}: {e}")
                    continue
            
            logger.info(f"MB API: Fetched {len(market_data)} data points for {symbol}")
            return market_data
            
        except Exception as e:
            logger.error(f"MB API: Error fetching OHLCV for {symbol}: {e}")
            return []
    
    async def get_asset_metadata(self, symbol: str) -> AssetMetadata:
        """Fetch metadata for an asset from Mercado Bitcoin"""
        try:
            mb_symbol = f"{symbol}-BRL"
            
            # Get 24h ticker for volume data
            ticker_data = await self._make_request(f"/api/v4/{mb_symbol}/ticker")
            
            if not ticker_data:
                logger.warning(f"MB API: No ticker data for {symbol}")
                return AssetMetadata(
                    symbol=symbol,
                    name=symbol,
                    market_cap_usd=None,
                    market_cap_brl=None,
                    daily_volume_brl=None,
                    listing_date=None,
                    is_active=False,
                    exchange='mercado_bitcoin',
                    sector=None,
                    source='mercado_bitcoin'
                )
            
            # Extract volume data
            daily_volume_brl = None
            if 'vol' in ticker_data:
                daily_volume_brl = float(ticker_data['vol'])
            
            return AssetMetadata(
                symbol=symbol,
                name=symbol,  # MB doesn't provide full names in basic API
                market_cap_usd=None,  # Not available from MB
                market_cap_brl=None,  # Not available from MB  
                daily_volume_brl=daily_volume_brl,
                listing_date=None,  # Not available from MB
                is_active=True,
                exchange='mercado_bitcoin',
                sector=None,  # Not available from MB
                source='mercado_bitcoin'
            )
            
        except Exception as e:
            logger.error(f"MB API: Error fetching metadata for {symbol}: {e}")
            return AssetMetadata(
                symbol=symbol, name=symbol, market_cap_usd=None, market_cap_brl=None,
                daily_volume_brl=None, listing_date=None, is_active=False,
                exchange='mercado_bitcoin', sector=None, source='mercado_bitcoin'
            )


class CoinGeckoConnector(DataConnector):
    """
    CoinGecko API connector using proven pycoingecko library
    Provides market cap, metadata, and price data
    """
    
    def __init__(self, api_key: str = None):
        self.cg = CoinGeckoAPI(api_key=api_key) if api_key else CoinGeckoAPI()
        self.symbol_to_id_map = {}  # Cache for symbol to CoinGecko ID mapping
        self._map_built = False
        
    async def __aenter__(self):
        # Build symbol map on initialization
        if not self._map_built:
            await self._build_symbol_map()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass  # pycoingecko handles session management
    
    async def _build_symbol_map(self):
        """Build mapping from symbol to CoinGecko ID"""
        try:
            # Run in thread to avoid blocking
            import asyncio
            loop = asyncio.get_event_loop()
            
            coins_list = await loop.run_in_executor(None, self.cg.get_coins_list)
            
            for coin in coins_list:
                symbol = coin.get('symbol', '').upper()
                coin_id = coin.get('id')
                if symbol and coin_id:
                    self.symbol_to_id_map[symbol] = coin_id
                        
            self._map_built = True
            logger.info(f"CoinGecko API: Built symbol map with {len(self.symbol_to_id_map)} entries")
                
        except Exception as e:
            logger.error(f"CoinGecko API: Error building symbol map: {e}")
            self._map_built = False
    
    def _symbol_to_id(self, symbol: str) -> Optional[str]:
        """Convert trading symbol to CoinGecko ID"""
        return self.symbol_to_id_map.get(symbol.upper())
    
    async def get_available_symbols(self) -> List[str]:
        """Get available symbols (not the primary source for this)"""
        if not self._map_built:
            await self._build_symbol_map()
        return list(self.symbol_to_id_map.keys())
    
    async def get_ohlcv_data(self, symbol: str, timeframe: str = "1d", limit: int = 100) -> List[MarketDataPoint]:
        """
        Get OHLCV data from CoinGecko (limited in free tier)
        Mainly for fallback when Mercado Bitcoin doesn't have the symbol
        """
        try:
            coin_id = self._symbol_to_id(symbol)
            if not coin_id:
                logger.warning(f"CoinGecko: No ID found for symbol {symbol}")
                return []
            
            # Get historical data (daily only in free tier)
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(
                None,
                self.cg.get_coin_market_chart_by_id,
                coin_id,
                'usd',
                limit
            )
            
            if not data or 'prices' not in data:
                logger.warning(f"CoinGecko: No OHLCV data for {symbol}")
                return []
            
            market_data = []
            prices = data['prices']
            volumes = data.get('total_volumes', [])
            
            # CoinGecko returns [timestamp_ms, price] pairs
            for i, price_point in enumerate(prices):
                try:
                    timestamp = datetime.fromtimestamp(price_point[0] / 1000)
                    price = float(price_point[1])
                    volume = float(volumes[i][1]) if i < len(volumes) else 0.0
                    
                    # CoinGecko doesn't provide OHLC, so we use price as all values
                    market_data.append(MarketDataPoint(
                        symbol=symbol,
                        timestamp=timestamp,
                        open_price=price,
                        high_price=price,
                        low_price=price,
                        close_price=price,
                        volume=volume,
                        quote_volume=volume * price,
                        source='coingecko'
                    ))
                except (IndexError, ValueError, TypeError) as e:
                    logger.warning(f"CoinGecko: Invalid price data for {symbol}: {e}")
                    continue
            
            logger.info(f"CoinGecko: Fetched {len(market_data)} data points for {symbol}")
            return market_data
            
        except Exception as e:
            logger.error(f"CoinGecko: Error fetching OHLCV for {symbol}: {e}")
            return []
    
    async def get_asset_metadata(self, symbol: str) -> AssetMetadata:
        """Fetch comprehensive metadata from CoinGecko"""
        try:
            coin_id = self._symbol_to_id(symbol)
            if not coin_id:
                logger.warning(f"CoinGecko: No ID found for symbol {symbol}")
                return AssetMetadata(
                    symbol=symbol, name=symbol, market_cap_usd=None, market_cap_brl=None,
                    daily_volume_brl=None, listing_date=None, is_active=False,
                    exchange='unknown', sector=None, source='coingecko'
                )
            
            loop = asyncio.get_event_loop()
            
            # Get basic price and market cap data
            price_data = await loop.run_in_executor(
                None,
                self.cg.get_price,
                coin_id,
                'usd,brl',
                True, True, True  # include market_cap, 24h_vol, 24h_change
            )
            
            # Get detailed coin information
            coin_info = await loop.run_in_executor(
                None,
                self.cg.get_coin_by_id,
                coin_id
            )
            
            coin_data = price_data.get(coin_id, {}) if price_data else {}
            
            # Extract BRL exchange rate for conversions
            usd_to_brl = coin_data.get('brl', 0) / coin_data.get('usd', 1) if coin_data.get('usd', 0) > 0 else 5.0
            
            return AssetMetadata(
                symbol=symbol,
                name=coin_info.get('name', symbol) if coin_info else symbol,
                market_cap_usd=coin_data.get('usd_market_cap'),
                market_cap_brl=coin_data.get('brl_market_cap'),
                daily_volume_brl=coin_data.get('brl_24h_vol'),
                listing_date=None,  # Complex to extract from CoinGecko
                is_active=True,
                exchange='multiple',
                sector=coin_info.get('categories', [None])[0] if coin_info and coin_info.get('categories') else None,
                source='coingecko'
            )
            
        except Exception as e:
            logger.error(f"CoinGecko API: Error fetching metadata for {symbol}: {e}")
            return AssetMetadata(
                symbol=symbol, name=symbol, market_cap_usd=None, market_cap_brl=None,
                daily_volume_brl=None, listing_date=None, is_active=False,
                exchange='unknown', sector=None, source='coingecko'
            )


class YahooFinanceConnector(DataConnector):
    """
    Yahoo Finance connector using yfinance library
    Fallback for crypto price data when other sources fail
    """
    
    def __init__(self):
        pass
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
    
    async def get_available_symbols(self) -> List[str]:
        """Yahoo Finance crypto symbols (limited list)"""
        # Common crypto symbols available on Yahoo Finance
        return ['BTC', 'ETH', 'ADA', 'DOT', 'LINK', 'UNI', 'LTC', 'BCH', 'XRP', 'BNB']
    
    async def get_ohlcv_data(self, symbol: str, timeframe: str = "1d", limit: int = 100) -> List[MarketDataPoint]:
        """
        Fetch OHLCV data from Yahoo Finance
        Uses symbol-USD format (e.g., BTC-USD)
        """
        try:
            # Map timeframe to yfinance format
            period_map = {
                '1d': f'{limit}d',
                '1h': f'{min(limit, 30)}d',  # Yahoo Finance limitation
                '5m': f'{min(limit, 7)}d'
            }
            
            period = period_map.get(timeframe, f'{limit}d')
            interval_map = {
                '1d': '1d',
                '1h': '1h', 
                '5m': '5m',
                '1m': '1m'
            }
            interval = interval_map.get(timeframe, '1d')
            
            # Yahoo Finance uses SYMBOL-USD format
            yf_symbol = f"{symbol}-USD"
            
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            ticker = await loop.run_in_executor(None, yf.Ticker, yf_symbol)
            hist = await loop.run_in_executor(None, ticker.history, period, interval)
            
            if hist.empty:
                logger.warning(f"Yahoo Finance: No data for {symbol}")
                return []
            
            market_data = []
            for timestamp, row in hist.iterrows():
                try:
                    market_data.append(MarketDataPoint(
                        symbol=symbol,
                        timestamp=timestamp.to_pydatetime(),
                        open_price=float(row['Open']),
                        high_price=float(row['High']),
                        low_price=float(row['Low']),
                        close_price=float(row['Close']),
                        volume=float(row['Volume']),
                        quote_volume=float(row['Volume']) * float(row['Close']),
                        source='yahoo_finance'
                    ))
                except (ValueError, TypeError) as e:
                    logger.warning(f"Yahoo Finance: Invalid data point for {symbol}: {e}")
                    continue
            
            logger.info(f"Yahoo Finance: Fetched {len(market_data)} data points for {symbol}")
            return market_data
            
        except Exception as e:
            logger.error(f"Yahoo Finance: Error fetching OHLCV for {symbol}: {e}")
            return []
    
    async def get_asset_metadata(self, symbol: str) -> AssetMetadata:
        """Limited metadata from Yahoo Finance"""
        try:
            yf_symbol = f"{symbol}-USD"
            
            loop = asyncio.get_event_loop()
            ticker = await loop.run_in_executor(None, yf.Ticker, yf_symbol)
            info = await loop.run_in_executor(None, getattr, ticker, 'info')
            
            return AssetMetadata(
                symbol=symbol,
                name=info.get('shortName', symbol) if info else symbol,
                market_cap_usd=info.get('marketCap') if info else None,
                market_cap_brl=None,  # Not available
                daily_volume_brl=None,  # Not available
                listing_date=None,  # Not available
                is_active=True,
                exchange='multiple',
                sector='Cryptocurrency',
                source='yahoo_finance'
            )
            
        except Exception as e:
            logger.error(f"Yahoo Finance: Error fetching metadata for {symbol}: {e}")
            return AssetMetadata(
                symbol=symbol, name=symbol, market_cap_usd=None, market_cap_brl=None,
                daily_volume_brl=None, listing_date=None, is_active=False,
                exchange='unknown', sector=None, source='yahoo_finance'
            )


class MarketDataCollector:
    """
    Primary market data collection orchestrator
    Uses proven data sources with intelligent fallback:
    1. Mercado Bitcoin (Brazilian exchange - primary for symbols/OHLCV)
    2. CoinGecko (metadata, market caps, fallback prices)
    3. Yahoo Finance (fallback OHLCV data)
    """
    
    def __init__(self, cg_api_key: str = None):
        self.connectors = {
            'mercado_bitcoin': MercadoBitcoinConnector(),
            'coingecko': CoinGeckoConnector(cg_api_key),
            'yahoo_finance': YahooFinanceConnector()
        }
        self.primary_exchange = 'mercado_bitcoin'
        self.metadata_source = 'coingecko'
        self.fallback_sources = ['coingecko', 'yahoo_finance']
        
    async def __aenter__(self):
        for connector in self.connectors.values():
            await connector.__aenter__()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        for connector in self.connectors.values():
            await connector.__aexit__(exc_type, exc_val, exc_tb)
    
    async def get_available_assets(self) -> List[str]:
        """Get universe of available assets from primary exchange"""
        try:
            symbols = await self.connectors[self.primary_exchange].get_available_symbols()
            logger.info(f"Market Data Collector: Found {len(symbols)} available assets")
            return symbols
            
        except Exception as e:
            logger.error(f"Market Data Collector: Error fetching available assets: {e}")
            return []
    
    async def collect_ohlcv_data(self, symbols: List[str], timeframe: str = "1d", 
                                limit: int = 100) -> Dict[str, List[MarketDataPoint]]:
        """
        Collect OHLCV data for multiple symbols with intelligent fallback
        
        Args:
            symbols: List of trading symbols
            timeframe: Data timeframe ('1m', '5m', '1h', '1d')
            limit: Number of data points per symbol
            
        Returns:
            Dictionary mapping symbols to their OHLCV data
        """
        logger.info(f"Collecting OHLCV data for {len(symbols)} symbols, timeframe={timeframe}")
        
        ohlcv_data = {}
        
        # Process symbols in batches to respect rate limits
        batch_size = 3  # Conservative batch size
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            
            # Try each symbol in the batch with fallback logic
            for symbol in batch:
                ohlcv_data[symbol] = await self._collect_single_ohlcv_with_fallback(
                    symbol, timeframe, limit
                )
                    
            logger.debug(f"Completed batch {i//batch_size + 1}/{(len(symbols)-1)//batch_size + 1}")
        
        successful_collections = sum(1 for data in ohlcv_data.values() if data)
        logger.info(f"OHLCV collection completed: {successful_collections}/{len(symbols)} symbols successful")
        
        return ohlcv_data
    
    async def _collect_single_ohlcv_with_fallback(self, symbol: str, timeframe: str, 
                                                 limit: int) -> List[MarketDataPoint]:
        """Collect OHLCV for single symbol with fallback sources"""
        
        # Try primary source first (Mercado Bitcoin)
        try:
            primary_connector = self.connectors[self.primary_exchange]
            data = await primary_connector.get_ohlcv_data(symbol, timeframe, limit)
            if data:
                logger.debug(f"Got {symbol} data from {self.primary_exchange}")
                return data
        except Exception as e:
            logger.warning(f"Primary source failed for {symbol}: {e}")
        
        # Try fallback sources
        for fallback_source in self.fallback_sources:
            try:
                fallback_connector = self.connectors[fallback_source]
                data = await fallback_connector.get_ohlcv_data(symbol, timeframe, limit)
                if data:
                    logger.info(f"Got {symbol} data from fallback source: {fallback_source}")
                    return data
            except Exception as e:
                logger.warning(f"Fallback source {fallback_source} failed for {symbol}: {e}")
        
        logger.error(f"All data sources failed for symbol: {symbol}")
        return []
    
    async def collect_metadata(self, symbols: List[str]) -> Dict[str, AssetMetadata]:
        """
        Collect comprehensive metadata for symbols
        Combines data from multiple sources for complete picture
        """
        logger.info(f"Collecting metadata for {len(symbols)} symbols")
        
        metadata = {}
        
        # Get basic metadata from primary exchange
        mb_connector = self.connectors[self.primary_exchange]
        cg_connector = self.connectors[self.metadata_source]
        
        # Collect from both sources
        tasks_mb = [mb_connector.get_asset_metadata(symbol) for symbol in symbols]
        tasks_cg = [cg_connector.get_asset_metadata(symbol) for symbol in symbols]
        
        mb_results = await asyncio.gather(*tasks_mb, return_exceptions=True)
        cg_results = await asyncio.gather(*tasks_cg, return_exceptions=True)
        
        # Merge metadata from multiple sources
        for i, symbol in enumerate(symbols):
            mb_meta = mb_results[i] if not isinstance(mb_results[i], Exception) else None
            cg_meta = cg_results[i] if not isinstance(cg_results[i], Exception) else None
            
            # Create combined metadata, preferring CoinGecko for fundamental data
            if cg_meta and cg_meta.market_cap_usd:
                # Use CoinGecko as primary if it has market cap data
                metadata[symbol] = AssetMetadata(
                    symbol=symbol,
                    name=cg_meta.name,
                    market_cap_usd=cg_meta.market_cap_usd,
                    market_cap_brl=cg_meta.market_cap_brl,
                    daily_volume_brl=mb_meta.daily_volume_brl if mb_meta else cg_meta.daily_volume_brl,
                    listing_date=cg_meta.listing_date,
                    is_active=mb_meta.is_active if mb_meta else cg_meta.is_active,
                    exchange='mercado_bitcoin',
                    sector=cg_meta.sector,
                    source='combined'
                )
            elif mb_meta:
                # Fallback to Mercado Bitcoin data
                metadata[symbol] = mb_meta
            else:
                # Create minimal metadata if both sources failed
                metadata[symbol] = AssetMetadata(
                    symbol=symbol, name=symbol, market_cap_usd=None, market_cap_brl=None,
                    daily_volume_brl=None, listing_date=None, is_active=False,
                    exchange='unknown', sector=None, source='fallback'
                )
        
        successful_meta = sum(1 for meta in metadata.values() if meta.is_active)
        logger.info(f"Metadata collection completed: {successful_meta}/{len(symbols)} symbols with data")
        
        return metadata
    
    async def collect_complete_dataset(self, timeframe: str = "1d", 
                                     lookback_days: int = 90) -> Tuple[Dict[str, List[MarketDataPoint]], Dict[str, AssetMetadata]]:
        """
        Collect complete dataset (OHLCV + metadata) for all available assets
        This is the main entry point for comprehensive data collection
        """
        logger.info("=" * 50)
        logger.info("Starting Complete Dataset Collection")
        logger.info("=" * 50)
        
        try:
            # Step 1: Get available assets
            symbols = await self.get_available_assets()
            if not symbols:
                raise ValueError("No assets available for collection")
            
            # Step 2: Calculate data points needed based on lookback
            timeframe_to_limit = {
                "1m": lookback_days * 24 * 60,
                "5m": lookback_days * 24 * 12,
                "1h": lookback_days * 24,
                "1d": lookback_days
            }
            limit = timeframe_to_limit.get(timeframe, lookback_days)
            
            # Step 3: Collect OHLCV and metadata in parallel
            logger.info(f"Collecting data for {len(symbols)} symbols, {lookback_days} days lookback")
            
            ohlcv_task = self.collect_ohlcv_data(symbols, timeframe, limit)
            metadata_task = self.collect_metadata(symbols)
            
            ohlcv_data, metadata = await asyncio.gather(ohlcv_task, metadata_task)
            
            # Step 4: Filter out assets with insufficient data
            filtered_symbols = []
            min_data_points = max(30, lookback_days // 3)  # At least 30 points or 1/3 of requested
            
            for symbol in symbols:
                if (symbol in ohlcv_data and len(ohlcv_data[symbol]) >= min_data_points and
                    symbol in metadata and metadata[symbol].is_active):
                    filtered_symbols.append(symbol)
            
            # Filter datasets
            filtered_ohlcv = {s: ohlcv_data[s] for s in filtered_symbols}
            filtered_metadata = {s: metadata[s] for s in filtered_symbols}
            
            logger.info("=" * 50)
            logger.info("Complete Dataset Collection Summary:")
            logger.info(f"• Total assets discovered: {len(symbols)}")
            logger.info(f"• Assets with sufficient data: {len(filtered_symbols)}")
            logger.info(f"• Data timeframe: {timeframe}")
            logger.info(f"• Lookback period: {lookback_days} days")
            logger.info(f"• Average data points per asset: {np.mean([len(data) for data in filtered_ohlcv.values()]):.1f}")
            logger.info("=" * 50)
            
            return filtered_ohlcv, filtered_metadata
            
        except Exception as e:
            logger.error(f"Complete dataset collection failed: {e}")
            raise


# Utility functions for working with collected data

def ohlcv_to_dataframe(ohlcv_data: Dict[str, List[MarketDataPoint]]) -> pd.DataFrame:
    """
    Convert OHLCV data to pandas DataFrame format
    
    Returns:
        DataFrame with MultiIndex columns (symbol, ohlcv_field)
    """
    if not ohlcv_data:
        return pd.DataFrame()
    
    dfs = []
    
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
                'volume': point.volume
            }
            for point in data_points
        ])
        
        df.set_index('timestamp', inplace=True)
        df.columns = [(symbol, col) for col in df.columns]
        dfs.append(df)
    
    if dfs:
        combined_df = pd.concat(dfs, axis=1, sort=True)
        combined_df.sort_index(inplace=True)
        return combined_df
    else:
        return pd.DataFrame()


def metadata_to_dataframe(metadata: Dict[str, AssetMetadata]) -> pd.DataFrame:
    """Convert metadata to pandas DataFrame"""
    if not metadata:
        return pd.DataFrame()
    
    records = []
    for symbol, meta in metadata.items():
        records.append({
            'symbol': meta.symbol,
            'name': meta.name,
            'market_cap_usd': meta.market_cap_usd,
            'market_cap_brl': meta.market_cap_brl,
            'daily_volume_brl': meta.daily_volume_brl,
            'listing_date': meta.listing_date,
            'is_active': meta.is_active,
            'exchange': meta.exchange,
            'sector': meta.sector,
            'source': meta.source
        })
    
    df = pd.DataFrame(records)
    df.set_index('symbol', inplace=True)
    return df
