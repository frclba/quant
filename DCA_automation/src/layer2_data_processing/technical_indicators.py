#!/usr/bin/env python3
"""
Module 2.1: Technical Indicators Engine

Comprehensive technical analysis engine implementing indicators from PFS Theory Notebook 3.
Calculates trend, momentum, volatility, and volume indicators for cryptocurrency analysis.

Features:
- Trend Indicators: SMA, EMA, MACD, ADX
- Momentum Indicators: RSI, Stochastic, Williams %R
- Volatility Indicators: Bollinger Bands, ATR, Volatility Cones
- Volume Indicators: VWAP, OBV, A/D Line
- Production-optimized with vectorized calculations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class IndicatorConfig:
    """Configuration for technical indicators"""
    
    # Trend Indicators
    sma_windows: List[int] = None
    ema_windows: List[int] = None
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    adx_window: int = 14
    
    # Momentum Indicators
    rsi_window: int = 14
    stoch_k_window: int = 14
    stoch_d_window: int = 3
    williams_r_window: int = 14
    
    # Volatility Indicators
    bb_window: int = 20
    bb_std_dev: float = 2.0
    atr_window: int = 14
    vol_cone_windows: List[int] = None
    
    # Volume Indicators
    vwap_window: int = 20
    
    def __post_init__(self):
        """Set default values for list parameters"""
        if self.sma_windows is None:
            self.sma_windows = [20, 50, 200]
        if self.ema_windows is None:
            self.ema_windows = [12, 26, 50]
        if self.vol_cone_windows is None:
            self.vol_cone_windows = [30, 60, 90, 180]


class TechnicalIndicatorsEngine:
    """
    Production-ready technical indicators calculation engine
    
    Based on PFS Theory Notebook 3 concepts with optimized implementations
    for real-time cryptocurrency analysis.
    """
    
    def __init__(self, config: Optional[IndicatorConfig] = None):
        self.config = config or IndicatorConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators for a given price dataset
        
        Args:
            df: DataFrame with OHLCV data (columns: open, high, low, close, volume)
            
        Returns:
            DataFrame with original data plus all calculated indicators
        """
        if df is None or df.empty:
            raise ValueError("Input DataFrame is empty")
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        self.logger.info(f"Calculating technical indicators for {len(df)} data points")
        
        # Create a copy to avoid modifying original data
        result_df = df.copy()
        
        # Calculate all indicator categories
        result_df = self._calculate_trend_indicators(result_df)
        result_df = self._calculate_momentum_indicators(result_df)
        result_df = self._calculate_volatility_indicators(result_df)
        result_df = self._calculate_volume_indicators(result_df)
        
        self.logger.info(f"Successfully calculated {len(result_df.columns) - len(df.columns)} technical indicators")
        return result_df
    
    def _calculate_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend-following indicators"""
        
        # Simple Moving Averages
        for window in self.config.sma_windows:
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
        
        # Exponential Moving Averages
        for window in self.config.ema_windows:
            df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
        
        # MACD (Moving Average Convergence Divergence)
        ema_fast = df['close'].ewm(span=self.config.macd_fast).mean()
        ema_slow = df['close'].ewm(span=self.config.macd_slow).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=self.config.macd_signal).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Average Directional Index (ADX)
        df = self._calculate_adx(df)
        
        # Trend Signals (binary indicators)
        df['trend_sma_bull'] = (df['close'] > df[f'sma_{self.config.sma_windows[0]}']).astype(int)
        df['trend_ema_bull'] = (df[f'ema_{self.config.ema_windows[0]}'] > df[f'ema_{self.config.ema_windows[1]}']).astype(int)
        df['trend_macd_bull'] = (df['macd'] > df['macd_signal']).astype(int)
        
        return df
    
    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum oscillators"""
        
        # RSI (Relative Strength Index)
        df['rsi'] = self._calculate_rsi(df['close'], self.config.rsi_window)
        
        # Stochastic Oscillator
        stoch_k, stoch_d = self._calculate_stochastic(df)
        df['stoch_k'] = stoch_k
        df['stoch_d'] = stoch_d
        
        # Williams %R
        df['williams_r'] = self._calculate_williams_r(df)
        
        # Rate of Change (ROC)
        df['roc_10'] = df['close'].pct_change(10) * 100
        df['roc_30'] = df['close'].pct_change(30) * 100
        
        # Momentum Signals
        df['momentum_rsi_oversold'] = (df['rsi'] < 30).astype(int)
        df['momentum_rsi_overbought'] = (df['rsi'] > 70).astype(int)
        df['momentum_stoch_oversold'] = ((df['stoch_k'] < 20) & (df['stoch_d'] < 20)).astype(int)
        df['momentum_stoch_overbought'] = ((df['stoch_k'] > 80) & (df['stoch_d'] > 80)).astype(int)
        
        return df
    
    def _calculate_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility-based indicators"""
        
        # Bollinger Bands
        bb_middle = df['close'].rolling(window=self.config.bb_window).mean()
        bb_std = df['close'].rolling(window=self.config.bb_window).std()
        df['bb_upper'] = bb_middle + (bb_std * self.config.bb_std_dev)
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_middle - (bb_std * self.config.bb_std_dev)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Average True Range (ATR)
        df['atr'] = self._calculate_atr(df)
        
        # Volatility Cones
        for window in self.config.vol_cone_windows:
            returns = df['close'].pct_change().rolling(window=window)
            df[f'volatility_{window}d'] = returns.std() * np.sqrt(252)  # Annualized volatility
        
        # Keltner Channels (ATR-based bands)
        keltner_middle = df[f'ema_{self.config.ema_windows[0]}']
        df['keltner_upper'] = keltner_middle + (2 * df['atr'])
        df['keltner_lower'] = keltner_middle - (2 * df['atr'])
        
        # Volatility Signals
        df['vol_bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(20).quantile(0.2)).astype(int)
        df['vol_bb_breakout'] = (df['close'] > df['bb_upper']).astype(int)
        df['vol_bb_breakdown'] = (df['close'] < df['bb_lower']).astype(int)
        
        return df
    
    def _calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based indicators"""
        
        # Volume Weighted Average Price (VWAP)
        df['vwap'] = self._calculate_vwap(df)
        
        # On Balance Volume (OBV)
        df['obv'] = self._calculate_obv(df)
        
        # Accumulation/Distribution Line (A/D Line)
        df['ad_line'] = self._calculate_ad_line(df)
        
        # Volume Rate of Change
        df['volume_roc'] = df['volume'].pct_change(5) * 100
        
        # Volume Moving Average
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # Volume Signals
        df['volume_surge'] = (df['volume_ratio'] > 2.0).astype(int)
        df['volume_above_average'] = (df['volume'] > df['volume_sma_20']).astype(int)
        
        return df
    
    # Helper methods for complex indicator calculations
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_stochastic(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator (%K and %D)"""
        low_min = df['low'].rolling(window=self.config.stoch_k_window).min()
        high_max = df['high'].rolling(window=self.config.stoch_k_window).max()
        
        stoch_k = 100 * ((df['close'] - low_min) / (high_max - low_min))
        stoch_d = stoch_k.rolling(window=self.config.stoch_d_window).mean()
        
        return stoch_k, stoch_d
    
    def _calculate_williams_r(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Williams %R"""
        high_max = df['high'].rolling(window=self.config.williams_r_window).max()
        low_min = df['low'].rolling(window=self.config.williams_r_window).min()
        
        williams_r = -100 * ((high_max - df['close']) / (high_max - low_min))
        return williams_r
    
    def _calculate_adx(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Average Directional Index (ADX)"""
        window = self.config.adx_window
        
        # True Range
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        
        # Directional Movement
        df['dm_plus'] = np.where(
            (df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
            np.maximum(df['high'] - df['high'].shift(1), 0),
            0
        )
        
        df['dm_minus'] = np.where(
            (df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
            np.maximum(df['low'].shift(1) - df['low'], 0),
            0
        )
        
        # Smoothed values
        df['atr_adx'] = df['tr'].rolling(window=window).mean()
        df['di_plus'] = 100 * (df['dm_plus'].rolling(window=window).mean() / df['atr_adx'])
        df['di_minus'] = 100 * (df['dm_minus'].rolling(window=window).mean() / df['atr_adx'])
        
        # ADX calculation
        df['dx'] = 100 * abs(df['di_plus'] - df['di_minus']) / (df['di_plus'] + df['di_minus'])
        df['adx'] = df['dx'].rolling(window=window).mean()
        
        # Clean up temporary columns
        df.drop(['tr', 'dm_plus', 'dm_minus', 'atr_adx', 'dx'], axis=1, inplace=True)
        
        return df
    
    def _calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close_prev = abs(df['high'] - df['close'].shift(1))
        low_close_prev = abs(df['low'] - df['close'].shift(1))
        
        true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        atr = true_range.rolling(window=self.config.atr_window).mean()
        
        return atr
    
    def _calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).rolling(window=self.config.vwap_window).sum() / \
               df['volume'].rolling(window=self.config.vwap_window).sum()
        return vwap
    
    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On Balance Volume"""
        obv = pd.Series(index=df.index, dtype=float)
        obv.iloc[0] = df['volume'].iloc[0]
        
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def _calculate_ad_line(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Accumulation/Distribution Line"""
        money_flow_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        money_flow_volume = money_flow_multiplier * df['volume']
        ad_line = money_flow_volume.cumsum()
        return ad_line
    
    def get_indicator_summary(self, df: pd.DataFrame) -> Dict[str, Dict[str, Union[float, int]]]:
        """
        Get summary statistics for all calculated indicators
        
        Returns:
            Dictionary with indicator categories and their latest values
        """
        if df.empty:
            return {}
        
        latest = df.iloc[-1]
        
        summary = {
            'trend': {
                'sma_20': latest.get('sma_20', np.nan),
                'ema_12': latest.get('ema_12', np.nan),
                'macd': latest.get('macd', np.nan),
                'macd_signal': latest.get('macd_signal', np.nan),
                'adx': latest.get('adx', np.nan),
                'trend_strength': 'strong' if latest.get('adx', 0) > 25 else 'weak'
            },
            'momentum': {
                'rsi': latest.get('rsi', np.nan),
                'stoch_k': latest.get('stoch_k', np.nan),
                'williams_r': latest.get('williams_r', np.nan),
                'momentum_regime': self._classify_momentum_regime(latest)
            },
            'volatility': {
                'bb_position': latest.get('bb_position', np.nan),
                'bb_width': latest.get('bb_width', np.nan),
                'atr': latest.get('atr', np.nan),
                'volatility_regime': self._classify_volatility_regime(latest)
            },
            'volume': {
                'volume_ratio': latest.get('volume_ratio', np.nan),
                'obv': latest.get('obv', np.nan),
                'volume_trend': 'increasing' if latest.get('volume_ratio', 1) > 1.2 else 'normal'
            }
        }
        
        return summary
    
    def _classify_momentum_regime(self, latest_data: pd.Series) -> str:
        """Classify momentum regime based on oscillator values"""
        rsi = latest_data.get('rsi', 50)
        stoch_k = latest_data.get('stoch_k', 50)
        
        if rsi > 70 or stoch_k > 80:
            return 'overbought'
        elif rsi < 30 or stoch_k < 20:
            return 'oversold'
        else:
            return 'neutral'
    
    def _classify_volatility_regime(self, latest_data: pd.Series) -> str:
        """Classify volatility regime based on band width and position"""
        bb_width = latest_data.get('bb_width', 0)
        vol_30d = latest_data.get('volatility_30d', 0)
        
        if bb_width < 0.1:  # Bands are tight
            return 'low_volatility'
        elif vol_30d > 0.8:  # High annualized volatility
            return 'high_volatility'
        else:
            return 'normal_volatility'


# Technical Indicators Engine is integrated into the main DCA system
# Use the main CLI for testing: python cli.py run
