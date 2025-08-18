#!/usr/bin/env python3
"""
Multi-Factor Asset Scoring Engine

Combines technical, fundamental, and risk factors into composite asset scores.
Implements the scoring methodology from the DCA architecture.

Primary Factors (70% weight):
- Momentum Score (40%): Multi-timeframe trend analysis
- Mean Reversion Score (20%): Oversold/overbought conditions  
- Volume/Liquidity Score (10%): Trading activity analysis

Secondary Factors (20% weight):
- Volatility Regime (10%): Risk-adjusted scoring
- Correlation Dynamics (10%): Diversification benefits

Meta Factors (10% weight):
- Market Cycle Position: BTC halving, altcoin season
- Brazilian Market Factors: BRL/USD, local liquidity
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ScoreComponent(Enum):
    """Scoring component types"""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion" 
    VOLUME_LIQUIDITY = "volume_liquidity"
    VOLATILITY_REGIME = "volatility_regime"
    CORRELATION_DYNAMICS = "correlation_dynamics"
    MARKET_CYCLE = "market_cycle"
    BRAZILIAN_FACTORS = "brazilian_factors"


@dataclass
class ScoringConfig:
    """Configuration for multi-factor scoring"""
    
    # Primary factor weights (must sum to 0.70)
    momentum_weight: float = 0.40
    mean_reversion_weight: float = 0.20
    volume_liquidity_weight: float = 0.10
    
    # Secondary factor weights (must sum to 0.20)
    volatility_regime_weight: float = 0.10
    correlation_dynamics_weight: float = 0.10
    
    # Meta factor weights (must sum to 0.10)
    market_cycle_weight: float = 0.05
    brazilian_factors_weight: float = 0.05
    
    # Momentum parameters
    momentum_windows: List[int] = field(default_factory=lambda: [21, 63, 126])  # 1M, 3M, 6M
    momentum_decay_factor: float = 0.94  # Exponential decay
    
    # Mean reversion parameters
    rsi_window: int = 14
    bollinger_window: int = 20
    bollinger_std: float = 2.0
    
    # Volume parameters
    volume_window: int = 30
    volume_threshold: float = 2.0  # Abnormal volume threshold (2σ)
    
    # Volatility parameters
    volatility_window: int = 30
    volatility_regime_threshold: float = 1.5  # High volatility threshold
    
    # Correlation parameters
    correlation_window: int = 60
    max_correlation_penalty: float = 0.5  # Penalty for high correlation
    
    # Score normalization
    min_score: float = 0.0
    max_score: float = 1.0
    score_decay_halflife: int = 5  # Days for score decay


@dataclass
class AssetScore:
    """Complete asset scoring result"""
    symbol: str
    composite_score: float
    component_scores: Dict[str, float]
    percentile_rank: Optional[float] = None
    score_timestamp: datetime = field(default_factory=datetime.now)
    
    # Detailed breakdowns
    momentum_breakdown: Dict[str, float] = field(default_factory=dict)
    mean_reversion_breakdown: Dict[str, float] = field(default_factory=dict)
    risk_metrics: Dict[str, float] = field(default_factory=dict)


class MultiFactorScoringEngine:
    """
    Multi-Factor Asset Scoring Engine
    
    Generates composite scores combining multiple quantitative factors
    for cryptocurrency asset ranking and selection.
    """
    
    def __init__(self, config: Optional[ScoringConfig] = None):
        self.config = config or ScoringConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Validate weights
        self._validate_config()
        
        # Cached calculations
        self._score_cache = {}
        self._benchmark_data = None
        
        self.logger.info(f"🎯 MultiFactorScoringEngine initialized")
        self.logger.info(f"  Primary weights: M={self.config.momentum_weight}, MR={self.config.mean_reversion_weight}, VL={self.config.volume_liquidity_weight}")
        self.logger.info(f"  Secondary weights: VR={self.config.volatility_regime_weight}, CD={self.config.correlation_dynamics_weight}")
    
    def _validate_config(self):
        """Validate scoring configuration"""
        primary_sum = (self.config.momentum_weight + 
                      self.config.mean_reversion_weight + 
                      self.config.volume_liquidity_weight)
        
        secondary_sum = (self.config.volatility_regime_weight + 
                        self.config.correlation_dynamics_weight)
        
        meta_sum = (self.config.market_cycle_weight + 
                   self.config.brazilian_factors_weight)
        
        total_weight = primary_sum + secondary_sum + meta_sum
        
        if not (0.98 <= total_weight <= 1.02):  # Allow small floating point errors
            raise ValueError(f"Factor weights must sum to 1.0, got {total_weight}")
        
        self.logger.debug(f"Weight validation: Primary={primary_sum:.3f}, Secondary={secondary_sum:.3f}, Meta={meta_sum:.3f}")
    
    def calculate_asset_scores(self, 
                             market_data: Dict[str, pd.DataFrame],
                             benchmark_symbol: str = 'BTC') -> Dict[str, AssetScore]:
        """
        Calculate composite scores for all assets
        
        Args:
            market_data: Dictionary of OHLCV DataFrames by symbol
            benchmark_symbol: Symbol to use for relative calculations
            
        Returns:
            Dictionary of AssetScore objects by symbol
        """
        self.logger.info(f"🔄 Calculating scores for {len(market_data)} assets...")
        
        # Set benchmark data
        self._benchmark_data = market_data.get(benchmark_symbol)
        if self._benchmark_data is None:
            self.logger.warning(f"Benchmark {benchmark_symbol} not found, using first asset")
            self._benchmark_data = list(market_data.values())[0]
        
        scores = {}
        score_components = {}
        
        # Calculate individual component scores
        for symbol, data in market_data.items():
            if len(data) < max(self.config.momentum_windows) + 30:  # Need sufficient data
                self.logger.warning(f"Insufficient data for {symbol}, skipping")
                continue
            
            try:
                # Calculate all scoring components
                momentum_score = self._calculate_momentum_score(data, symbol)
                mean_reversion_score = self._calculate_mean_reversion_score(data, symbol)
                volume_score = self._calculate_volume_liquidity_score(data, symbol)
                volatility_score = self._calculate_volatility_regime_score(data, symbol)
                correlation_score = self._calculate_correlation_score(data, symbol, market_data)
                cycle_score = self._calculate_market_cycle_score(data, symbol)
                brazilian_score = self._calculate_brazilian_factors_score(data, symbol)
                
                # Store component scores
                components = {
                    ScoreComponent.MOMENTUM.value: momentum_score,
                    ScoreComponent.MEAN_REVERSION.value: mean_reversion_score,
                    ScoreComponent.VOLUME_LIQUIDITY.value: volume_score,
                    ScoreComponent.VOLATILITY_REGIME.value: volatility_score,
                    ScoreComponent.CORRELATION_DYNAMICS.value: correlation_score,
                    ScoreComponent.MARKET_CYCLE.value: cycle_score,
                    ScoreComponent.BRAZILIAN_FACTORS.value: brazilian_score
                }
                
                # Calculate composite score
                composite_score = (
                    momentum_score * self.config.momentum_weight +
                    mean_reversion_score * self.config.mean_reversion_weight +
                    volume_score * self.config.volume_liquidity_weight +
                    volatility_score * self.config.volatility_regime_weight +
                    correlation_score * self.config.correlation_dynamics_weight +
                    cycle_score * self.config.market_cycle_weight +
                    brazilian_score * self.config.brazilian_factors_weight
                )
                
                # Create asset score object
                asset_score = AssetScore(
                    symbol=symbol,
                    composite_score=np.clip(composite_score, self.config.min_score, self.config.max_score),
                    component_scores=components,
                    momentum_breakdown=self._get_momentum_breakdown(data),
                    mean_reversion_breakdown=self._get_mean_reversion_breakdown(data),
                    risk_metrics=self._get_risk_metrics(data)
                )
                
                scores[symbol] = asset_score
                score_components[symbol] = components
                
                self.logger.debug(f"  {symbol}: Composite={composite_score:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error calculating scores for {symbol}: {e}")
                continue
        
        # Calculate percentile ranks
        composite_scores = [score.composite_score for score in scores.values()]
        if composite_scores:
            for symbol, score in scores.items():
                percentile = (np.sum(np.array(composite_scores) <= score.composite_score) / len(composite_scores)) * 100
                score.percentile_rank = percentile
        
        self.logger.info(f"✅ Calculated scores for {len(scores)} assets")
        return scores
    
    def _calculate_momentum_score(self, data: pd.DataFrame, symbol: str) -> float:
        """Calculate momentum score using multiple timeframes"""
        
        # Calculate returns for different periods
        momentum_scores = []
        
        for window in self.config.momentum_windows:
            if len(data) < window:
                continue
                
            # Simple momentum (price change)
            period_return = (data['close'].iloc[-1] / data['close'].iloc[-window]) - 1
            
            # Risk-adjusted momentum (return / volatility)
            returns = data['close'].pct_change(window).dropna()
            if len(returns) > 0:
                volatility = returns.std() * np.sqrt(252)  # Annualized
                risk_adj_momentum = period_return / max(volatility, 0.01)  # Avoid division by zero
                momentum_scores.append(risk_adj_momentum)
        
        if not momentum_scores:
            return 0.5  # Neutral score
        
        # Weight by decay factor (more recent = higher weight)
        weights = [self.config.momentum_decay_factor ** i for i in range(len(momentum_scores))]
        weighted_score = np.average(momentum_scores, weights=weights)
        
        # Normalize to [0, 1] using tanh transformation
        normalized_score = (np.tanh(weighted_score) + 1) / 2
        
        return np.clip(normalized_score, 0.0, 1.0)
    
    def _calculate_mean_reversion_score(self, data: pd.DataFrame, symbol: str) -> float:
        """Calculate mean reversion score using RSI and Bollinger Bands"""
        
        # RSI calculation
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=self.config.rsi_window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.config.rsi_window).mean()
        rs = gain / loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        
        # RSI mean reversion signal (inverted - low RSI = high score)
        current_rsi = rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else 50
        rsi_score = (100 - current_rsi) / 100  # Invert so oversold = high score
        
        # Bollinger Bands
        bb_window = self.config.bollinger_window
        bb_std = self.config.bollinger_std
        
        sma = data['close'].rolling(window=bb_window).mean()
        std = data['close'].rolling(window=bb_window).std()
        upper_band = sma + (std * bb_std)
        lower_band = sma - (std * bb_std)
        
        # Bollinger position (0 = at lower band, 1 = at upper band)
        current_price = data['close'].iloc[-1]
        current_upper = upper_band.iloc[-1]
        current_lower = lower_band.iloc[-1]
        
        if not (np.isnan(current_upper) or np.isnan(current_lower)):
            bb_position = (current_price - current_lower) / (current_upper - current_lower)
            bb_score = 1 - bb_position  # Invert so near lower band = high score
        else:
            bb_score = 0.5
        
        # Combine RSI and Bollinger signals
        mean_reversion_score = (rsi_score * 0.6 + bb_score * 0.4)
        
        return np.clip(mean_reversion_score, 0.0, 1.0)
    
    def _calculate_volume_liquidity_score(self, data: pd.DataFrame, symbol: str) -> float:
        """Calculate volume and liquidity score"""
        
        # Volume analysis
        volume_sma = data['volume'].rolling(window=self.config.volume_window).mean()
        volume_std = data['volume'].rolling(window=self.config.volume_window).std()
        
        # Recent average volume
        recent_volume = data['volume'].tail(5).mean()
        baseline_volume = volume_sma.iloc[-1]
        volume_volatility = volume_std.iloc[-1]
        
        # Abnormal volume detection
        if volume_volatility > 0:
            volume_z_score = (recent_volume - baseline_volume) / volume_volatility
            abnormal_volume_score = min(abs(volume_z_score) / self.config.volume_threshold, 1.0)
        else:
            abnormal_volume_score = 0.0
        
        # Liquidity proxy (higher volume = better liquidity)
        liquidity_score = min(np.log1p(recent_volume) / np.log1p(baseline_volume * 10), 1.0)
        
        # Combine volume signals
        volume_liquidity_score = (abnormal_volume_score * 0.4 + liquidity_score * 0.6)
        
        return np.clip(volume_liquidity_score, 0.0, 1.0)
    
    def _calculate_volatility_regime_score(self, data: pd.DataFrame, symbol: str) -> float:
        """Calculate volatility regime score"""
        
        # Calculate rolling volatility
        returns = data['close'].pct_change().dropna()
        rolling_vol = returns.rolling(window=self.config.volatility_window).std() * np.sqrt(252)
        
        current_vol = rolling_vol.iloc[-1]
        historical_vol = rolling_vol.median()
        
        # Volatility regime classification
        vol_ratio = current_vol / historical_vol if historical_vol > 0 else 1.0
        
        if vol_ratio > self.config.volatility_regime_threshold:
            # High volatility regime - lower score (higher risk)
            volatility_score = 1 / (1 + vol_ratio - self.config.volatility_regime_threshold)
        else:
            # Normal/low volatility regime - higher score
            volatility_score = 0.5 + 0.5 * (self.config.volatility_regime_threshold - vol_ratio) / self.config.volatility_regime_threshold
        
        return np.clip(volatility_score, 0.0, 1.0)
    
    def _calculate_correlation_score(self, data: pd.DataFrame, symbol: str, market_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate correlation dynamics score"""
        
        if len(market_data) < 2:
            return 0.5  # Neutral if no other assets to compare
        
        # Calculate returns for current asset
        asset_returns = data['close'].pct_change().dropna().tail(self.config.correlation_window)
        
        correlations = []
        for other_symbol, other_data in market_data.items():
            if other_symbol == symbol:
                continue
                
            # Calculate returns for other asset
            other_returns = other_data['close'].pct_change().dropna().tail(self.config.correlation_window)
            
            # Align data
            aligned_data = pd.concat([asset_returns, other_returns], axis=1, join='inner')
            if len(aligned_data) < 20:  # Need minimum data for correlation
                continue
                
            correlation = aligned_data.iloc[:, 0].corr(aligned_data.iloc[:, 1])
            if not np.isnan(correlation):
                correlations.append(abs(correlation))
        
        if not correlations:
            return 0.5
        
        # Average absolute correlation
        avg_correlation = np.mean(correlations)
        
        # Lower correlation = higher diversification score
        diversification_score = 1 - min(avg_correlation, self.config.max_correlation_penalty)
        
        return np.clip(diversification_score, 0.0, 1.0)
    
    def _calculate_market_cycle_score(self, data: pd.DataFrame, symbol: str) -> float:
        """Calculate market cycle position score"""
        
        # Simple implementation: relative performance vs benchmark
        if self._benchmark_data is None:
            return 0.5
        
        # Calculate 30-day relative performance
        asset_return = (data['close'].iloc[-1] / data['close'].iloc[-30]) - 1 if len(data) >= 30 else 0
        benchmark_return = (self._benchmark_data['close'].iloc[-1] / self._benchmark_data['close'].iloc[-30]) - 1 if len(self._benchmark_data) >= 30 else 0
        
        relative_performance = asset_return - benchmark_return
        
        # Normalize using tanh
        cycle_score = (np.tanh(relative_performance * 5) + 1) / 2
        
        return np.clip(cycle_score, 0.0, 1.0)
    
    def _calculate_brazilian_factors_score(self, data: pd.DataFrame, symbol: str) -> float:
        """Calculate Brazilian market factors score"""
        
        # Placeholder implementation - could be enhanced with:
        # - BRL/USD strength analysis  
        # - Mercado Bitcoin specific liquidity metrics
        # - Local regulatory sentiment
        
        # For now, use trading activity as proxy for local interest
        recent_volume = data['volume'].tail(7).mean()
        historical_volume = data['volume'].median()
        
        volume_ratio = recent_volume / historical_volume if historical_volume > 0 else 1.0
        brazilian_score = min(volume_ratio, 2.0) / 2.0  # Cap at 2x historical volume
        
        return np.clip(brazilian_score, 0.0, 1.0)
    
    def _get_momentum_breakdown(self, data: pd.DataFrame) -> Dict[str, float]:
        """Get detailed momentum score breakdown"""
        breakdown = {}
        
        for i, window in enumerate(self.config.momentum_windows):
            if len(data) >= window:
                period_return = (data['close'].iloc[-1] / data['close'].iloc[-window]) - 1
                breakdown[f'{window}d_return'] = period_return
        
        return breakdown
    
    def _get_mean_reversion_breakdown(self, data: pd.DataFrame) -> Dict[str, float]:
        """Get detailed mean reversion score breakdown"""
        
        # RSI
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=self.config.rsi_window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.config.rsi_window).mean()
        rs = gain / loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        
        return {
            'rsi': rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else 50,
            'rsi_oversold': rsi.iloc[-1] < 30 if not np.isnan(rsi.iloc[-1]) else False,
            'rsi_overbought': rsi.iloc[-1] > 70 if not np.isnan(rsi.iloc[-1]) else False
        }
    
    def _get_risk_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Get risk metrics for the asset"""
        returns = data['close'].pct_change().dropna()
        
        return {
            'volatility_30d': returns.tail(30).std() * np.sqrt(252),
            'max_drawdown_30d': self._calculate_max_drawdown(data['close'].tail(30)),
            'sharpe_ratio_estimate': self._calculate_sharpe_ratio(returns.tail(30))
        }
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown"""
        peak = prices.expanding(min_periods=1).max()
        drawdown = (prices / peak) - 1
        return abs(drawdown.min())
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio estimate"""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        excess_returns = returns.mean() * 252 - risk_free_rate  # Annualized
        volatility = returns.std() * np.sqrt(252)
        
        return excess_returns / volatility
    
    def get_top_assets(self, scores: Dict[str, AssetScore], n: int = 10) -> List[AssetScore]:
        """Get top N assets by composite score"""
        sorted_assets = sorted(scores.values(), key=lambda x: x.composite_score, reverse=True)
        return sorted_assets[:n]
    
    def get_scoring_summary(self, scores: Dict[str, AssetScore]) -> Dict[str, any]:
        """Get summary statistics of scoring results"""
        
        if not scores:
            return {'error': 'No scores available'}
        
        composite_scores = [score.composite_score for score in scores.values()]
        
        # Component analysis
        component_averages = {}
        for component in ScoreComponent:
            component_scores = [score.component_scores.get(component.value, 0) for score in scores.values()]
            component_averages[component.value] = np.mean(component_scores)
        
        return {
            'total_assets': len(scores),
            'score_statistics': {
                'mean': np.mean(composite_scores),
                'std': np.std(composite_scores),
                'min': np.min(composite_scores),
                'max': np.max(composite_scores),
                'median': np.median(composite_scores)
            },
            'component_averages': component_averages,
            'top_10_assets': [score.symbol for score in self.get_top_assets(scores, 10)]
        }


# Multi-Factor Scoring Engine is integrated into the main DCA system
# Use the main CLI for testing: python cli.py run
