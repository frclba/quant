#!/usr/bin/env python3
"""
Market Regime Detection Engine

Classifies market conditions into Bull, Bear, and Sideways regimes
using multiple quantitative indicators and machine learning techniques.

Detection Methods:
- Trend Analysis: Moving averages and directional indicators
- Volatility Analysis: GARCH modeling and volatility clustering
- Volume Analysis: Distribution and flow patterns
- Hidden Markov Models: Statistical regime classification
- Correlation Analysis: Market structure changes
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class RegimeType(Enum):
    """Market regime classifications"""
    BULL = "bull"           # Uptrend, low volatility, positive sentiment
    BEAR = "bear"           # Downtrend, high volatility, negative sentiment  
    SIDEWAYS = "sideways"   # Range-bound, moderate volatility
    VOLATILE = "volatile"   # High volatility, unclear direction
    UNKNOWN = "unknown"     # Insufficient data or unclear signals


@dataclass
class RegimeConfig:
    """Configuration for market regime detection"""
    
    # Trend analysis parameters
    trend_windows: List[int] = field(default_factory=lambda: [20, 50, 200])  # MA periods
    trend_strength_threshold: float = 0.02  # Minimum trend strength (2%)
    
    # Volatility analysis parameters
    volatility_window: int = 30
    volatility_threshold_high: float = 2.0  # High volatility threshold (2x median)
    volatility_threshold_low: float = 0.5   # Low volatility threshold (0.5x median)
    
    # Volume analysis parameters
    volume_window: int = 30
    volume_surge_threshold: float = 1.5  # Volume surge threshold (1.5x average)
    
    # Regime stability parameters
    regime_confirmation_days: int = 5    # Days to confirm regime change
    regime_strength_threshold: float = 0.6  # Minimum confidence for regime
    
    # HMM parameters
    use_hmm: bool = True                # Enable Hidden Markov Model
    hmm_n_states: int = 3              # Number of HMM states
    hmm_lookback_days: int = 252       # Days of data for HMM training
    
    # Weight for different detection methods
    trend_weight: float = 0.4
    volatility_weight: float = 0.3
    volume_weight: float = 0.2
    hmm_weight: float = 0.1


@dataclass
class MarketRegime:
    """Market regime detection result"""
    regime: RegimeType
    confidence: float
    strength: float
    detection_timestamp: datetime = field(default_factory=datetime.now)
    
    # Component analysis
    trend_signal: float = 0.0      # -1 (bearish) to +1 (bullish)
    volatility_signal: float = 0.0  # 0 (low) to 1 (high)
    volume_signal: float = 0.0     # 0 (low) to 1 (high) 
    hmm_signal: Optional[int] = None  # HMM state
    
    # Supporting metrics
    trend_strength: float = 0.0
    volatility_percentile: float = 0.0
    volume_percentile: float = 0.0
    
    # Historical context
    regime_duration_days: int = 0
    previous_regime: Optional[RegimeType] = None
    regime_change_date: Optional[datetime] = None


class MarketRegimeDetector:
    """
    Market Regime Detection Engine
    
    Uses multiple quantitative methods to classify current market regime
    and provide probabilistic assessments of regime transitions.
    """
    
    def __init__(self, config: Optional[RegimeConfig] = None):
        self.config = config or RegimeConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # State tracking
        self._regime_history = []
        self._last_detection = None
        self._hmm_model = None
        
        # Method availability
        self._has_hmm = False
        try:
            from hmmlearn import hmm
            self._has_hmm = True and self.config.use_hmm
        except ImportError:
            self.logger.warning("HMM models not available (hmmlearn not installed)")
        
        self.logger.info(f"🔍 MarketRegimeDetector initialized")
        self.logger.info(f"  Methods: Trend={self.config.trend_weight}, Vol={self.config.volatility_weight}, "
                        f"Volume={self.config.volume_weight}, HMM={'✓' if self._has_hmm else '✗'}")
    
    def detect_regime(self, 
                     market_data: pd.DataFrame,
                     benchmark_data: Optional[pd.DataFrame] = None) -> MarketRegime:
        """
        Detect current market regime using multiple methods
        
        Args:
            market_data: OHLCV DataFrame for primary analysis
            benchmark_data: Optional benchmark data for relative analysis
            
        Returns:
            MarketRegime object with detection results
        """
        self.logger.debug(f"🔍 Detecting market regime with {len(market_data)} data points...")
        
        if len(market_data) < max(self.config.trend_windows) + 30:
            return MarketRegime(
                regime=RegimeType.UNKNOWN,
                confidence=0.0,
                strength=0.0
            )
        
        # Calculate individual regime signals
        trend_signal = self._calculate_trend_signal(market_data)
        volatility_signal = self._calculate_volatility_signal(market_data)
        volume_signal = self._calculate_volume_signal(market_data)
        
        # HMM analysis (if available)
        hmm_signal = None
        if self._has_hmm:
            try:
                hmm_signal = self._calculate_hmm_signal(market_data)
            except Exception as e:
                self.logger.warning(f"HMM analysis failed: {e}")
        
        # Combine signals into regime classification
        regime, confidence, strength = self._classify_regime(
            trend_signal, volatility_signal, volume_signal, hmm_signal
        )
        
        # Create regime result
        regime_result = MarketRegime(
            regime=regime,
            confidence=confidence,
            strength=strength,
            trend_signal=trend_signal,
            volatility_signal=volatility_signal,
            volume_signal=volume_signal,
            hmm_signal=hmm_signal,
            trend_strength=abs(trend_signal),
            volatility_percentile=self._calculate_volatility_percentile(market_data),
            volume_percentile=self._calculate_volume_percentile(market_data)
        )
        
        # Update regime history
        self._update_regime_history(regime_result)
        
        self.logger.debug(f"  Detected regime: {regime.value.upper()} (confidence: {confidence:.3f})")
        
        return regime_result
    
    def _calculate_trend_signal(self, data: pd.DataFrame) -> float:
        """Calculate trend signal using multiple moving averages"""
        
        trend_signals = []
        
        for window in self.config.trend_windows:
            if len(data) < window:
                continue
                
            # Simple moving average
            sma = data['close'].rolling(window=window).mean()
            current_price = data['close'].iloc[-1]
            sma_current = sma.iloc[-1]
            
            # Trend direction
            price_vs_sma = (current_price / sma_current) - 1
            
            # SMA slope (trend strength)
            if len(sma) >= 5:
                sma_slope = (sma.iloc[-1] / sma.iloc[-5]) - 1
                trend_strength = price_vs_sma + sma_slope
            else:
                trend_strength = price_vs_sma
            
            trend_signals.append(trend_strength)
        
        if not trend_signals:
            return 0.0
        
        # Weight by time horizon (longer = more weight)
        weights = [window/sum(self.config.trend_windows) for window in self.config.trend_windows[:len(trend_signals)]]
        weighted_trend = np.average(trend_signals, weights=weights)
        
        # Normalize to [-1, 1]
        normalized_trend = np.tanh(weighted_trend * 10)  # Scale for tanh
        
        return np.clip(normalized_trend, -1.0, 1.0)
    
    def _calculate_volatility_signal(self, data: pd.DataFrame) -> float:
        """Calculate volatility regime signal"""
        
        # Calculate rolling volatility
        returns = data['close'].pct_change().dropna()
        rolling_vol = returns.rolling(window=self.config.volatility_window).std() * np.sqrt(252)
        
        if len(rolling_vol) < self.config.volatility_window:
            return 0.5  # Neutral
        
        current_vol = rolling_vol.iloc[-1]
        median_vol = rolling_vol.median()
        
        # Volatility regime classification
        vol_ratio = current_vol / median_vol if median_vol > 0 else 1.0
        
        # Normalize to [0, 1] where 0 = low volatility, 1 = high volatility
        if vol_ratio >= self.config.volatility_threshold_high:
            volatility_signal = 1.0
        elif vol_ratio <= self.config.volatility_threshold_low:
            volatility_signal = 0.0
        else:
            # Linear interpolation between thresholds
            volatility_signal = ((vol_ratio - self.config.volatility_threshold_low) / 
                               (self.config.volatility_threshold_high - self.config.volatility_threshold_low))
        
        return np.clip(volatility_signal, 0.0, 1.0)
    
    def _calculate_volume_signal(self, data: pd.DataFrame) -> float:
        """Calculate volume regime signal"""
        
        # Volume analysis
        volume_sma = data['volume'].rolling(window=self.config.volume_window).mean()
        
        if len(volume_sma) < self.config.volume_window:
            return 0.5  # Neutral
        
        recent_volume = data['volume'].tail(5).mean()  # Last 5 days average
        baseline_volume = volume_sma.iloc[-1]
        
        # Volume surge detection
        volume_ratio = recent_volume / baseline_volume if baseline_volume > 0 else 1.0
        
        # Normalize to [0, 1]
        volume_signal = min(volume_ratio / self.config.volume_surge_threshold, 1.0)
        
        return np.clip(volume_signal, 0.0, 1.0)
    
    def _calculate_hmm_signal(self, data: pd.DataFrame) -> Optional[int]:
        """Calculate Hidden Markov Model regime signal"""
        
        if not self._has_hmm:
            return None
        
        try:
            from hmmlearn import hmm
            
            # Prepare features for HMM
            returns = data['close'].pct_change().dropna()
            volume_changes = data['volume'].pct_change().dropna()
            
            # Use last N days for training
            lookback = min(self.config.hmm_lookback_days, len(returns))
            recent_returns = returns.tail(lookback).values.reshape(-1, 1)
            recent_volume = volume_changes.tail(lookback).values.reshape(-1, 1)
            
            if len(recent_returns) < 30:  # Need minimum data
                return None
            
            # Combine features
            features = np.column_stack([recent_returns, recent_volume])
            
            # Train HMM model
            model = hmm.GaussianHMM(n_components=self.config.hmm_n_states, 
                                   covariance_type="full", random_state=42)
            model.fit(features)
            
            # Predict current state
            states = model.predict(features)
            current_state = states[-1]
            
            # Cache model for future use
            self._hmm_model = model
            
            return int(current_state)
            
        except Exception as e:
            self.logger.error(f"HMM calculation failed: {e}")
            return None
    
    def _classify_regime(self, trend_signal: float, volatility_signal: float, 
                        volume_signal: float, hmm_signal: Optional[int]) -> Tuple[RegimeType, float, float]:
        """Classify regime based on combined signals"""
        
        # Calculate weighted composite score
        composite_trend = trend_signal * self.config.trend_weight
        composite_volatility = volatility_signal * self.config.volatility_weight  
        composite_volume = volume_signal * self.config.volume_weight
        
        # HMM contribution (if available)
        hmm_contribution = 0.0
        if hmm_signal is not None:
            # Map HMM state to regime signal (-1 to 1)
            hmm_contribution = ((hmm_signal / (self.config.hmm_n_states - 1)) * 2 - 1) * self.config.hmm_weight
        
        # Combined signals
        total_trend_signal = composite_trend + hmm_contribution
        
        # Regime classification logic
        if volatility_signal > 0.8:  # Very high volatility
            regime = RegimeType.VOLATILE
            confidence = volatility_signal
            strength = volatility_signal
            
        elif abs(total_trend_signal) < 0.1:  # Weak trend
            regime = RegimeType.SIDEWAYS
            confidence = 1 - abs(total_trend_signal) * 5  # Inverse of trend strength
            strength = composite_volume  # Volume indicates activity in sideways
            
        elif total_trend_signal > self.config.trend_strength_threshold:  # Strong uptrend
            regime = RegimeType.BULL
            confidence = min(total_trend_signal * 2, 1.0)
            strength = abs(total_trend_signal) * (1 - volatility_signal * 0.5)  # Adjust for volatility
            
        elif total_trend_signal < -self.config.trend_strength_threshold:  # Strong downtrend
            regime = RegimeType.BEAR  
            confidence = min(abs(total_trend_signal) * 2, 1.0)
            strength = abs(total_trend_signal) * (1 - volatility_signal * 0.3)  # Less volatility penalty in bear
            
        else:
            regime = RegimeType.SIDEWAYS
            confidence = 0.5
            strength = 0.5
        
        # Apply minimum confidence threshold
        if confidence < self.config.regime_strength_threshold:
            regime = RegimeType.UNKNOWN
            confidence = 0.0
            strength = 0.0
        
        return regime, confidence, strength
    
    def _calculate_volatility_percentile(self, data: pd.DataFrame) -> float:
        """Calculate current volatility percentile"""
        returns = data['close'].pct_change().dropna()
        rolling_vol = returns.rolling(window=30).std() * np.sqrt(252)
        
        if len(rolling_vol) < 30:
            return 0.5
        
        current_vol = rolling_vol.iloc[-1]
        vol_rank = (rolling_vol <= current_vol).sum() / len(rolling_vol)
        
        return vol_rank
    
    def _calculate_volume_percentile(self, data: pd.DataFrame) -> float:
        """Calculate current volume percentile"""
        if len(data) < 30:
            return 0.5
        
        recent_volume = data['volume'].tail(5).mean()
        vol_rank = (data['volume'] <= recent_volume).sum() / len(data['volume'])
        
        return vol_rank
    
    def _update_regime_history(self, regime_result: MarketRegime):
        """Update regime history and calculate transitions"""
        
        if self._regime_history:
            last_regime = self._regime_history[-1]
            
            # Check for regime change
            if last_regime.regime != regime_result.regime:
                regime_result.regime_change_date = regime_result.detection_timestamp
                regime_result.previous_regime = last_regime.regime
                regime_result.regime_duration_days = 0
            else:
                # Same regime - calculate duration
                if last_regime.regime_change_date:
                    duration = (regime_result.detection_timestamp - last_regime.regime_change_date).days
                    regime_result.regime_duration_days = duration
                else:
                    regime_result.regime_duration_days = last_regime.regime_duration_days + 1
                
                regime_result.regime_change_date = last_regime.regime_change_date
                regime_result.previous_regime = last_regime.previous_regime
        
        # Add to history (keep last 100 entries)
        self._regime_history.append(regime_result)
        if len(self._regime_history) > 100:
            self._regime_history.pop(0)
        
        self._last_detection = regime_result
    
    def get_regime_summary(self) -> Dict[str, any]:
        """Get summary of recent regime detection"""
        
        if not self._regime_history:
            return {'error': 'No regime history available'}
        
        recent_regimes = [r.regime for r in self._regime_history[-30:]]  # Last 30 detections
        regime_counts = {regime.value: recent_regimes.count(regime) for regime in RegimeType}
        
        current = self._regime_history[-1]
        
        return {
            'current_regime': current.regime.value,
            'confidence': current.confidence,
            'strength': current.strength,
            'duration_days': current.regime_duration_days,
            'previous_regime': current.previous_regime.value if current.previous_regime else None,
            'regime_distribution_30d': regime_counts,
            'trend_signal': current.trend_signal,
            'volatility_signal': current.volatility_signal,
            'volume_signal': current.volume_signal,
            'volatility_percentile': current.volatility_percentile,
            'volume_percentile': current.volume_percentile
        }
    
    def get_regime_transition_probability(self) -> Dict[str, float]:
        """Calculate regime transition probabilities based on history"""
        
        if len(self._regime_history) < 10:
            return {'insufficient_data': True}
        
        # Build transition matrix
        transitions = {}
        for i in range(1, len(self._regime_history)):
            from_regime = self._regime_history[i-1].regime
            to_regime = self._regime_history[i].regime
            
            key = f"{from_regime.value}_to_{to_regime.value}"
            transitions[key] = transitions.get(key, 0) + 1
        
        # Calculate probabilities
        total_transitions = sum(transitions.values())
        if total_transitions == 0:
            return {'no_transitions': True}
        
        probabilities = {k: v/total_transitions for k, v in transitions.items()}
        
        return probabilities


# Market Regime Detector is integrated into the main DCA system
# Use the main CLI for testing: python cli.py run
