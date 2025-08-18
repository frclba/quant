#!/usr/bin/env python3
"""
Predictive Signal Engine

Generates forward-looking predictive signals using advanced techniques
including machine learning, statistical models, and market microstructure analysis.

Signal Types:
- Momentum Persistence: LSTM-enhanced trend continuation
- Mean Reversion: Statistical overbought/oversold signals
- Breakout Detection: Support/resistance level breaks
- Volume Flow: Institutional vs retail flow analysis
- Cycle Signals: Market cycle and seasonal patterns
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Types of predictive signals"""
    MOMENTUM_PERSISTENCE = "momentum_persistence"
    MEAN_REVERSION = "mean_reversion" 
    BREAKOUT = "breakout"
    VOLUME_FLOW = "volume_flow"
    CYCLE_SIGNAL = "cycle_signal"
    ML_PREDICTION = "ml_prediction"


class SignalDirection(Enum):
    """Signal direction"""
    BULLISH = "bullish"      # Positive price expectation
    BEARISH = "bearish"      # Negative price expectation
    NEUTRAL = "neutral"      # No directional bias
    UNKNOWN = "unknown"      # Insufficient data


@dataclass
class SignalConfig:
    """Configuration for predictive signal generation"""
    
    # Momentum persistence parameters
    momentum_lookback_days: int = 126        # 6 months lookback
    momentum_persistence_threshold: float = 0.3  # Minimum persistence score
    momentum_decay_factor: float = 0.95     # Decay factor for older signals
    
    # Mean reversion parameters  
    mean_reversion_window: int = 20         # Bollinger Band window
    mean_reversion_std: float = 2.0         # Standard deviation multiplier
    rsi_oversold: float = 30                # RSI oversold threshold
    rsi_overbought: float = 70              # RSI overbought threshold
    
    # Breakout detection parameters
    support_resistance_window: int = 50     # Lookback for S/R levels
    breakout_volume_threshold: float = 1.5  # Volume confirmation threshold
    breakout_price_threshold: float = 0.02  # Price move threshold (2%)
    
    # Volume flow parameters
    volume_flow_window: int = 20           # Analysis window
    flow_divergence_threshold: float = 0.3  # Divergence threshold
    institutional_flow_proxy: float = 2.0   # Large volume threshold
    
    # Cycle signal parameters
    cycle_analysis_window: int = 252        # 1 year cycle analysis
    seasonal_strength_threshold: float = 0.25  # Minimum seasonal strength
    
    # Machine Learning parameters (if available)
    use_ml_predictions: bool = True         # Enable ML predictions
    ml_feature_window: int = 30            # Features lookback window
    ml_prediction_horizon: int = 5          # Days ahead to predict
    ml_confidence_threshold: float = 0.6    # Minimum ML confidence
    
    # Signal combination parameters
    signal_weights: Dict[str, float] = field(default_factory=lambda: {
        'momentum_persistence': 0.25,
        'mean_reversion': 0.20,
        'breakout': 0.20,
        'volume_flow': 0.15,
        'cycle_signal': 0.10,
        'ml_prediction': 0.10
    })
    
    # Signal validation
    min_data_points: int = 100             # Minimum data points required
    signal_decay_days: int = 3             # Days for signal decay
    confidence_decay_factor: float = 0.8   # Daily confidence decay


@dataclass 
class PredictiveSignal:
    """Predictive signal result"""
    symbol: str
    signal_type: SignalType
    direction: SignalDirection
    strength: float                        # 0 to 1 signal strength
    confidence: float                      # 0 to 1 confidence level
    time_horizon_days: int                 # Signal time horizon
    signal_timestamp: datetime = field(default_factory=datetime.now)
    
    # Signal details
    entry_price: Optional[float] = None    # Suggested entry price
    target_price: Optional[float] = None   # Price target
    stop_loss: Optional[float] = None      # Risk management level
    expected_return: Optional[float] = None # Expected return %
    
    # Supporting data
    technical_indicators: Dict[str, float] = field(default_factory=dict)
    risk_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Meta information
    signal_components: Dict[str, float] = field(default_factory=dict)
    validation_score: float = 0.0          # Signal validation score


class PredictiveSignalEngine:
    """
    Predictive Signal Engine
    
    Generates forward-looking signals combining technical analysis,
    statistical models, and machine learning predictions.
    """
    
    def __init__(self, config: Optional[SignalConfig] = None):
        self.config = config or SignalConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Signal history tracking
        self._signal_history = {}
        self._performance_tracking = {}
        
        # ML model (placeholder for actual implementation)
        self._ml_model = None
        self._ml_available = False
        
        # Try to initialize ML components
        try:
            # Placeholder for ML model initialization
            # from sklearn.ensemble import RandomForestRegressor
            # self._ml_model = RandomForestRegressor(...)
            self._ml_available = False  # Set to True when ML is implemented
        except ImportError:
            self.logger.warning("ML libraries not available for predictive signals")
        
        self.logger.info(f"🎯 PredictiveSignalEngine initialized")
        self.logger.info(f"  Signal types: {len(self.config.signal_weights)} enabled")
        self.logger.info(f"  ML predictions: {'✓' if self._ml_available else '✗'}")
    
    def generate_signals(self, 
                        market_data: Dict[str, pd.DataFrame],
                        benchmark_symbol: str = 'BTC') -> Dict[str, List[PredictiveSignal]]:
        """
        Generate predictive signals for all assets
        
        Args:
            market_data: Dictionary of OHLCV DataFrames by symbol
            benchmark_symbol: Benchmark for relative analysis
            
        Returns:
            Dictionary of signal lists by symbol
        """
        self.logger.info(f"🔮 Generating predictive signals for {len(market_data)} assets...")
        
        all_signals = {}
        
        for symbol, data in market_data.items():
            if len(data) < self.config.min_data_points:
                self.logger.debug(f"Insufficient data for {symbol}, skipping")
                continue
            
            try:
                signals = self._generate_asset_signals(symbol, data, market_data.get(benchmark_symbol))
                if signals:
                    all_signals[symbol] = signals
                    self.logger.debug(f"  {symbol}: {len(signals)} signals generated")
                
            except Exception as e:
                self.logger.error(f"Error generating signals for {symbol}: {e}")
                continue
        
        self.logger.info(f"✅ Generated signals for {len(all_signals)} assets")
        return all_signals
    
    def _generate_asset_signals(self, 
                               symbol: str, 
                               data: pd.DataFrame,
                               benchmark_data: Optional[pd.DataFrame] = None) -> List[PredictiveSignal]:
        """Generate all signal types for a single asset"""
        
        signals = []
        
        # 1. Momentum Persistence Signal
        momentum_signal = self._generate_momentum_persistence_signal(symbol, data)
        if momentum_signal:
            signals.append(momentum_signal)
        
        # 2. Mean Reversion Signal
        mean_reversion_signal = self._generate_mean_reversion_signal(symbol, data)
        if mean_reversion_signal:
            signals.append(mean_reversion_signal)
        
        # 3. Breakout Signal
        breakout_signal = self._generate_breakout_signal(symbol, data)
        if breakout_signal:
            signals.append(breakout_signal)
        
        # 4. Volume Flow Signal
        volume_flow_signal = self._generate_volume_flow_signal(symbol, data)
        if volume_flow_signal:
            signals.append(volume_flow_signal)
        
        # 5. Cycle Signal
        cycle_signal = self._generate_cycle_signal(symbol, data, benchmark_data)
        if cycle_signal:
            signals.append(cycle_signal)
        
        # 6. ML Prediction Signal (if available)
        if self._ml_available:
            ml_signal = self._generate_ml_signal(symbol, data)
            if ml_signal:
                signals.append(ml_signal)
        
        # Filter and validate signals
        validated_signals = [s for s in signals if self._validate_signal(s)]
        
        return validated_signals
    
    def _generate_momentum_persistence_signal(self, symbol: str, data: pd.DataFrame) -> Optional[PredictiveSignal]:
        """Generate momentum persistence signal"""
        
        if len(data) < self.config.momentum_lookback_days:
            return None
        
        # Calculate momentum over multiple timeframes
        returns_1m = (data['close'].iloc[-21] / data['close'].iloc[-42]) - 1 if len(data) >= 42 else 0
        returns_3m = (data['close'].iloc[-1] / data['close'].iloc[-63]) - 1 if len(data) >= 63 else 0
        returns_6m = (data['close'].iloc[-1] / data['close'].iloc[-126]) - 1 if len(data) >= 126 else 0
        
        # Weight recent momentum more heavily
        momentum_score = (returns_1m * 0.5 + returns_3m * 0.3 + returns_6m * 0.2)
        
        # Calculate momentum persistence (trend stability)
        price_changes = data['close'].pct_change().dropna().tail(self.config.momentum_lookback_days // 2)
        if len(price_changes) < 30:
            return None
        
        # Persistence = consistency of direction
        positive_days = (price_changes > 0).sum()
        persistence = abs(positive_days / len(price_changes) - 0.5) * 2  # 0 to 1
        
        # Only generate signal if persistence is above threshold
        if persistence < self.config.momentum_persistence_threshold:
            return None
        
        # Determine signal direction and strength
        if momentum_score > 0.05:  # 5% positive momentum
            direction = SignalDirection.BULLISH
            strength = min(momentum_score * 5, 1.0)  # Scale to 0-1
        elif momentum_score < -0.05:  # 5% negative momentum
            direction = SignalDirection.BEARISH
            strength = min(abs(momentum_score) * 5, 1.0)
        else:
            direction = SignalDirection.NEUTRAL
            strength = 0.5
        
        # Calculate confidence based on persistence and volatility
        volatility = price_changes.std() * np.sqrt(252)
        confidence = persistence * (1 - min(volatility / 2.0, 0.8))  # Lower vol = higher confidence
        
        return PredictiveSignal(
            symbol=symbol,
            signal_type=SignalType.MOMENTUM_PERSISTENCE,
            direction=direction,
            strength=strength,
            confidence=confidence,
            time_horizon_days=21,  # 1 month horizon
            entry_price=data['close'].iloc[-1],
            expected_return=momentum_score,
            technical_indicators={
                'momentum_1m': returns_1m,
                'momentum_3m': returns_3m,
                'momentum_6m': returns_6m,
                'persistence_score': persistence
            }
        )
    
    def _generate_mean_reversion_signal(self, symbol: str, data: pd.DataFrame) -> Optional[PredictiveSignal]:
        """Generate mean reversion signal"""
        
        window = self.config.mean_reversion_window
        if len(data) < window + 20:
            return None
        
        # Bollinger Bands
        sma = data['close'].rolling(window=window).mean()
        std = data['close'].rolling(window=window).std()
        upper_band = sma + (std * self.config.mean_reversion_std)
        lower_band = sma - (std * self.config.mean_reversion_std)
        
        current_price = data['close'].iloc[-1]
        current_sma = sma.iloc[-1]
        current_upper = upper_band.iloc[-1]
        current_lower = lower_band.iloc[-1]
        
        # RSI calculation
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        # Mean reversion conditions
        bb_position = (current_price - current_lower) / (current_upper - current_lower)
        
        signal = None
        
        # Oversold conditions (bullish mean reversion)
        if (current_rsi < self.config.rsi_oversold or bb_position < 0.1) and current_price < current_sma:
            direction = SignalDirection.BULLISH
            strength = (self.config.rsi_oversold - current_rsi) / self.config.rsi_oversold + (0.1 - bb_position)
            strength = min(strength, 1.0)
            confidence = strength * 0.8  # Mean reversion less certain than momentum
            target_price = current_sma  # Target mean
            
            signal = PredictiveSignal(
                symbol=symbol,
                signal_type=SignalType.MEAN_REVERSION,
                direction=direction,
                strength=strength,
                confidence=confidence,
                time_horizon_days=10,  # Shorter horizon for mean reversion
                entry_price=current_price,
                target_price=target_price,
                expected_return=(target_price - current_price) / current_price,
                technical_indicators={
                    'rsi': current_rsi,
                    'bb_position': bb_position,
                    'price_vs_sma': (current_price / current_sma) - 1
                }
            )
        
        # Overbought conditions (bearish mean reversion)
        elif (current_rsi > self.config.rsi_overbought or bb_position > 0.9) and current_price > current_sma:
            direction = SignalDirection.BEARISH
            strength = (current_rsi - self.config.rsi_overbought) / (100 - self.config.rsi_overbought) + (bb_position - 0.9)
            strength = min(strength, 1.0)
            confidence = strength * 0.8
            target_price = current_sma  # Target mean
            
            signal = PredictiveSignal(
                symbol=symbol,
                signal_type=SignalType.MEAN_REVERSION,
                direction=direction,
                strength=strength,
                confidence=confidence,
                time_horizon_days=10,
                entry_price=current_price,
                target_price=target_price,
                expected_return=(target_price - current_price) / current_price,
                technical_indicators={
                    'rsi': current_rsi,
                    'bb_position': bb_position,
                    'price_vs_sma': (current_price / current_sma) - 1
                }
            )
        
        return signal
    
    def _generate_breakout_signal(self, symbol: str, data: pd.DataFrame) -> Optional[PredictiveSignal]:
        """Generate breakout signal"""
        
        window = self.config.support_resistance_window
        if len(data) < window + 10:
            return None
        
        # Calculate support and resistance levels
        recent_data = data.tail(window)
        high_levels = recent_data['high'].rolling(window=10).max()
        low_levels = recent_data['low'].rolling(window=10).min()
        
        # Current levels
        resistance = high_levels.iloc[-5:].min()  # Recent resistance
        support = low_levels.iloc[-5:].max()      # Recent support
        current_price = data['close'].iloc[-1]
        current_volume = data['volume'].iloc[-1]
        avg_volume = data['volume'].rolling(window=20).mean().iloc[-1]
        
        # Breakout conditions
        signal = None
        
        # Upside breakout
        price_above_resistance = (current_price - resistance) / resistance
        volume_confirmation = current_volume > (avg_volume * self.config.breakout_volume_threshold)
        
        if (price_above_resistance > self.config.breakout_price_threshold and 
            volume_confirmation):
            
            strength = min(price_above_resistance * 10, 1.0)
            confidence = strength * (0.7 if volume_confirmation else 0.4)
            target_price = resistance * (1 + price_above_resistance)  # Project move
            
            signal = PredictiveSignal(
                symbol=symbol,
                signal_type=SignalType.BREAKOUT,
                direction=SignalDirection.BULLISH,
                strength=strength,
                confidence=confidence,
                time_horizon_days=14,  # Medium-term breakout
                entry_price=current_price,
                target_price=target_price,
                stop_loss=resistance * 0.98,  # Stop below resistance
                expected_return=(target_price - current_price) / current_price,
                technical_indicators={
                    'resistance_level': resistance,
                    'breakout_magnitude': price_above_resistance,
                    'volume_ratio': current_volume / avg_volume
                }
            )
        
        # Downside breakdown
        price_below_support = (support - current_price) / support
        
        if (price_below_support > self.config.breakout_price_threshold and 
            volume_confirmation):
            
            strength = min(price_below_support * 10, 1.0)
            confidence = strength * (0.7 if volume_confirmation else 0.4)
            target_price = support * (1 - price_below_support)  # Project move
            
            signal = PredictiveSignal(
                symbol=symbol,
                signal_type=SignalType.BREAKOUT,
                direction=SignalDirection.BEARISH,
                strength=strength,
                confidence=confidence,
                time_horizon_days=14,
                entry_price=current_price,
                target_price=target_price,
                stop_loss=support * 1.02,  # Stop above support
                expected_return=(target_price - current_price) / current_price,
                technical_indicators={
                    'support_level': support,
                    'breakdown_magnitude': price_below_support,
                    'volume_ratio': current_volume / avg_volume
                }
            )
        
        return signal
    
    def _generate_volume_flow_signal(self, symbol: str, data: pd.DataFrame) -> Optional[PredictiveSignal]:
        """Generate volume flow signal"""
        
        window = self.config.volume_flow_window
        if len(data) < window + 10:
            return None
        
        # Volume-Price Analysis
        recent_data = data.tail(window)
        
        # On-Balance Volume (OBV)
        obv = []
        obv_val = 0
        for i in range(len(recent_data)):
            if i == 0:
                obv_val = recent_data['volume'].iloc[i]
            else:
                if recent_data['close'].iloc[i] > recent_data['close'].iloc[i-1]:
                    obv_val += recent_data['volume'].iloc[i]
                elif recent_data['close'].iloc[i] < recent_data['close'].iloc[i-1]:
                    obv_val -= recent_data['volume'].iloc[i]
            obv.append(obv_val)
        
        obv_series = pd.Series(obv, index=recent_data.index)
        
        # Volume-weighted price
        vwap = (recent_data['close'] * recent_data['volume']).cumsum() / recent_data['volume'].cumsum()
        current_price = data['close'].iloc[-1]
        current_vwap = vwap.iloc[-1]
        
        # Flow divergence analysis
        price_trend = (recent_data['close'].iloc[-1] / recent_data['close'].iloc[0]) - 1
        obv_trend = (obv[-1] / obv[0]) - 1 if obv[0] != 0 else 0
        
        # Institutional flow proxy (large volume days)
        high_volume_days = recent_data[recent_data['volume'] > 
                                     recent_data['volume'].mean() * self.config.institutional_flow_proxy]
        
        if len(high_volume_days) == 0:
            return None
        
        # Price performance on high volume days
        high_vol_returns = high_volume_days['close'].pct_change().mean()
        
        # Generate signal based on flow analysis
        signal = None
        
        # Positive flow divergence (volume supporting price)
        if (obv_trend > price_trend + self.config.flow_divergence_threshold and 
            current_price < current_vwap and high_vol_returns > 0):
            
            strength = min(abs(obv_trend - price_trend), 1.0)
            confidence = strength * 0.6  # Volume signals less reliable
            
            signal = PredictiveSignal(
                symbol=symbol,
                signal_type=SignalType.VOLUME_FLOW,
                direction=SignalDirection.BULLISH,
                strength=strength,
                confidence=confidence,
                time_horizon_days=7,  # Short-term flow signal
                entry_price=current_price,
                target_price=current_vwap,  # Target VWAP
                expected_return=(current_vwap - current_price) / current_price,
                technical_indicators={
                    'obv_trend': obv_trend,
                    'price_trend': price_trend,
                    'flow_divergence': obv_trend - price_trend,
                    'price_vs_vwap': (current_price / current_vwap) - 1,
                    'institutional_flow': high_vol_returns
                }
            )
        
        return signal
    
    def _generate_cycle_signal(self, symbol: str, data: pd.DataFrame, 
                             benchmark_data: Optional[pd.DataFrame] = None) -> Optional[PredictiveSignal]:
        """Generate cycle-based signal"""
        
        if len(data) < self.config.cycle_analysis_window:
            return None
        
        # Seasonal analysis (day of month, day of week effects)
        data_with_time = data.copy()
        data_with_time['day_of_month'] = data_with_time.index.day
        data_with_time['day_of_week'] = data_with_time.index.dayofweek
        data_with_time['returns'] = data_with_time['close'].pct_change()
        
        # Average returns by day of month
        monthly_pattern = data_with_time.groupby('day_of_month')['returns'].mean()
        current_day = datetime.now().day
        
        # Seasonal strength
        monthly_std = monthly_pattern.std()
        if monthly_std == 0:
            return None
        
        seasonal_strength = abs(monthly_pattern[current_day]) / monthly_std
        
        if seasonal_strength < self.config.seasonal_strength_threshold:
            return None
        
        # Market cycle relative to benchmark
        cycle_strength = 0.5  # Default
        if benchmark_data is not None and len(benchmark_data) >= 60:
            # Relative performance over cycle
            asset_60d = (data['close'].iloc[-1] / data['close'].iloc[-60]) - 1
            benchmark_60d = (benchmark_data['close'].iloc[-1] / benchmark_data['close'].iloc[-60]) - 1
            relative_performance = asset_60d - benchmark_60d
            cycle_strength = (np.tanh(relative_performance * 3) + 1) / 2  # Normalize
        
        # Signal direction based on seasonal pattern
        seasonal_return = monthly_pattern[current_day]
        direction = (SignalDirection.BULLISH if seasonal_return > 0 else 
                    SignalDirection.BEARISH if seasonal_return < 0 else 
                    SignalDirection.NEUTRAL)
        
        return PredictiveSignal(
            symbol=symbol,
            signal_type=SignalType.CYCLE_SIGNAL,
            direction=direction,
            strength=seasonal_strength,
            confidence=seasonal_strength * cycle_strength * 0.5,  # Lower confidence for cycles
            time_horizon_days=30,  # Monthly cycle
            entry_price=data['close'].iloc[-1],
            expected_return=seasonal_return,
            technical_indicators={
                'seasonal_return': seasonal_return,
                'seasonal_strength': seasonal_strength,
                'cycle_position': cycle_strength,
                'current_day_of_month': current_day
            }
        )
    
    def _generate_ml_signal(self, symbol: str, data: pd.DataFrame) -> Optional[PredictiveSignal]:
        """Generate ML-based prediction signal (placeholder)"""
        
        # Placeholder for ML implementation
        # This would use trained models to generate predictions
        
        if not self._ml_available or len(data) < self.config.ml_feature_window + 10:
            return None
        
        # Feature engineering (placeholder)
        features = self._extract_ml_features(data)
        if features is None:
            return None
        
        # Model prediction (placeholder)
        try:
            # prediction = self._ml_model.predict(features)
            # confidence = self._ml_model.predict_proba(features).max()
            
            # Placeholder values
            prediction = np.random.choice([-1, 0, 1])  # -1=bearish, 0=neutral, 1=bullish  
            confidence = np.random.uniform(0.5, 0.9)
            
            if confidence < self.config.ml_confidence_threshold:
                return None
            
            direction_map = {-1: SignalDirection.BEARISH, 0: SignalDirection.NEUTRAL, 1: SignalDirection.BULLISH}
            direction = direction_map[prediction]
            
            return PredictiveSignal(
                symbol=symbol,
                signal_type=SignalType.ML_PREDICTION,
                direction=direction,
                strength=abs(prediction),
                confidence=confidence,
                time_horizon_days=self.config.ml_prediction_horizon,
                entry_price=data['close'].iloc[-1],
                technical_indicators={
                    'ml_prediction': float(prediction),
                    'ml_confidence': confidence
                }
            )
            
        except Exception as e:
            self.logger.error(f"ML prediction failed for {symbol}: {e}")
            return None
    
    def _extract_ml_features(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Extract features for ML model (placeholder)"""
        
        # Placeholder feature extraction
        # Real implementation would extract technical indicators,
        # price patterns, volume features, etc.
        
        try:
            window = self.config.ml_feature_window
            recent_data = data.tail(window)
            
            # Simple features (placeholder)
            features = [
                recent_data['close'].pct_change().mean(),  # Average return
                recent_data['close'].pct_change().std(),   # Volatility
                recent_data['volume'].mean(),              # Average volume
                (recent_data['close'].iloc[-1] / recent_data['close'].iloc[0]) - 1  # Period return
            ]
            
            return np.array(features).reshape(1, -1)
            
        except Exception:
            return None
    
    def _validate_signal(self, signal: PredictiveSignal) -> bool:
        """Validate signal quality and consistency"""
        
        # Basic validation checks
        if signal.strength <= 0 or signal.confidence <= 0:
            return False
        
        if signal.strength > 1.0 or signal.confidence > 1.0:
            return False
        
        # Signal should have reasonable time horizon
        if signal.time_horizon_days <= 0 or signal.time_horizon_days > 90:
            return False
        
        # Check for reasonable expected returns (if provided)
        if signal.expected_return is not None:
            if abs(signal.expected_return) > 2.0:  # > 200% expected return seems unrealistic
                return False
        
        return True
    
    def combine_signals(self, signals: List[PredictiveSignal]) -> Optional[PredictiveSignal]:
        """Combine multiple signals into a composite signal"""
        
        if not signals:
            return None
        
        # Group signals by direction
        bullish_signals = [s for s in signals if s.direction == SignalDirection.BULLISH]
        bearish_signals = [s for s in signals if s.direction == SignalDirection.BEARISH]
        neutral_signals = [s for s in signals if s.direction == SignalDirection.NEUTRAL]
        
        # Weighted scoring
        bullish_score = sum(s.strength * s.confidence * self.config.signal_weights.get(s.signal_type.value, 0.1) 
                           for s in bullish_signals)
        bearish_score = sum(s.strength * s.confidence * self.config.signal_weights.get(s.signal_type.value, 0.1) 
                           for s in bearish_signals)
        
        # Determine combined direction
        if bullish_score > bearish_score * 1.2:  # 20% threshold for directional bias
            direction = SignalDirection.BULLISH
            strength = bullish_score / (bullish_score + bearish_score) if bearish_score > 0 else 1.0
        elif bearish_score > bullish_score * 1.2:
            direction = SignalDirection.BEARISH
            strength = bearish_score / (bullish_score + bearish_score) if bullish_score > 0 else 1.0
        else:
            direction = SignalDirection.NEUTRAL
            strength = 0.5
        
        # Combined confidence (weighted average)
        total_weight = sum(self.config.signal_weights.get(s.signal_type.value, 0.1) for s in signals)
        combined_confidence = sum(s.confidence * self.config.signal_weights.get(s.signal_type.value, 0.1) 
                                 for s in signals) / total_weight if total_weight > 0 else 0
        
        # Combine other metrics
        avg_time_horizon = int(np.mean([s.time_horizon_days for s in signals]))
        avg_expected_return = np.mean([s.expected_return for s in signals if s.expected_return is not None])
        
        return PredictiveSignal(
            symbol=signals[0].symbol,
            signal_type=SignalType.ML_PREDICTION,  # Combined signal type
            direction=direction,
            strength=strength,
            confidence=combined_confidence,
            time_horizon_days=avg_time_horizon,
            entry_price=signals[0].entry_price,
            expected_return=avg_expected_return if not np.isnan(avg_expected_return) else None,
            signal_components={s.signal_type.value: s.strength * s.confidence for s in signals}
        )
    
    def get_signal_summary(self, all_signals: Dict[str, List[PredictiveSignal]]) -> Dict[str, any]:
        """Get summary of all generated signals"""
        
        if not all_signals:
            return {'error': 'No signals available'}
        
        # Count signals by type and direction
        signal_counts = {}
        direction_counts = {d.value: 0 for d in SignalDirection}
        
        for signals in all_signals.values():
            for signal in signals:
                signal_type = signal.signal_type.value
                signal_counts[signal_type] = signal_counts.get(signal_type, 0) + 1
                direction_counts[signal.direction.value] += 1
        
        # Top signals by confidence
        all_signal_list = [s for signals in all_signals.values() for s in signals]
        top_signals = sorted(all_signal_list, key=lambda x: x.confidence * x.strength, reverse=True)[:10]
        
        return {
            'total_assets_with_signals': len(all_signals),
            'total_signals_generated': len(all_signal_list),
            'signal_type_distribution': signal_counts,
            'direction_distribution': direction_counts,
            'average_confidence': np.mean([s.confidence for s in all_signal_list]),
            'average_strength': np.mean([s.strength for s in all_signal_list]),
            'top_signals': [{'symbol': s.symbol, 
                            'type': s.signal_type.value,
                            'direction': s.direction.value,
                            'strength': s.strength,
                            'confidence': s.confidence} for s in top_signals]
        }


# Predictive Signal Engine is integrated into the main DCA system
# Use the main CLI for testing: python cli.py run
