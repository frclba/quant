#!/usr/bin/env python3
"""
Asset Classification System

Classifies assets into investment tiers based on market cap, liquidity,
risk profile, and other quantitative criteria for portfolio allocation.

Tier Classification:
- Tier 1 (50% allocation): $50M-$400M market cap, high liquidity, moderate risk
- Tier 2 (30% allocation): $20M-$150M market cap, medium liquidity, growth profile  
- Tier 3 (15% allocation): $5M-$50M market cap, lower liquidity, speculative
- Reserve (5% allocation): Cash/stablecoins for opportunities and risk buffer
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class AssetTier(Enum):
    """Asset tier classifications"""
    TIER_1 = "tier_1"       # Large, liquid, moderate risk
    TIER_2 = "tier_2"       # Medium, growth-oriented
    TIER_3 = "tier_3"       # Small, speculative, high alpha
    RESERVE = "reserve"     # Cash, stablecoins
    EXCLUDED = "excluded"   # Does not meet criteria


@dataclass
class ClassificationConfig:
    """Configuration for asset classification"""
    
    # Tier 1 criteria (50% allocation)
    tier1_market_cap_min: float = 50_000_000    # $50M USD
    tier1_market_cap_max: float = 400_000_000   # $400M USD
    tier1_daily_volume_min: float = 100_000     # R$100k BRL daily
    tier1_max_assets: int = 8
    tier1_allocation_target: float = 0.50
    
    # Tier 2 criteria (30% allocation)
    tier2_market_cap_min: float = 20_000_000    # $20M USD
    tier2_market_cap_max: float = 150_000_000   # $150M USD  
    tier2_daily_volume_min: float = 50_000      # R$50k BRL daily
    tier2_max_assets: int = 6
    tier2_allocation_target: float = 0.30
    
    # Tier 3 criteria (15% allocation)
    tier3_market_cap_min: float = 5_000_000     # $5M USD
    tier3_market_cap_max: float = 50_000_000    # $50M USD
    tier3_daily_volume_min: float = 25_000      # R$25k BRL daily
    tier3_max_assets: int = 4
    tier3_allocation_target: float = 0.15
    
    # Reserve allocation (5%)
    reserve_allocation_target: float = 0.05
    
    # General filters
    min_listing_age_days: int = 90              # Minimum 90 days listed
    min_data_quality_score: float = 0.7        # Minimum data quality
    max_volatility_filter: float = 3.0         # Max 300% annualized volatility
    
    # Stability requirements
    stability_window_days: int = 30             # Must meet criteria for 30 days
    liquidity_consistency_threshold: float = 0.5  # Volume consistency requirement
    
    # Exchange and regulatory filters
    required_exchange: str = "mercado_bitcoin"  # Must be on MB
    exclude_stablecoins_from_tiers: bool = True  # Stablecoins only in reserve
    
    # Brazilian market specific
    min_brl_volume_consistency: float = 0.7    # BRL volume consistency
    local_interest_weight: float = 0.1         # Weight for local trading interest


@dataclass
class AssetClassification:
    """Asset classification result"""
    symbol: str
    tier: AssetTier
    confidence: float
    classification_timestamp: datetime = field(default_factory=datetime.now)
    
    # Qualification metrics
    market_cap_usd: Optional[float] = None
    daily_volume_brl: Optional[float] = None
    volatility_annualized: Optional[float] = None
    liquidity_score: Optional[float] = None
    data_quality_score: Optional[float] = None
    
    # Tier-specific metrics
    tier_score: float = 0.0              # Score within tier
    tier_rank: Optional[int] = None      # Rank within tier
    allocation_weight: float = 0.0       # Suggested allocation weight
    
    # Qualification details
    meets_market_cap: bool = False
    meets_volume: bool = False
    meets_liquidity: bool = False
    meets_stability: bool = False
    meets_quality: bool = False
    
    # Disqualification reasons
    disqualification_reasons: List[str] = field(default_factory=list)
    
    # Historical context
    tier_history: List[AssetTier] = field(default_factory=list)
    tier_stability_days: int = 0


class AssetClassificationSystem:
    """
    Asset Classification System
    
    Systematically classifies assets into investment tiers based on
    quantitative criteria for portfolio construction and risk management.
    """
    
    def __init__(self, config: Optional[ClassificationConfig] = None):
        self.config = config or ClassificationConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # State tracking
        self._classification_history = {}
        self._market_data_cache = {}
        
        # Exchange rate for USD/BRL conversion (placeholder)
        self._usd_brl_rate = 5.0  # Should be updated with real FX data
        
        self.logger.info(f"📊 AssetClassificationSystem initialized")
        self.logger.info(f"  Tiers: T1={self.config.tier1_max_assets}, T2={self.config.tier2_max_assets}, T3={self.config.tier3_max_assets}")
        self.logger.info(f"  Allocations: T1={self.config.tier1_allocation_target:.0%}, T2={self.config.tier2_allocation_target:.0%}, T3={self.config.tier3_allocation_target:.0%}")
    
    def classify_assets(self, 
                       market_data: Dict[str, pd.DataFrame],
                       market_caps: Optional[Dict[str, float]] = None,
                       volume_data: Optional[Dict[str, float]] = None) -> Dict[str, AssetClassification]:
        """
        Classify all assets into appropriate tiers
        
        Args:
            market_data: Dictionary of OHLCV DataFrames by symbol
            market_caps: Optional market cap data in USD
            volume_data: Optional daily volume data in BRL
            
        Returns:
            Dictionary of AssetClassification objects by symbol
        """
        self.logger.info(f"📊 Classifying {len(market_data)} assets into tiers...")
        
        classifications = {}
        
        # Step 1: Calculate metrics for all assets
        asset_metrics = self._calculate_asset_metrics(market_data, market_caps, volume_data)
        
        # Step 2: Apply filters and classify each asset
        for symbol, metrics in asset_metrics.items():
            classification = self._classify_single_asset(symbol, metrics)
            classifications[symbol] = classification
        
        # Step 3: Apply tier limits and ranking
        classifications = self._apply_tier_limits(classifications)
        
        # Step 4: Calculate allocation weights
        classifications = self._calculate_allocation_weights(classifications)
        
        # Step 5: Update history
        self._update_classification_history(classifications)
        
        # Summary
        tier_counts = {}
        for tier in AssetTier:
            count = sum(1 for c in classifications.values() if c.tier == tier)
            tier_counts[tier.value] = count
        
        self.logger.info(f"✅ Classification completed:")
        for tier, count in tier_counts.items():
            if count > 0:
                self.logger.info(f"  {tier.upper()}: {count} assets")
        
        return classifications
    
    def _calculate_asset_metrics(self, 
                                market_data: Dict[str, pd.DataFrame],
                                market_caps: Optional[Dict[str, float]],
                                volume_data: Optional[Dict[str, float]]) -> Dict[str, Dict]:
        """Calculate metrics for each asset"""
        
        metrics = {}
        
        for symbol, data in market_data.items():
            if len(data) < self.config.min_listing_age_days:
                continue
            
            try:
                # Basic metrics from OHLCV data
                recent_data = data.tail(30)  # Last 30 days
                
                # Volatility calculation
                returns = data['close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252)  # Annualized
                
                # Volume metrics
                avg_volume_30d = recent_data['volume'].mean()
                volume_consistency = self._calculate_volume_consistency(data['volume'])
                
                # Price metrics
                current_price = data['close'].iloc[-1]
                
                # Market cap (from external data or estimate)
                market_cap = None
                if market_caps and symbol in market_caps:
                    market_cap = market_caps[symbol]
                else:
                    # Estimate if not provided (very rough estimate)
                    # This should ideally come from external data sources
                    market_cap = None
                
                # Daily volume in BRL
                daily_volume_brl = None
                if volume_data and symbol in volume_data:
                    daily_volume_brl = volume_data[symbol]
                else:
                    # Estimate from trading volume (price * volume)
                    daily_volume_brl = current_price * avg_volume_30d
                
                # Liquidity score
                liquidity_score = self._calculate_liquidity_score(data)
                
                # Data quality score
                data_quality = self._calculate_data_quality_score(data)
                
                # Listing age
                listing_age = len(data)  # Approximate
                
                metrics[symbol] = {
                    'volatility_annualized': volatility,
                    'daily_volume_brl': daily_volume_brl,
                    'market_cap_usd': market_cap,
                    'liquidity_score': liquidity_score,
                    'data_quality_score': data_quality,
                    'volume_consistency': volume_consistency,
                    'listing_age_days': listing_age,
                    'current_price': current_price,
                    'avg_volume_30d': avg_volume_30d
                }
                
            except Exception as e:
                self.logger.error(f"Error calculating metrics for {symbol}: {e}")
                continue
        
        return metrics
    
    def _classify_single_asset(self, symbol: str, metrics: Dict) -> AssetClassification:
        """Classify a single asset based on its metrics"""
        
        classification = AssetClassification(symbol=symbol, tier=AssetTier.EXCLUDED, confidence=0.0)
        
        # Extract metrics
        market_cap = metrics.get('market_cap_usd')
        daily_volume = metrics.get('daily_volume_brl', 0)
        volatility = metrics.get('volatility_annualized', 0)
        liquidity_score = metrics.get('liquidity_score', 0)
        data_quality = metrics.get('data_quality_score', 0)
        volume_consistency = metrics.get('volume_consistency', 0)
        listing_age = metrics.get('listing_age_days', 0)
        
        # Update classification metrics
        classification.market_cap_usd = market_cap
        classification.daily_volume_brl = daily_volume
        classification.volatility_annualized = volatility
        classification.liquidity_score = liquidity_score
        classification.data_quality_score = data_quality
        
        # Apply basic filters first
        disqualifications = []
        
        # Listing age filter
        if listing_age < self.config.min_listing_age_days:
            disqualifications.append(f"listing_age_too_short_{listing_age}d")
        
        # Data quality filter
        if data_quality < self.config.min_data_quality_score:
            disqualifications.append(f"data_quality_low_{data_quality:.2f}")
        
        # Volatility filter
        if volatility > self.config.max_volatility_filter:
            disqualifications.append(f"volatility_too_high_{volatility:.1f}")
        
        # Volume consistency filter
        if volume_consistency < self.config.liquidity_consistency_threshold:
            disqualifications.append(f"volume_inconsistent_{volume_consistency:.2f}")
        
        # Stablecoin filter for tiers (if enabled)
        if self.config.exclude_stablecoins_from_tiers and self._is_stablecoin(symbol):
            classification.tier = AssetTier.RESERVE
            classification.confidence = 0.9
            return classification
        
        # If disqualified, return excluded
        if disqualifications:
            classification.disqualification_reasons = disqualifications
            return classification
        
        # Market cap based tier assignment (if market cap available)
        if market_cap is not None:
            # Tier 1 check
            if (self.config.tier1_market_cap_min <= market_cap <= self.config.tier1_market_cap_max and
                daily_volume >= self.config.tier1_daily_volume_min):
                
                classification.tier = AssetTier.TIER_1
                classification.meets_market_cap = True
                classification.meets_volume = True
                
            # Tier 2 check
            elif (self.config.tier2_market_cap_min <= market_cap <= self.config.tier2_market_cap_max and
                  daily_volume >= self.config.tier2_daily_volume_min):
                
                classification.tier = AssetTier.TIER_2
                classification.meets_market_cap = True
                classification.meets_volume = True
                
            # Tier 3 check
            elif (self.config.tier3_market_cap_min <= market_cap <= self.config.tier3_market_cap_max and
                  daily_volume >= self.config.tier3_daily_volume_min):
                
                classification.tier = AssetTier.TIER_3
                classification.meets_market_cap = True
                classification.meets_volume = True
        
        # If no market cap data, use volume-based classification
        else:
            if daily_volume >= self.config.tier1_daily_volume_min:
                classification.tier = AssetTier.TIER_1
                classification.meets_volume = True
            elif daily_volume >= self.config.tier2_daily_volume_min:
                classification.tier = AssetTier.TIER_2
                classification.meets_volume = True
            elif daily_volume >= self.config.tier3_daily_volume_min:
                classification.tier = AssetTier.TIER_3
                classification.meets_volume = True
        
        # Additional qualification checks
        classification.meets_liquidity = liquidity_score >= 0.5
        classification.meets_stability = volume_consistency >= self.config.liquidity_consistency_threshold
        classification.meets_quality = data_quality >= self.config.min_data_quality_score
        
        # Calculate tier score
        if classification.tier != AssetTier.EXCLUDED:
            tier_score = self._calculate_tier_score(classification.tier, metrics)
            classification.tier_score = tier_score
            
            # Confidence based on meeting criteria
            criteria_met = sum([
                classification.meets_market_cap,
                classification.meets_volume,
                classification.meets_liquidity,
                classification.meets_stability,
                classification.meets_quality
            ])
            classification.confidence = criteria_met / 5.0
        
        return classification
    
    def _calculate_liquidity_score(self, data: pd.DataFrame) -> float:
        """Calculate liquidity score based on volume patterns"""
        
        if len(data) < 30:
            return 0.0
        
        # Volume metrics
        volume = data['volume']
        volume_sma = volume.rolling(window=30).mean()
        volume_std = volume.rolling(window=30).std()
        
        # Recent volume vs historical
        recent_volume = volume.tail(7).mean()
        historical_volume = volume_sma.iloc[-1]
        
        if historical_volume == 0:
            return 0.0
        
        # Volume consistency (lower coefficient of variation = higher liquidity)
        cv = volume_std.iloc[-1] / historical_volume if historical_volume > 0 else 2.0
        consistency_score = max(0, 1 - cv)
        
        # Volume magnitude (log scale)
        magnitude_score = min(np.log1p(historical_volume) / 20, 1.0)  # Scaled
        
        # Combined liquidity score
        liquidity_score = (consistency_score * 0.6 + magnitude_score * 0.4)
        
        return np.clip(liquidity_score, 0.0, 1.0)
    
    def _calculate_data_quality_score(self, data: pd.DataFrame) -> float:
        """Calculate data quality score"""
        
        if len(data) == 0:
            return 0.0
        
        quality_factors = []
        
        # Completeness (no missing values)
        completeness = 1.0 - (data.isna().sum().sum() / (len(data) * len(data.columns)))
        quality_factors.append(completeness)
        
        # Price consistency (no extreme jumps)
        price_returns = data['close'].pct_change().dropna()
        if len(price_returns) > 0:
            extreme_moves = (abs(price_returns) > 0.5).sum()  # >50% daily moves
            price_consistency = max(0, 1 - extreme_moves / len(price_returns) * 10)
            quality_factors.append(price_consistency)
        
        # Volume consistency
        volume_returns = data['volume'].pct_change().dropna()
        if len(volume_returns) > 0:
            zero_volume_days = (data['volume'] == 0).sum()
            volume_consistency = max(0, 1 - zero_volume_days / len(data) * 5)
            quality_factors.append(volume_consistency)
        
        # OHLC consistency (High >= Low, etc.)
        if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            ohlc_valid = ((data['high'] >= data['low']) & 
                         (data['high'] >= data['open']) & 
                         (data['high'] >= data['close']) &
                         (data['low'] <= data['open']) & 
                         (data['low'] <= data['close'])).sum()
            ohlc_consistency = ohlc_valid / len(data)
            quality_factors.append(ohlc_consistency)
        
        return np.mean(quality_factors) if quality_factors else 0.0
    
    def _calculate_volume_consistency(self, volume_series: pd.Series) -> float:
        """Calculate volume consistency score"""
        
        if len(volume_series) < 30:
            return 0.0
        
        # Remove zero volume days
        non_zero_volume = volume_series[volume_series > 0]
        
        if len(non_zero_volume) < 20:
            return 0.0
        
        # Coefficient of variation
        mean_vol = non_zero_volume.mean()
        std_vol = non_zero_volume.std()
        
        if mean_vol == 0:
            return 0.0
        
        cv = std_vol / mean_vol
        
        # Convert to consistency score (lower CV = higher consistency)
        consistency = max(0, 1 - cv)
        
        return consistency
    
    def _calculate_tier_score(self, tier: AssetTier, metrics: Dict) -> float:
        """Calculate score within tier for ranking"""
        
        score_components = []
        
        # Volume score
        daily_volume = metrics.get('daily_volume_brl', 0)
        if tier == AssetTier.TIER_1:
            volume_score = min(daily_volume / self.config.tier1_daily_volume_min, 5.0) / 5.0
        elif tier == AssetTier.TIER_2:
            volume_score = min(daily_volume / self.config.tier2_daily_volume_min, 5.0) / 5.0
        else:  # TIER_3
            volume_score = min(daily_volume / self.config.tier3_daily_volume_min, 5.0) / 5.0
        score_components.append(volume_score)
        
        # Liquidity score
        liquidity_score = metrics.get('liquidity_score', 0)
        score_components.append(liquidity_score)
        
        # Quality score
        quality_score = metrics.get('data_quality_score', 0)
        score_components.append(quality_score)
        
        # Volatility score (lower volatility = higher score for T1, reverse for T3)
        volatility = metrics.get('volatility_annualized', 1.0)
        if tier == AssetTier.TIER_1:
            vol_score = max(0, 1 - volatility / 2.0)  # Prefer lower volatility
        else:  # TIER_2, TIER_3
            vol_score = min(volatility / 2.0, 1.0)    # Accept higher volatility
        score_components.append(vol_score)
        
        return np.mean(score_components)
    
    def _is_stablecoin(self, symbol: str) -> bool:
        """Check if asset is a stablecoin"""
        stablecoin_indicators = ['USD', 'USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'STABLE']
        return any(indicator in symbol.upper() for indicator in stablecoin_indicators)
    
    def _apply_tier_limits(self, classifications: Dict[str, AssetClassification]) -> Dict[str, AssetClassification]:
        """Apply tier asset limits and ranking"""
        
        # Group by tier
        tier_assets = {tier: [] for tier in AssetTier}
        for classification in classifications.values():
            tier_assets[classification.tier].append(classification)
        
        # Sort each tier by tier_score and apply limits
        for tier in [AssetTier.TIER_1, AssetTier.TIER_2, AssetTier.TIER_3]:
            tier_list = tier_assets[tier]
            tier_list.sort(key=lambda x: x.tier_score, reverse=True)
            
            # Determine tier limit
            if tier == AssetTier.TIER_1:
                max_assets = self.config.tier1_max_assets
            elif tier == AssetTier.TIER_2:
                max_assets = self.config.tier2_max_assets
            else:  # TIER_3
                max_assets = self.config.tier3_max_assets
            
            # Keep top assets, demote others
            for i, asset in enumerate(tier_list):
                if i < max_assets:
                    asset.tier_rank = i + 1
                else:
                    # Demote to lower tier or exclude
                    if tier == AssetTier.TIER_1:
                        asset.tier = AssetTier.TIER_2
                    elif tier == AssetTier.TIER_2:
                        asset.tier = AssetTier.TIER_3
                    else:
                        asset.tier = AssetTier.EXCLUDED
                        asset.disqualification_reasons.append("tier_limit_exceeded")
        
        return classifications
    
    def _calculate_allocation_weights(self, classifications: Dict[str, AssetClassification]) -> Dict[str, AssetClassification]:
        """Calculate allocation weights for each asset"""
        
        # Count assets by tier
        tier_counts = {tier: 0 for tier in AssetTier}
        for classification in classifications.values():
            tier_counts[classification.tier] += 1
        
        # Calculate weights
        for classification in classifications.values():
            tier = classification.tier
            
            if tier == AssetTier.TIER_1 and tier_counts[tier] > 0:
                classification.allocation_weight = self.config.tier1_allocation_target / tier_counts[tier]
            elif tier == AssetTier.TIER_2 and tier_counts[tier] > 0:
                classification.allocation_weight = self.config.tier2_allocation_target / tier_counts[tier]
            elif tier == AssetTier.TIER_3 and tier_counts[tier] > 0:
                classification.allocation_weight = self.config.tier3_allocation_target / tier_counts[tier]
            elif tier == AssetTier.RESERVE:
                classification.allocation_weight = self.config.reserve_allocation_target
            else:
                classification.allocation_weight = 0.0
        
        return classifications
    
    def _update_classification_history(self, classifications: Dict[str, AssetClassification]):
        """Update classification history for stability tracking"""
        
        for symbol, classification in classifications.items():
            if symbol not in self._classification_history:
                self._classification_history[symbol] = []
            
            # Add current classification to history
            self._classification_history[symbol].append(classification.tier)
            
            # Keep only recent history
            if len(self._classification_history[symbol]) > 90:  # 90 days
                self._classification_history[symbol].pop(0)
            
            # Calculate tier stability
            recent_tiers = self._classification_history[symbol][-self.config.stability_window_days:]
            if len(recent_tiers) >= self.config.stability_window_days:
                current_tier = classification.tier
                stable_days = sum(1 for t in reversed(recent_tiers) if t == current_tier)
                classification.tier_stability_days = stable_days
            
            # Set tier history
            classification.tier_history = recent_tiers.copy()
    
    def get_tier_summary(self, classifications: Dict[str, AssetClassification]) -> Dict[str, any]:
        """Get summary of tier classifications"""
        
        if not classifications:
            return {'error': 'No classifications available'}
        
        tier_summary = {}
        
        for tier in AssetTier:
            tier_assets = [c for c in classifications.values() if c.tier == tier]
            
            if tier_assets:
                # Calculate tier statistics
                tier_scores = [a.tier_score for a in tier_assets if a.tier_score > 0]
                allocation_weights = [a.allocation_weight for a in tier_assets]
                
                tier_summary[tier.value] = {
                    'count': len(tier_assets),
                    'total_allocation': sum(allocation_weights),
                    'avg_tier_score': np.mean(tier_scores) if tier_scores else 0,
                    'assets': [{'symbol': a.symbol, 
                              'tier_score': a.tier_score,
                              'allocation_weight': a.allocation_weight,
                              'tier_rank': a.tier_rank} for a in tier_assets]
                }
        
        return tier_summary
    
    def get_portfolio_allocation(self, classifications: Dict[str, AssetClassification]) -> Dict[str, float]:
        """Get final portfolio allocation weights"""
        
        allocation = {}
        for symbol, classification in classifications.items():
            if classification.allocation_weight > 0:
                allocation[symbol] = classification.allocation_weight
        
        # Normalize to ensure sum = 1.0
        total_weight = sum(allocation.values())
        if total_weight > 0:
            allocation = {k: v/total_weight for k, v in allocation.items()}
        
        return allocation


# Asset Classification System is integrated into the main DCA system
# Use the main CLI for testing: python cli.py run
