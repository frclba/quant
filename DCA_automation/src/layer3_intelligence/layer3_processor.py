#!/usr/bin/env python3
"""
Layer 3 Intelligence Processor

Main orchestrator for Layer 3: Intelligence & Scoring.
Coordinates multi-factor scoring, regime detection, asset classification,
and predictive signal generation for intelligent investment decisions.

Integrates:
- MultiFactorScoringEngine: Asset ranking and selection
- MarketRegimeDetector: Market condition classification  
- AssetClassificationSystem: Tier-based portfolio structure
- PredictiveSignalEngine: Forward-looking market signals
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import asyncio

from .scoring_engine import MultiFactorScoringEngine, ScoringConfig, AssetScore
from .regime_detector import MarketRegimeDetector, RegimeConfig, MarketRegime
from .asset_classifier import AssetClassificationSystem, ClassificationConfig, AssetClassification
from .signal_engine import PredictiveSignalEngine, SignalConfig, PredictiveSignal

logger = logging.getLogger(__name__)


@dataclass
class Layer3Config:
    """Configuration for Layer 3 Intelligence processing"""
    
    # Component configurations
    scoring_config: Optional[ScoringConfig] = None
    regime_config: Optional[RegimeConfig] = None
    classification_config: Optional[ClassificationConfig] = None
    signal_config: Optional[SignalConfig] = None
    
    # Processing options
    enable_scoring: bool = True
    enable_regime_detection: bool = True
    enable_classification: bool = True
    enable_signals: bool = True
    
    # Integration settings
    min_assets_for_processing: int = 10
    benchmark_symbol: str = 'BTC'
    
    # Output settings
    max_tier1_assets: int = 8
    max_tier2_assets: int = 6
    max_tier3_assets: int = 4
    min_composite_score: float = 0.3
    
    # Performance settings
    parallel_processing: bool = True
    cache_results: bool = True
    result_expiry_minutes: int = 60


@dataclass
class IntelligenceResult:
    """Complete Layer 3 intelligence analysis result"""
    
    # Core results
    asset_scores: Dict[str, AssetScore] = field(default_factory=dict)
    market_regime: Optional[MarketRegime] = None
    asset_classifications: Dict[str, AssetClassification] = field(default_factory=dict)
    predictive_signals: Dict[str, List[PredictiveSignal]] = field(default_factory=dict)
    
    # Processing metadata
    processing_timestamp: datetime = field(default_factory=datetime.now)
    assets_processed: int = 0
    processing_duration: Optional[timedelta] = None
    
    # Investment recommendations
    recommended_portfolio: Dict[str, float] = field(default_factory=dict)  # symbol -> weight
    tier_allocations: Dict[str, List[str]] = field(default_factory=dict)   # tier -> symbols
    regime_adjusted_weights: Dict[str, float] = field(default_factory=dict)
    
    # Intelligence insights
    market_outlook: str = ""
    key_opportunities: List[str] = field(default_factory=list)
    risk_alerts: List[str] = field(default_factory=list)
    
    # Performance metrics
    processing_stats: Dict[str, any] = field(default_factory=dict)


class Layer3IntelligenceProcessor:
    """
    Layer 3 Intelligence Processor
    
    Main coordinator for intelligence generation, combining multiple
    analytical engines to provide comprehensive market intelligence
    and investment recommendations.
    """
    
    def __init__(self, config: Optional[Layer3Config] = None):
        self.config = config or Layer3Config()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize engines based on configuration
        self.scoring_engine = None
        self.regime_detector = None
        self.classification_system = None
        self.signal_engine = None
        
        if self.config.enable_scoring:
            self.scoring_engine = MultiFactorScoringEngine(self.config.scoring_config)
        
        if self.config.enable_regime_detection:
            self.regime_detector = MarketRegimeDetector(self.config.regime_config)
        
        if self.config.enable_classification:
            self.classification_system = AssetClassificationSystem(self.config.classification_config)
        
        if self.config.enable_signals:
            self.signal_engine = PredictiveSignalEngine(self.config.signal_config)
        
        # Result cache
        self._result_cache = {}
        
        self.logger.info(f"🧠 Layer3IntelligenceProcessor initialized")
        enabled_engines = [
            "Scoring" if self.scoring_engine else None,
            "Regime" if self.regime_detector else None, 
            "Classification" if self.classification_system else None,
            "Signals" if self.signal_engine else None
        ]
        enabled_engines = [e for e in enabled_engines if e]
        self.logger.info(f"  Engines: {', '.join(enabled_engines)}")
    
    async def process_intelligence(self, 
                                 market_data: Dict[str, pd.DataFrame],
                                 market_caps: Optional[Dict[str, float]] = None,
                                 volume_data: Optional[Dict[str, float]] = None) -> IntelligenceResult:
        """
        Process complete intelligence analysis
        
        Args:
            market_data: Dictionary of OHLCV DataFrames by symbol
            market_caps: Optional market cap data in USD
            volume_data: Optional daily volume data in BRL
            
        Returns:
            IntelligenceResult with complete analysis
        """
        start_time = datetime.now()
        
        self.logger.info(f"🧠 Processing intelligence for {len(market_data)} assets...")
        
        if len(market_data) < self.config.min_assets_for_processing:
            raise ValueError(f"Need at least {self.config.min_assets_for_processing} assets for processing")
        
        # Initialize result
        result = IntelligenceResult(assets_processed=len(market_data))
        
        try:
            # Process components in parallel if enabled
            if self.config.parallel_processing:
                tasks = []
                
                if self.scoring_engine:
                    tasks.append(self._process_scoring(market_data))
                
                if self.regime_detector:
                    tasks.append(self._process_regime_detection(market_data))
                
                if self.classification_system:
                    tasks.append(self._process_classification(market_data, market_caps, volume_data))
                
                if self.signal_engine:
                    tasks.append(self._process_signals(market_data))
                
                # Execute all tasks
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for i, task_result in enumerate(results):
                    if isinstance(task_result, Exception):
                        self.logger.error(f"Task {i} failed: {task_result}")
                    else:
                        # Assign results based on task order
                        if i == 0 and self.scoring_engine:
                            result.asset_scores = task_result
                        elif (i == 1 if self.scoring_engine else i == 0) and self.regime_detector:
                            result.market_regime = task_result
                        # And so on...
            
            else:
                # Sequential processing
                if self.scoring_engine:
                    result.asset_scores = await self._process_scoring(market_data)
                
                if self.regime_detector:
                    benchmark_data = market_data.get(self.config.benchmark_symbol)
                    if benchmark_data is not None:
                        result.market_regime = await self._process_regime_detection(market_data)
                
                if self.classification_system:
                    result.asset_classifications = await self._process_classification(market_data, market_caps, volume_data)
                
                if self.signal_engine:
                    result.predictive_signals = await self._process_signals(market_data)
            
            # Generate integrated recommendations
            result = await self._generate_recommendations(result, market_data)
            
            # Calculate processing stats
            end_time = datetime.now()
            result.processing_duration = end_time - start_time
            result.processing_stats = self._calculate_processing_stats(result)
            
            self.logger.info(f"✅ Intelligence processing completed in {result.processing_duration}")
            
            # 🎨 GENERATE VISUALIZATIONS
            try:
                from ..visualization import PortfolioVisualizer
                visualizer = PortfolioVisualizer()
                
                self.logger.info("🎨 Generating Layer 3 intelligence visualizations...")
                layer3_plots = visualizer.visualize_layer3_intelligence(result, "layer3")
                
                if layer3_plots:
                    self.logger.info(f"✅ Created {len(layer3_plots)} Layer 3 visualizations:")
                    for plot_type, file_path in layer3_plots.items():
                        if file_path:
                            self.logger.info(f"  📊 {plot_type}: {file_path}")
                            
            except Exception as e:
                self.logger.error(f"Failed to create Layer 3 visualizations: {e}")
            
            # Cache result if enabled
            if self.config.cache_results:
                self._cache_result(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Intelligence processing failed: {e}")
            raise
    
    async def _process_scoring(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, AssetScore]:
        """Process multi-factor asset scoring"""
        self.logger.debug("🎯 Processing asset scoring...")
        
        return self.scoring_engine.calculate_asset_scores(
            market_data, 
            benchmark_symbol=self.config.benchmark_symbol
        )
    
    async def _process_regime_detection(self, market_data: Dict[str, pd.DataFrame]) -> Optional[MarketRegime]:
        """Process market regime detection"""
        self.logger.debug("🔍 Processing regime detection...")
        
        # Use benchmark data for regime detection
        benchmark_data = market_data.get(self.config.benchmark_symbol)
        if benchmark_data is None:
            self.logger.warning(f"Benchmark {self.config.benchmark_symbol} not found for regime detection")
            return None
        
        return self.regime_detector.detect_regime(benchmark_data)
    
    async def _process_classification(self, 
                                    market_data: Dict[str, pd.DataFrame],
                                    market_caps: Optional[Dict[str, float]],
                                    volume_data: Optional[Dict[str, float]]) -> Dict[str, AssetClassification]:
        """Process asset classification"""
        self.logger.debug("📊 Processing asset classification...")
        
        return self.classification_system.classify_assets(
            market_data, 
            market_caps, 
            volume_data
        )
    
    async def _process_signals(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, List[PredictiveSignal]]:
        """Process predictive signals"""
        self.logger.debug("🔮 Processing predictive signals...")
        
        return self.signal_engine.generate_signals(
            market_data, 
            benchmark_symbol=self.config.benchmark_symbol
        )
    
    async def _generate_recommendations(self, 
                                      result: IntelligenceResult, 
                                      market_data: Dict[str, pd.DataFrame]) -> IntelligenceResult:
        """Generate integrated investment recommendations"""
        self.logger.debug("💡 Generating recommendations...")
        
        # 1. Portfolio recommendations based on scores and classifications
        if result.asset_scores and result.asset_classifications:
            result.recommended_portfolio = self._generate_portfolio_weights(
                result.asset_scores, 
                result.asset_classifications
            )
        
        # 2. Tier allocations
        if result.asset_classifications:
            result.tier_allocations = self._generate_tier_allocations(result.asset_classifications)
        
        # 3. Regime-adjusted weights
        if result.market_regime and result.recommended_portfolio:
            result.regime_adjusted_weights = self._adjust_weights_for_regime(
                result.recommended_portfolio, 
                result.market_regime
            )
        
        # 4. Market outlook and insights
        result.market_outlook = self._generate_market_outlook(result)
        result.key_opportunities = self._identify_opportunities(result)
        result.risk_alerts = self._identify_risks(result)
        
        return result
    
    def _generate_portfolio_weights(self, 
                                  asset_scores: Dict[str, AssetScore],
                                  asset_classifications: Dict[str, AssetClassification]) -> Dict[str, float]:
        """Generate portfolio weights based on scores and classifications"""
        
        # Filter assets that meet minimum criteria
        qualified_assets = {}
        
        for symbol, classification in asset_classifications.items():
            if (classification.tier.value in ['tier_1', 'tier_2', 'tier_3'] and
                symbol in asset_scores and
                asset_scores[symbol].composite_score >= self.config.min_composite_score):
                
                qualified_assets[symbol] = {
                    'score': asset_scores[symbol].composite_score,
                    'tier': classification.tier.value,
                    'allocation_weight': classification.allocation_weight
                }
        
        if not qualified_assets:
            return {}
        
        # Calculate weights combining classification allocation and scores
        portfolio_weights = {}
        
        # Group by tier
        tier_groups = {'tier_1': [], 'tier_2': [], 'tier_3': []}
        for symbol, data in qualified_assets.items():
            tier_groups[data['tier']].append((symbol, data))
        
        # Allocate within each tier based on scores
        for tier, assets in tier_groups.items():
            if not assets:
                continue
            
            # Sort by score within tier
            assets.sort(key=lambda x: x[1]['score'], reverse=True)
            
            # Apply tier limits
            if tier == 'tier_1':
                assets = assets[:self.config.max_tier1_assets]
            elif tier == 'tier_2':
                assets = assets[:self.config.max_tier2_assets]
            elif tier == 'tier_3':
                assets = assets[:self.config.max_tier3_assets]
            
            # Calculate weights within tier (score-weighted)
            total_score = sum(asset[1]['score'] for asset in assets)
            tier_allocation = assets[0][1]['allocation_weight'] * len(assets) if assets else 0
            
            for symbol, data in assets:
                score_weight = data['score'] / total_score if total_score > 0 else 1.0 / len(assets)
                portfolio_weights[symbol] = tier_allocation * score_weight
        
        # Normalize weights to sum to 1.0 (excluding reserve allocation)
        total_weight = sum(portfolio_weights.values())
        if total_weight > 0:
            target_allocation = 0.95  # Leave 5% for reserve
            portfolio_weights = {k: v * target_allocation / total_weight 
                               for k, v in portfolio_weights.items()}
        
        return portfolio_weights
    
    def _generate_tier_allocations(self, asset_classifications: Dict[str, AssetClassification]) -> Dict[str, List[str]]:
        """Generate tier-based asset allocations"""
        
        tier_allocations = {
            'tier_1': [],
            'tier_2': [],
            'tier_3': [],
            'reserve': []
        }
        
        for symbol, classification in asset_classifications.items():
            tier_key = classification.tier.value
            if tier_key in tier_allocations:
                tier_allocations[tier_key].append(symbol)
        
        # Sort each tier by tier_rank
        for tier in tier_allocations:
            tier_assets = [(symbol, asset_classifications[symbol]) for symbol in tier_allocations[tier]]
            tier_assets.sort(key=lambda x: x[1].tier_rank or 999)  # None ranks last
            tier_allocations[tier] = [symbol for symbol, _ in tier_assets]
        
        return tier_allocations
    
    def _adjust_weights_for_regime(self, 
                                 portfolio_weights: Dict[str, float],
                                 market_regime: MarketRegime) -> Dict[str, float]:
        """Adjust portfolio weights based on market regime"""
        
        adjusted_weights = portfolio_weights.copy()
        
        # Regime-based adjustments
        if market_regime.regime.value == 'bull':
            # Bull market: Increase risk assets (Tier 2, 3)
            risk_multiplier = 1 + (market_regime.confidence * 0.2)
            safe_multiplier = 1 - (market_regime.confidence * 0.1)
            
        elif market_regime.regime.value == 'bear':
            # Bear market: Decrease risk assets, increase safe assets
            risk_multiplier = 1 - (market_regime.confidence * 0.3)
            safe_multiplier = 1 + (market_regime.confidence * 0.2)
            
        elif market_regime.regime.value == 'volatile':
            # Volatile market: Reduce all allocations, increase cash
            risk_multiplier = 0.7
            safe_multiplier = 0.8
            
        else:  # sideways or unknown
            # No adjustment
            return adjusted_weights
        
        # Apply adjustments (placeholder - would need tier information)
        # This is simplified - real implementation would adjust based on asset tiers
        total_adjustment = sum(adjusted_weights.values()) * 0.1  # Max 10% adjustment
        
        for symbol in adjusted_weights:
            # Simplified regime adjustment
            adjusted_weights[symbol] *= 0.95  # Slight reduction to create cash buffer
        
        return adjusted_weights
    
    def _generate_market_outlook(self, result: IntelligenceResult) -> str:
        """Generate market outlook narrative"""
        
        outlook_parts = []
        
        # Regime-based outlook
        if result.market_regime:
            regime = result.market_regime.regime.value.upper()
            confidence = result.market_regime.confidence
            outlook_parts.append(f"Market regime: {regime} (confidence: {confidence:.1%})")
        
        # Scoring insights
        if result.asset_scores:
            top_scores = sorted(result.asset_scores.values(), 
                              key=lambda x: x.composite_score, reverse=True)[:5]
            avg_score = np.mean([score.composite_score for score in result.asset_scores.values()])
            outlook_parts.append(f"Average asset score: {avg_score:.3f}")
            outlook_parts.append(f"Top opportunities: {', '.join([s.symbol for s in top_scores])}")
        
        # Signal insights
        if result.predictive_signals:
            bullish_signals = sum(1 for signals in result.predictive_signals.values() 
                                for s in signals if s.direction.value == 'bullish')
            bearish_signals = sum(1 for signals in result.predictive_signals.values() 
                                for s in signals if s.direction.value == 'bearish')
            outlook_parts.append(f"Signal sentiment: {bullish_signals} bullish, {bearish_signals} bearish")
        
        return ". ".join(outlook_parts) + "."
    
    def _identify_opportunities(self, result: IntelligenceResult) -> List[str]:
        """Identify key opportunities"""
        
        opportunities = []
        
        # High-scoring assets
        if result.asset_scores:
            top_assets = sorted(result.asset_scores.values(), 
                              key=lambda x: x.composite_score, reverse=True)[:3]
            for asset in top_assets:
                if asset.composite_score > 0.7:
                    opportunities.append(f"{asset.symbol}: High composite score ({asset.composite_score:.3f})")
        
        # Strong bullish signals
        if result.predictive_signals:
            for symbol, signals in result.predictive_signals.items():
                strong_bullish = [s for s in signals if s.direction.value == 'bullish' and s.confidence > 0.7]
                if strong_bullish:
                    opportunities.append(f"{symbol}: Strong bullish signals ({len(strong_bullish)} signals)")
        
        return opportunities[:5]  # Limit to top 5
    
    def _identify_risks(self, result: IntelligenceResult) -> List[str]:
        """Identify key risks"""
        
        risks = []
        
        # Regime-based risks
        if result.market_regime:
            if result.market_regime.regime.value in ['bear', 'volatile']:
                risks.append(f"Market regime risk: {result.market_regime.regime.value} market")
            
            if result.market_regime.volatility_signal > 0.8:
                risks.append("High volatility environment")
        
        # Classification risks
        if result.asset_classifications:
            excluded_count = sum(1 for c in result.asset_classifications.values() 
                               if c.tier.value == 'excluded')
            if excluded_count > len(result.asset_classifications) * 0.5:
                risks.append(f"Many assets excluded from investment universe ({excluded_count})")
        
        # Signal risks
        if result.predictive_signals:
            strong_bearish_count = sum(1 for signals in result.predictive_signals.values() 
                                     for s in signals if s.direction.value == 'bearish' and s.confidence > 0.6)
            if strong_bearish_count > 5:
                risks.append(f"Multiple strong bearish signals ({strong_bearish_count})")
        
        return risks[:5]  # Limit to top 5
    
    def _calculate_processing_stats(self, result: IntelligenceResult) -> Dict[str, any]:
        """Calculate processing statistics"""
        
        stats = {
            'assets_processed': result.assets_processed,
            'processing_duration_seconds': result.processing_duration.total_seconds() if result.processing_duration else 0,
        }
        
        if result.asset_scores:
            stats['assets_scored'] = len(result.asset_scores)
            stats['avg_composite_score'] = np.mean([s.composite_score for s in result.asset_scores.values()])
        
        if result.asset_classifications:
            tier_counts = {}
            for c in result.asset_classifications.values():
                tier_counts[c.tier.value] = tier_counts.get(c.tier.value, 0) + 1
            stats['tier_distribution'] = tier_counts
        
        if result.predictive_signals:
            stats['total_signals'] = sum(len(signals) for signals in result.predictive_signals.values())
            stats['assets_with_signals'] = len(result.predictive_signals)
        
        if result.recommended_portfolio:
            stats['portfolio_assets'] = len(result.recommended_portfolio)
            stats['total_allocation'] = sum(result.recommended_portfolio.values())
        
        return stats
    
    def _cache_result(self, result: IntelligenceResult):
        """Cache processing result"""
        cache_key = f"intelligence_{result.processing_timestamp.strftime('%Y%m%d_%H%M')}"
        self._result_cache[cache_key] = result
        
        # Clean old cache entries
        cutoff_time = datetime.now() - timedelta(minutes=self.config.result_expiry_minutes * 2)
        self._result_cache = {k: v for k, v in self._result_cache.items() 
                            if v.processing_timestamp > cutoff_time}
    
    def get_processing_summary(self, result: IntelligenceResult) -> Dict[str, any]:
        """Get summary of processing results"""
        
        if not result:
            return {'error': 'No results available'}
        
        summary = {
            'processing_timestamp': result.processing_timestamp.isoformat(),
            'assets_processed': result.assets_processed,
            'processing_duration': str(result.processing_duration),
            'market_outlook': result.market_outlook,
            'key_opportunities': result.key_opportunities,
            'risk_alerts': result.risk_alerts,
            'processing_stats': result.processing_stats
        }
        
        # Portfolio summary
        if result.recommended_portfolio:
            sorted_portfolio = sorted(result.recommended_portfolio.items(), 
                                    key=lambda x: x[1], reverse=True)
            summary['top_portfolio_positions'] = sorted_portfolio[:10]
            summary['total_portfolio_allocation'] = sum(result.recommended_portfolio.values())
        
        # Tier summary
        if result.tier_allocations:
            summary['tier_allocations'] = {k: len(v) for k, v in result.tier_allocations.items()}
        
        return summary


# Process Layer 2 data through Layer 3 intelligence
async def process_layer2_intelligence(layer2_data_path: str, 
                                    config: Optional[Layer3Config] = None) -> IntelligenceResult:
    """
    Process Layer 2 data through Layer 3 intelligence engines
    
    Args:
        layer2_data_path: Path to Layer 2 processed data
        config: Optional Layer 3 configuration
        
    Returns:
        IntelligenceResult with complete analysis
    """
    processor = Layer3IntelligenceProcessor(config)
    
    # Load market data (placeholder - would load from Layer 2 output)
    # This would typically load processed data from Layer 2
    market_data = {}  # Load from layer2_data_path
    
    # Process intelligence
    return await processor.process_intelligence(market_data)


# Layer 3 Intelligence is integrated into the main DCA system
# Use the main CLI for testing: python cli.py run
