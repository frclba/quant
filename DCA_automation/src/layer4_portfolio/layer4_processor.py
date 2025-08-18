#!/usr/bin/env python3
"""
Layer 4 Portfolio Processor

Main orchestrator for portfolio optimization using Modern Portfolio Theory.
Integrates Markowitz optimization, risk-adjusted allocation, and rebalancing.

Key Features:
- Efficient Frontier Construction
- Multi-objective Optimization (Return, Risk, Diversification)
- Tier-based Allocation Constraints
- Dynamic Rebalancing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class Layer4Config:
    """Configuration for Layer 4 Portfolio Optimization"""
    
    # Optimization objectives
    target_return: Optional[float] = None      # Target annual return (None for max Sharpe)
    risk_aversion: float = 3.0                 # Risk aversion parameter
    max_volatility: float = 0.8                # Maximum portfolio volatility
    
    # Constraints
    max_individual_weight: float = 0.05        # 5% max per asset
    min_individual_weight: float = 0.005       # 0.5% min per asset (if included)
    max_tier_concentration: float = 0.6        # 60% max in any tier
    min_portfolio_assets: int = 8              # Minimum diversification
    
    # Tier allocations (from architecture)
    tier1_allocation: Tuple[float, float] = (0.45, 0.55)  # 45-55%
    tier2_allocation: Tuple[float, float] = (0.25, 0.35)  # 25-35%
    tier3_allocation: Tuple[float, float] = (0.10, 0.20)  # 10-20%
    reserve_allocation: float = 0.05                       # 5% reserve
    
    # Rebalancing parameters
    rebalance_threshold: float = 0.25          # 25% drift threshold
    min_trade_size: float = 0.001              # 0.1% minimum trade
    max_turnover: float = 0.20                 # 20% monthly turnover limit
    
    # Risk management
    enable_risk_budgeting: bool = True
    equal_risk_contribution: bool = False
    max_correlation_exposure: float = 0.3      # Max allocation to correlated assets


@dataclass
class OptimizationResult:
    """Portfolio optimization result"""
    
    # Core results
    optimal_weights: Dict[str, float] = field(default_factory=dict)
    expected_return: float = 0.0
    expected_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    
    # Portfolio metrics
    diversification_ratio: float = 0.0
    maximum_drawdown_estimate: float = 0.0
    var_95: float = 0.0
    
    # Optimization details
    optimization_method: str = "max_sharpe"
    constraints_satisfied: bool = True
    convergence_status: str = "success"
    
    # Rebalancing
    rebalance_actions: List[str] = field(default_factory=list)
    estimated_transaction_costs: float = 0.0
    
    # Metadata
    optimization_timestamp: datetime = field(default_factory=datetime.now)
    assets_included: int = 0
    tier_allocations: Dict[str, float] = field(default_factory=dict)


class Layer4PortfolioProcessor:
    """
    Layer 4 Portfolio Processor
    
    Implements Modern Portfolio Theory optimization with tier-based constraints
    and dynamic rebalancing for the DCA system.
    """
    
    def __init__(self, config: Optional[Layer4Config] = None):
        self.config = config or Layer4Config()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Portfolio state
        self._current_portfolio = {}
        self._historical_performance = []
        
        self.logger.info(f"📈 Layer4PortfolioProcessor initialized")
        self.logger.info(f"  Max individual weight: {self.config.max_individual_weight:.1%}")
        self.logger.info(f"  Target tier allocations: T1={self.config.tier1_allocation}, T2={self.config.tier2_allocation}, T3={self.config.tier3_allocation}")
    
    async def optimize_portfolio(self, 
                                intelligence_data: Dict,
                                market_data: Dict[str, pd.DataFrame],
                                current_positions: Optional[Dict[str, float]] = None) -> OptimizationResult:
        """
        Optimize portfolio allocation based on intelligence insights
        
        Args:
            intelligence_data: Results from Layer 3 intelligence
            market_data: Market data for return/risk estimation
            current_positions: Current portfolio positions
            
        Returns:
            OptimizationResult with optimal allocations
        """
        self.logger.info(f"📈 Optimizing portfolio with {len(market_data)} assets...")
        
        result = OptimizationResult()
        
        try:
            # 1. Prepare optimization inputs
            expected_returns, covariance_matrix, asset_tiers = self._prepare_optimization_inputs(
                intelligence_data, market_data
            )
            
            if expected_returns is None or len(expected_returns) < self.config.min_portfolio_assets:
                self.logger.warning("Insufficient assets for optimization")
                return result
            
            # 2. Run optimization
            optimal_weights = self._optimize_allocation(
                expected_returns, covariance_matrix, asset_tiers
            )
            
            # 3. Calculate portfolio metrics
            portfolio_return = np.sum(optimal_weights * expected_returns)
            portfolio_risk = np.sqrt(np.dot(optimal_weights, np.dot(covariance_matrix, optimal_weights)))
            sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
            
            # 4. Generate rebalancing actions
            rebalance_actions = []
            if current_positions:
                rebalance_actions = self._generate_rebalance_actions(
                    current_positions, optimal_weights
                )
            
            # 5. Populate result
            result.optimal_weights = {asset: weight for asset, weight in zip(expected_returns.index, optimal_weights) if weight > 0.001}
            result.expected_return = portfolio_return
            result.expected_volatility = portfolio_risk
            result.sharpe_ratio = sharpe_ratio
            result.assets_included = len(result.optimal_weights)
            result.rebalance_actions = rebalance_actions
            result.tier_allocations = self._calculate_tier_allocations(result.optimal_weights, asset_tiers)
            
            # 6. Risk metrics
            result.var_95 = self._calculate_var(optimal_weights, covariance_matrix)
            result.diversification_ratio = self._calculate_diversification_ratio(optimal_weights, covariance_matrix)
            
            self.logger.info(f"✅ Portfolio optimized: {result.assets_included} assets, Sharpe={result.sharpe_ratio:.3f}")
            
            # 🎨 GENERATE VISUALIZATIONS
            try:
                from ..visualization import PortfolioVisualizer
                visualizer = PortfolioVisualizer()
                
                self.logger.info("🎨 Generating Layer 4 portfolio optimization visualizations...")
                layer4_plots = visualizer.visualize_layer4_portfolio(result, market_data, "layer4")
                
                if layer4_plots:
                    self.logger.info(f"✅ Created {len(layer4_plots)} Layer 4 visualizations:")
                    for plot_type, file_path in layer4_plots.items():
                        if file_path:
                            self.logger.info(f"  📊 {plot_type}: {file_path}")
                            
            except Exception as e:
                self.logger.error(f"Failed to create Layer 4 visualizations: {e}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Portfolio optimization failed: {e}")
            result.convergence_status = f"error: {e}"
            return result
    
    def _prepare_optimization_inputs(self, intelligence_data: Dict, market_data: Dict[str, pd.DataFrame]) -> Tuple:
        """Prepare expected returns and covariance matrix for optimization"""
        
        # Get asset scores and classifications from intelligence
        asset_scores = intelligence_data.asset_scores if hasattr(intelligence_data, 'asset_scores') else {}
        asset_classifications = intelligence_data.asset_classifications if hasattr(intelligence_data, 'asset_classifications') else {}
        recommended_portfolio = intelligence_data.recommended_portfolio if hasattr(intelligence_data, 'recommended_portfolio') else {}
        
        # Filter assets that are recommended for inclusion
        eligible_assets = []
        for symbol in recommended_portfolio.keys():
            if symbol in market_data and symbol in asset_classifications:
                classification = asset_classifications[symbol]
                if classification.tier.value in ['tier_1', 'tier_2', 'tier_3']:
                    eligible_assets.append(symbol)
        
        if len(eligible_assets) < self.config.min_portfolio_assets:
            return None, None, None
        
        # Calculate expected returns (using asset scores as proxy)
        expected_returns = []
        asset_tiers = {}
        
        for symbol in eligible_assets:
            # Use composite score as return expectation (scaled)
            score = asset_scores.get(symbol, {'composite_score': 0.5})
            if hasattr(score, 'composite_score'):
                expected_return = (score.composite_score - 0.5) * 0.4  # Scale to reasonable range
            else:
                expected_return = (score.get('composite_score', 0.5) - 0.5) * 0.4
            
            expected_returns.append(expected_return)
            
            # Store tier information
            classification = asset_classifications[symbol]
            asset_tiers[symbol] = classification.tier.value
        
        expected_returns = pd.Series(expected_returns, index=eligible_assets)
        
        # Calculate covariance matrix from historical returns
        returns_data = []
        for symbol in eligible_assets:
            if symbol in market_data:
                returns = market_data[symbol]['close'].pct_change().dropna().tail(252)  # Last year
                returns_data.append(returns)
        
        if not returns_data:
            return None, None, None
        
        # Align returns data
        returns_df = pd.concat(returns_data, axis=1, keys=eligible_assets, join='inner')
        
        if len(returns_df) < 60:  # Need minimum data
            self.logger.warning("Insufficient return history for covariance estimation")
            return None, None, None
        
        # Covariance matrix (annualized)
        covariance_matrix = returns_df.cov() * 252
        
        return expected_returns, covariance_matrix, asset_tiers
    
    def _optimize_allocation(self, expected_returns: pd.Series, covariance_matrix: pd.DataFrame, asset_tiers: Dict[str, str]) -> np.ndarray:
        """
        Optimize portfolio allocation using REAL Markowitz Modern Portfolio Theory
        """
        from .markowitz_optimizer import MarkowitzPortfolioOptimizer, MarkowitzConfig
        
        self.logger.info("Running REAL Markowitz portfolio optimization...")
        
        # Configure optimizer for our DCA context
        optimizer_config = MarkowitzConfig(
            max_weight=self.config.max_individual_weight,
            min_weight=self.config.min_individual_weight,
            target_return=self.config.target_return,
            risk_free_rate=0.10,  # Brazilian SELIC rate
            allow_shorting=False,  # DCA doesn't allow short selling
            max_portfolio_volatility=self.config.max_volatility
        )
        
        # Initialize optimizer
        optimizer = MarkowitzPortfolioOptimizer(optimizer_config)
        
        try:
            # Run real optimization
            optimization_result = optimizer.optimize_portfolio(
                expected_returns=expected_returns,
                covariance_matrix=covariance_matrix
            )
            
            if optimization_result.optimization_success:
                # Extract weights as numpy array in correct order
                asset_symbols = expected_returns.index
                weights = np.array([optimization_result.optimal_weights.get(symbol, 0.0) for symbol in asset_symbols])
                
                # Apply additional DCA-specific constraints (tier allocations, etc.)
                weights = self._apply_constraints(weights, asset_symbols, asset_tiers)
                
                self.logger.info(f"Markowitz optimization successful:")
                self.logger.info(f"  Portfolio expected return: {optimization_result.expected_return:.1%}")
                self.logger.info(f"  Portfolio volatility: {optimization_result.expected_volatility:.1%}")
                self.logger.info(f"  Sharpe ratio: {optimization_result.sharpe_ratio:.3f}")
                self.logger.info(f"  Diversification ratio: {optimization_result.diversification_ratio:.2f}")
                
                return weights
            else:
                self.logger.error(f"Markowitz optimization failed: {optimization_result.optimization_message}")
                # Fallback to equal weights
                return np.ones(len(expected_returns)) / len(expected_returns)
                
        except Exception as e:
            self.logger.error(f"Markowitz optimization error: {e}")
            # Fallback to equal weights with risk adjustment
            asset_volatilities = np.sqrt(np.diag(covariance_matrix))
            weights = 1 / asset_volatilities
            weights = weights / weights.sum()
            return weights
    
    def _apply_constraints(self, weights: np.ndarray, asset_symbols: pd.Index, asset_tiers: Dict[str, str]) -> np.ndarray:
        """Apply portfolio constraints"""
        
        # Individual weight constraints
        weights = np.clip(weights, self.config.min_individual_weight, self.config.max_individual_weight)
        
        # Tier allocation constraints
        tier_weights = {'tier_1': 0, 'tier_2': 0, 'tier_3': 0}
        
        for i, symbol in enumerate(asset_symbols):
            tier = asset_tiers.get(symbol, 'tier_3')
            tier_weights[tier] += weights[i]
        
        # Adjust if tier allocations are out of bounds
        for tier in tier_weights:
            if tier == 'tier_1':
                target_min, target_max = self.config.tier1_allocation
            elif tier == 'tier_2':
                target_min, target_max = self.config.tier2_allocation
            else:  # tier_3
                target_min, target_max = self.config.tier3_allocation
            
            current_allocation = tier_weights[tier]
            
            if current_allocation > target_max:
                # Reduce tier allocation proportionally
                scale_factor = target_max / current_allocation
                for i, symbol in enumerate(asset_symbols):
                    if asset_tiers.get(symbol) == tier:
                        weights[i] *= scale_factor
            
            elif current_allocation < target_min:
                # Increase tier allocation proportionally
                scale_factor = target_min / current_allocation if current_allocation > 0 else 1
                for i, symbol in enumerate(asset_symbols):
                    if asset_tiers.get(symbol) == tier:
                        weights[i] *= scale_factor
        
        # Normalize to sum to 1 (minus reserve allocation)
        target_sum = 1.0 - self.config.reserve_allocation
        weights = weights * (target_sum / weights.sum())
        
        return weights
    
    def _generate_rebalance_actions(self, current_positions: Dict[str, float], target_weights: Dict[str, float]) -> List[str]:
        """Generate rebalancing actions"""
        
        actions = []
        
        for symbol, target_weight in target_weights.items():
            current_weight = current_positions.get(symbol, 0.0)
            weight_diff = target_weight - current_weight
            
            if abs(weight_diff) > self.config.rebalance_threshold * current_weight:
                if abs(weight_diff) > self.config.min_trade_size:
                    action_type = "BUY" if weight_diff > 0 else "SELL"
                    actions.append(f"{action_type} {symbol}: {current_weight:.3f} -> {target_weight:.3f} ({weight_diff:+.3f})")
        
        return actions
    
    def _calculate_tier_allocations(self, optimal_weights: Dict[str, float], asset_tiers: Dict[str, str]) -> Dict[str, float]:
        """Calculate allocation by tier"""
        
        tier_allocations = {'tier_1': 0, 'tier_2': 0, 'tier_3': 0, 'reserve': self.config.reserve_allocation}
        
        for symbol, weight in optimal_weights.items():
            tier = asset_tiers.get(symbol, 'tier_3')
            tier_allocations[tier] += weight
        
        return tier_allocations
    
    def _calculate_var(self, weights: np.ndarray, covariance_matrix: pd.DataFrame, confidence: float = 0.05) -> float:
        """Calculate Value at Risk"""
        
        portfolio_variance = np.dot(weights, np.dot(covariance_matrix.values, weights))
        portfolio_std = np.sqrt(portfolio_variance)
        
        # Assuming normal distribution
        from scipy.stats import norm
        var_95 = norm.ppf(confidence) * portfolio_std
        
        return abs(var_95)
    
    def _calculate_diversification_ratio(self, weights: np.ndarray, covariance_matrix: pd.DataFrame) -> float:
        """Calculate diversification ratio"""
        
        # Weighted average of individual volatilities
        individual_vols = np.sqrt(np.diag(covariance_matrix.values))
        weighted_avg_vol = np.sum(weights * individual_vols)
        
        # Portfolio volatility
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(covariance_matrix.values, weights)))
        
        # Diversification ratio
        diversification_ratio = weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 1.0
        
        return diversification_ratio


# Process Layer 3 intelligence through Layer 4 optimization
async def process_intelligence_optimization(intelligence_result: Dict, 
                                          market_data: Dict[str, pd.DataFrame],
                                          config: Optional[Layer4Config] = None) -> OptimizationResult:
    """
    Process Layer 3 intelligence through Layer 4 portfolio optimization
    
    Args:
        intelligence_result: Results from Layer 3 intelligence
        market_data: Market data for optimization
        config: Optional Layer 4 configuration
        
    Returns:
        OptimizationResult with optimal portfolio
    """
    processor = Layer4PortfolioProcessor(config)
    return await processor.optimize_portfolio(intelligence_result, market_data)


# Layer 4 Portfolio Optimization is integrated into the main DCA system
# Use the main CLI for testing: python cli.py run
