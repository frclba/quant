import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

@dataclass
class MarkowitzConfig:
    """Configuration for Markowitz Portfolio Optimization"""
    
    # Optimization constraints
    max_weight: float = 0.3                    # Maximum weight per asset (30%)
    min_weight: float = 0.01                   # Minimum weight per asset (1%)
    target_return: Optional[float] = None       # Target return (if None, optimize Sharpe)
    risk_free_rate: float = 0.10               # Brazilian SELIC rate (~10% annually)
    
    # Optimization settings
    optimization_method: str = 'SLSQP'         # Scipy optimization method
    max_iterations: int = 1000                 # Maximum optimization iterations
    tolerance: float = 1e-6                    # Optimization tolerance
    
    # Portfolio constraints
    allow_shorting: bool = False               # No short selling for DCA
    force_full_investment: bool = True         # Weights must sum to 1.0
    diversification_penalty: float = 0.0      # Penalty for concentration
    
    # Risk management
    max_portfolio_volatility: float = 0.6     # Maximum portfolio volatility (60%)
    exclude_high_correlation: bool = True     # Exclude highly correlated assets
    max_correlation_threshold: float = 0.9    # Maximum correlation between assets

@dataclass
class OptimizationResult:
    """Results from Markowitz portfolio optimization"""
    
    # Optimal portfolio
    optimal_weights: Dict[str, float] = field(default_factory=dict)
    expected_return: float = 0.0
    expected_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    
    # Efficiency metrics
    diversification_ratio: float = 0.0
    portfolio_beta: float = 1.0
    var_95: float = 0.0                       # 5% Value at Risk
    max_drawdown_estimate: float = 0.0
    
    # Optimization metadata
    optimization_success: bool = False
    optimization_message: str = ""
    iterations_used: int = 0
    
    # Risk decomposition
    risk_contribution: Dict[str, float] = field(default_factory=dict)
    correlation_matrix: Optional[pd.DataFrame] = None
    
    # Performance attribution
    return_attribution: Dict[str, float] = field(default_factory=dict)
    risk_metrics: Dict[str, float] = field(default_factory=dict)

class MarkowitzPortfolioOptimizer:
    """
    Modern Portfolio Theory implementation using Markowitz optimization
    
    Features:
    - Sharpe ratio maximization (risk-adjusted returns)
    - Efficient frontier calculation
    - Risk parity and equal weight alternatives
    - Constraint handling (no shorting, weight limits)
    - Brazilian market focus (BRL risk-free rate)
    """
    
    def __init__(self, config: Optional[MarkowitzConfig] = None):
        self.config = config or MarkowitzConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Optimization cache
        self._last_optimization = None
        self._efficient_frontier_cache = None
        
        self.logger.info(f"Markowitz optimizer initialized")
        self.logger.info(f"  Max weight per asset: {self.config.max_weight:.1%}")
        self.logger.info(f"  Risk-free rate: {self.config.risk_free_rate:.1%}")
        self.logger.info(f"  Shorting allowed: {self.config.allow_shorting}")
    
    def optimize_portfolio(self, 
                          expected_returns: pd.Series,
                          covariance_matrix: pd.DataFrame,
                          current_weights: Optional[Dict[str, float]] = None) -> OptimizationResult:
        """
        Optimize portfolio using Markowitz Modern Portfolio Theory
        
        Args:
            expected_returns: Expected annual returns for each asset
            covariance_matrix: Covariance matrix of asset returns (annualized)
            current_weights: Current portfolio weights (for comparison)
            
        Returns:
            OptimizationResult with optimal portfolio and metrics
        """
        self.logger.info(f"Optimizing portfolio for {len(expected_returns)} assets...")
        
        # Align data
        assets = list(expected_returns.index)
        returns = expected_returns.values
        cov_matrix = covariance_matrix.loc[assets, assets].values
        
        result = OptimizationResult()
        
        try:
            # Optimize for maximum Sharpe ratio
            if self.config.target_return is None:
                optimal_weights = self._maximize_sharpe_ratio(returns, cov_matrix, assets)
            else:
                optimal_weights = self._minimize_variance_target_return(returns, cov_matrix, assets)
            
            # Calculate portfolio metrics
            portfolio_return = np.dot(optimal_weights, returns)
            portfolio_variance = np.dot(optimal_weights, np.dot(cov_matrix, optimal_weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            sharpe_ratio = (portfolio_return - self.config.risk_free_rate) / portfolio_volatility
            
            # Create results
            result.optimal_weights = dict(zip(assets, optimal_weights))
            result.expected_return = portfolio_return
            result.expected_volatility = portfolio_volatility
            result.sharpe_ratio = sharpe_ratio
            result.optimization_success = True
            
            # Calculate additional metrics
            result.risk_metrics = self._calculate_risk_metrics(optimal_weights, returns, cov_matrix, assets)
            result.risk_contribution = self._calculate_risk_contribution(optimal_weights, cov_matrix, assets)
            result.correlation_matrix = covariance_matrix
            
            # Log results
            self.logger.info(f"Optimization successful:")
            self.logger.info(f"  Expected return: {portfolio_return:.1%}")
            self.logger.info(f"  Expected volatility: {portfolio_volatility:.1%}")
            self.logger.info(f"  Sharpe ratio: {sharpe_ratio:.3f}")
            self.logger.info(f"  Assets in portfolio: {len([w for w in optimal_weights if w > 0.01])}")
            
            self._last_optimization = result
            return result
            
        except Exception as e:
            self.logger.error(f"Portfolio optimization failed: {e}")
            result.optimization_success = False
            result.optimization_message = str(e)
            return result
    
    def _maximize_sharpe_ratio(self, returns: np.ndarray, cov_matrix: np.ndarray, assets: List[str]) -> np.ndarray:
        """Maximize Sharpe ratio using scipy optimization"""
        
        n_assets = len(returns)
        
        # Objective function (negative Sharpe ratio to minimize)
        def negative_sharpe(weights):
            portfolio_return = np.dot(weights, returns)
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            sharpe_ratio = (portfolio_return - self.config.risk_free_rate) / portfolio_volatility
            return -sharpe_ratio  # Minimize negative Sharpe = maximize Sharpe
        
        # Constraints
        constraints = []
        
        # Weights sum to 1
        if self.config.force_full_investment:
            constraints.append({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        
        # Weight bounds
        if self.config.allow_shorting:
            bounds = [(-self.config.max_weight, self.config.max_weight) for _ in range(n_assets)]
        else:
            bounds = [(self.config.min_weight, self.config.max_weight) for _ in range(n_assets)]
        
        # Initial guess (equal weights)
        initial_weights = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(
            negative_sharpe,
            initial_weights,
            method=self.config.optimization_method,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.config.max_iterations, 'ftol': self.config.tolerance}
        )
        
        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")
        
        return result.x
    
    def _minimize_variance_target_return(self, returns: np.ndarray, cov_matrix: np.ndarray, assets: List[str]) -> np.ndarray:
        """Minimize variance for target return (efficient frontier point)"""
        
        n_assets = len(returns)
        
        # Objective function (portfolio variance)
        def portfolio_variance(weights):
            return np.dot(weights, np.dot(cov_matrix, weights))
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
            {'type': 'eq', 'fun': lambda w: np.dot(w, returns) - self.config.target_return}  # Target return
        ]
        
        # Bounds
        bounds = [(self.config.min_weight, self.config.max_weight) for _ in range(n_assets)]
        
        # Initial guess
        initial_weights = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(
            portfolio_variance,
            initial_weights,
            method=self.config.optimization_method,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.config.max_iterations}
        )
        
        if not result.success:
            raise ValueError(f"Target return optimization failed: {result.message}")
        
        return result.x
    
    def _calculate_risk_metrics(self, weights: np.ndarray, returns: np.ndarray, cov_matrix: np.ndarray, assets: List[str]) -> Dict[str, float]:
        """Calculate comprehensive risk metrics"""
        
        portfolio_return = np.dot(weights, returns)
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Diversification ratio
        weighted_avg_vol = np.dot(weights, np.sqrt(np.diag(cov_matrix)))
        diversification_ratio = weighted_avg_vol / portfolio_volatility
        
        # VaR (assuming normal distribution)
        var_95 = portfolio_return - 1.645 * portfolio_volatility  # 5% VaR
        
        return {
            'portfolio_volatility': portfolio_volatility,
            'diversification_ratio': diversification_ratio,
            'var_95': var_95,
            'tracking_error': 0.0,  # Would need benchmark
            'information_ratio': 0.0,  # Would need benchmark
            'maximum_drawdown_estimate': 2.5 * portfolio_volatility  # Rough estimate
        }
    
    def _calculate_risk_contribution(self, weights: np.ndarray, cov_matrix: np.ndarray, assets: List[str]) -> Dict[str, float]:
        """Calculate each asset's contribution to portfolio risk"""
        
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
        
        # Marginal contribution to risk
        marginal_contrib = np.dot(cov_matrix, weights) / np.sqrt(portfolio_variance)
        
        # Risk contribution (marginal * weight)
        risk_contrib = weights * marginal_contrib
        risk_contrib_pct = risk_contrib / np.sum(risk_contrib)
        
        return dict(zip(assets, risk_contrib_pct))
