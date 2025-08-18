#!/usr/bin/env python3
"""
Layer 5 Execution Processor

Main orchestrator for trade execution and order management.
Provides intelligent order routing, risk controls, and position management.

Key Features:
- Smart Order Execution
- Real-time Risk Controls
- Position Reconciliation
- Transaction Cost Analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    DCA = "dca"  # Dollar-cost averaging


class OrderStatus(Enum):
    """Order execution status"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Layer5Config:
    """Configuration for Layer 5 Execution"""
    
    # Order execution settings
    default_order_type: OrderType = OrderType.MARKET
    max_slippage_tolerance: float = 0.02       # 2% maximum slippage
    order_timeout_minutes: int = 15            # Order timeout
    
    # Risk control settings
    max_position_size: float = 0.05            # 5% max position
    max_daily_trades: int = 50                 # Daily trade limit
    max_order_value: float = 10000.0           # Maximum order value (BRL)
    min_order_value: float = 25.0              # Minimum order value (BRL)
    
    # DCA execution settings
    dca_split_threshold: float = 1000.0        # Split orders above R$1000
    dca_split_parts: int = 4                   # Split into 4 parts
    dca_interval_minutes: int = 30             # 30min intervals
    
    # Position management
    enable_position_tracking: bool = True
    reconciliation_frequency_hours: int = 6   # Reconcile every 6 hours
    max_position_drift: float = 0.1            # 10% position drift tolerance
    
    # Transaction cost settings
    estimated_trading_fee: float = 0.005       # 0.5% trading fee
    market_impact_factor: float = 0.001        # Market impact estimate


@dataclass
class ExecutionResult:
    """Execution result summary"""
    
    # Order results
    orders_submitted: int = 0
    orders_filled: int = 0
    orders_failed: int = 0
    total_value_executed: float = 0.0
    
    # Execution quality
    average_slippage: float = 0.0
    total_transaction_costs: float = 0.0
    execution_time_seconds: float = 0.0
    
    # Position updates
    positions_updated: int = 0
    position_errors: List[str] = field(default_factory=list)
    
    # Risk controls
    risk_checks_passed: int = 0
    risk_checks_failed: int = 0
    risk_violations: List[str] = field(default_factory=list)
    
    # Metadata
    execution_timestamp: datetime = field(default_factory=datetime.now)
    execution_session_id: str = ""


class Layer5ExecutionProcessor:
    """
    Layer 5 Execution Processor
    
    Handles intelligent order execution with risk controls and position management
    for the DCA investment system.
    """
    
    def __init__(self, config: Optional[Layer5Config] = None):
        self.config = config or Layer5Config()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Execution state
        self._active_orders = {}
        self._current_positions = {}
        self._daily_trade_count = 0
        self._last_reconciliation = None
        
        # Risk tracking
        self._daily_volume = 0.0
        self._risk_violations = []
        
        self.logger.info(f"⚡ Layer5ExecutionProcessor initialized")
        self.logger.info(f"  Max slippage: {self.config.max_slippage_tolerance:.1%}")
        self.logger.info(f"  Order limits: R${self.config.min_order_value:.0f} - R${self.config.max_order_value:.0f}")
    
    async def execute_portfolio_changes(self, 
                                       target_portfolio: Dict[str, float],
                                       current_positions: Optional[Dict[str, float]] = None,
                                       available_capital: float = 1000.0) -> ExecutionResult:
        """
        Execute portfolio changes based on optimization results
        
        Args:
            target_portfolio: Target portfolio weights
            current_positions: Current portfolio positions  
            available_capital: Available capital for investment
            
        Returns:
            ExecutionResult with execution summary
        """
        self.logger.info(f"⚡ Executing portfolio changes with R${available_capital:.2f} capital...")
        
        result = ExecutionResult(
            execution_session_id=f"exec_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        start_time = datetime.now()
        
        try:
            current_positions = current_positions or {}
            
            # 1. Calculate required trades
            required_trades = self._calculate_required_trades(
                target_portfolio, current_positions, available_capital
            )
            
            if not required_trades:
                self.logger.info("No trades required")
                return result
            
            # 2. Apply risk controls
            approved_trades = []
            for trade in required_trades:
                risk_check = self._perform_risk_check(trade)
                if risk_check['passed']:
                    approved_trades.append(trade)
                    result.risk_checks_passed += 1
                else:
                    result.risk_checks_failed += 1
                    result.risk_violations.extend(risk_check['violations'])
            
            # 3. Execute approved trades
            for trade in approved_trades:
                try:
                    execution_result = await self._execute_trade(trade)
                    
                    if execution_result['status'] == 'filled':
                        result.orders_filled += 1
                        result.total_value_executed += execution_result['value']
                        result.average_slippage += execution_result['slippage']
                        
                        # Update position tracking
                        if self.config.enable_position_tracking:
                            self._update_position(trade['symbol'], execution_result)
                            result.positions_updated += 1
                    
                    else:
                        result.orders_failed += 1
                    
                    result.orders_submitted += 1
                    
                except Exception as e:
                    self.logger.error(f"Trade execution failed for {trade['symbol']}: {e}")
                    result.orders_failed += 1
                    result.position_errors.append(f"{trade['symbol']}: {str(e)}")
            
            # 4. Calculate execution metrics
            if result.orders_filled > 0:
                result.average_slippage /= result.orders_filled
            
            result.total_transaction_costs = result.total_value_executed * self.config.estimated_trading_fee
            result.execution_time_seconds = (datetime.now() - start_time).total_seconds()
            
            self.logger.info(f"✅ Execution completed: {result.orders_filled}/{result.orders_submitted} orders filled")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Portfolio execution failed: {e}")
            result.position_errors.append(f"Execution error: {str(e)}")
            return result
    
    async def execute_dca_investment(self, 
                                   dca_amount: float,
                                   target_allocations: Dict[str, float]) -> ExecutionResult:
        """
        Execute DCA investment across target allocations
        
        Args:
            dca_amount: Total amount to invest (BRL)
            target_allocations: Target allocation weights
            
        Returns:
            ExecutionResult with DCA execution summary  
        """
        self.logger.info(f"💰 Executing DCA investment of R${dca_amount:.2f}...")
        
        result = ExecutionResult(
            execution_session_id=f"dca_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        start_time = datetime.now()
        
        try:
            # Calculate individual asset amounts
            asset_amounts = {}
            for symbol, weight in target_allocations.items():
                amount = dca_amount * weight
                if amount >= self.config.min_order_value:
                    asset_amounts[symbol] = amount
            
            # Execute DCA orders
            for symbol, amount in asset_amounts.items():
                try:
                    # Split large orders if needed
                    if amount > self.config.dca_split_threshold:
                        split_amount = amount / self.config.dca_split_parts
                        
                        for i in range(self.config.dca_split_parts):
                            trade = {
                                'symbol': symbol,
                                'side': 'buy',
                                'amount': split_amount,
                                'type': OrderType.DCA,
                                'split_order': i + 1
                            }
                            
                            execution_result = await self._execute_trade(trade)
                            self._process_execution_result(execution_result, result)
                            
                            # Wait between split orders
                            if i < self.config.dca_split_parts - 1:
                                await asyncio.sleep(self.config.dca_interval_minutes * 60)
                    
                    else:
                        trade = {
                            'symbol': symbol,
                            'side': 'buy', 
                            'amount': amount,
                            'type': OrderType.DCA
                        }
                        
                        execution_result = await self._execute_trade(trade)
                        self._process_execution_result(execution_result, result)
                
                except Exception as e:
                    self.logger.error(f"DCA execution failed for {symbol}: {e}")
                    result.orders_failed += 1
            
            # Final metrics
            result.execution_time_seconds = (datetime.now() - start_time).total_seconds()
            result.total_transaction_costs = result.total_value_executed * self.config.estimated_trading_fee
            
            self.logger.info(f"✅ DCA execution completed: R${result.total_value_executed:.2f} invested")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ DCA execution failed: {e}")
            result.position_errors.append(f"DCA error: {str(e)}")
            return result
    
    def _calculate_required_trades(self, target_portfolio: Dict[str, float], 
                                 current_positions: Dict[str, float],
                                 available_capital: float) -> List[Dict]:
        """Calculate trades needed to reach target portfolio"""
        
        trades = []
        
        for symbol, target_weight in target_portfolio.items():
            current_weight = current_positions.get(symbol, 0.0)
            weight_diff = target_weight - current_weight
            
            # Skip if difference is too small
            if abs(weight_diff) < 0.001:  # 0.1% threshold
                continue
            
            # Calculate trade amount
            trade_amount = available_capital * abs(weight_diff)
            
            if trade_amount >= self.config.min_order_value:
                trade = {
                    'symbol': symbol,
                    'side': 'buy' if weight_diff > 0 else 'sell',
                    'amount': trade_amount,
                    'target_weight': target_weight,
                    'current_weight': current_weight,
                    'type': OrderType.MARKET
                }
                trades.append(trade)
        
        return trades
    
    def _perform_risk_check(self, trade: Dict) -> Dict:
        """Perform risk checks on trade"""
        
        violations = []
        
        # Check order size limits
        if trade['amount'] > self.config.max_order_value:
            violations.append(f"Order exceeds max value: R${trade['amount']:.2f} > R${self.config.max_order_value:.2f}")
        
        if trade['amount'] < self.config.min_order_value:
            violations.append(f"Order below min value: R${trade['amount']:.2f} < R${self.config.min_order_value:.2f}")
        
        # Check daily limits
        if self._daily_trade_count >= self.config.max_daily_trades:
            violations.append(f"Daily trade limit exceeded: {self._daily_trade_count}")
        
        # Check position limits (for buy orders)
        if trade['side'] == 'buy':
            target_weight = trade.get('target_weight', 0)
            if target_weight > self.config.max_position_size:
                violations.append(f"Position exceeds limit: {target_weight:.1%} > {self.config.max_position_size:.1%}")
        
        # Check daily volume limit
        if self._daily_volume + trade['amount'] > self.config.max_order_value * 10:
            violations.append("Daily volume limit exceeded")
        
        return {
            'passed': len(violations) == 0,
            'violations': violations
        }
    
    async def _execute_trade(self, trade: Dict) -> Dict:
        """Execute individual trade (mock implementation)"""
        
        # Mock execution - replace with real exchange API
        symbol = trade['symbol']
        amount = trade['amount']
        
        # Simulate execution
        await asyncio.sleep(0.1)  # Simulate network delay
        
        # Mock execution result
        mock_slippage = np.random.normal(0, 0.005)  # Random slippage
        mock_price = 100.0 * (1 + np.random.normal(0, 0.01))  # Mock price
        
        execution_result = {
            'symbol': symbol,
            'status': 'filled',
            'amount': amount,
            'price': mock_price,
            'slippage': abs(mock_slippage),
            'value': amount,
            'timestamp': datetime.now(),
            'order_id': f"order_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        }
        
        # Update tracking
        self._daily_trade_count += 1
        self._daily_volume += amount
        
        self.logger.debug(f"  Executed {trade['side']} {symbol}: R${amount:.2f}")
        
        return execution_result
    
    def _process_execution_result(self, execution_result: Dict, result: ExecutionResult):
        """Process and update execution result"""
        
        if execution_result['status'] == 'filled':
            result.orders_filled += 1
            result.total_value_executed += execution_result['value']
            result.average_slippage += execution_result['slippage']
            
            # Update positions
            if self.config.enable_position_tracking:
                self._update_position(execution_result['symbol'], execution_result)
                result.positions_updated += 1
        else:
            result.orders_failed += 1
        
        result.orders_submitted += 1
    
    def _update_position(self, symbol: str, execution_result: Dict):
        """Update position tracking"""
        
        if symbol not in self._current_positions:
            self._current_positions[symbol] = {
                'quantity': 0.0,
                'value': 0.0,
                'avg_price': 0.0,
                'last_updated': datetime.now()
            }
        
        position = self._current_positions[symbol]
        
        # Update position based on trade
        if execution_result.get('side', 'buy') == 'buy':
            new_quantity = position['quantity'] + (execution_result['value'] / execution_result['price'])
            new_value = position['value'] + execution_result['value']
            position['avg_price'] = new_value / new_quantity if new_quantity > 0 else 0
        else:  # sell
            new_quantity = position['quantity'] - (execution_result['value'] / execution_result['price'])
            new_value = position['value'] - execution_result['value']
            position['avg_price'] = new_value / new_quantity if new_quantity > 0 else 0
        
        position['quantity'] = max(0, new_quantity)
        position['value'] = max(0, new_value)
        position['last_updated'] = datetime.now()
    
    def get_execution_summary(self, result: ExecutionResult) -> Dict[str, any]:
        """Get execution summary"""
        
        success_rate = result.orders_filled / result.orders_submitted * 100 if result.orders_submitted > 0 else 0
        
        return {
            'execution_session_id': result.execution_session_id,
            'execution_timestamp': result.execution_timestamp.isoformat(),
            'orders_summary': {
                'submitted': result.orders_submitted,
                'filled': result.orders_filled,
                'failed': result.orders_failed,
                'success_rate': f"{success_rate:.1f}%"
            },
            'execution_quality': {
                'total_value_executed': f"R${result.total_value_executed:.2f}",
                'average_slippage': f"{result.average_slippage:.3%}",
                'total_transaction_costs': f"R${result.total_transaction_costs:.2f}",
                'execution_time': f"{result.execution_time_seconds:.1f}s"
            },
            'risk_controls': {
                'checks_passed': result.risk_checks_passed,
                'checks_failed': result.risk_checks_failed,
                'violations': result.risk_violations
            },
            'position_updates': {
                'positions_updated': result.positions_updated,
                'errors': result.position_errors
            }
        }


# Process Layer 4 optimization through Layer 5 execution
async def process_optimization_execution(optimization_result: Dict,
                                       available_capital: float = 1000.0,
                                       config: Optional[Layer5Config] = None) -> ExecutionResult:
    """
    Process Layer 4 optimization through Layer 5 execution
    
    Args:
        optimization_result: Results from Layer 4 optimization
        available_capital: Available capital for investment
        config: Optional Layer 5 configuration
        
    Returns:
        ExecutionResult with execution summary
    """
    processor = Layer5ExecutionProcessor(config)
    
    # Extract target portfolio from optimization result
    target_portfolio = optimization_result.get('optimal_weights', {})
    current_positions = {}  # Would come from position management system
    
    return await processor.execute_portfolio_changes(target_portfolio, current_positions, available_capital)


# Layer 5 Execution is integrated into the main DCA system
# Use the main CLI for testing: python cli.py run
