#!/usr/bin/env python3
"""
Layer 6 Orchestration Processor

Main system orchestrator that coordinates all layers and provides comprehensive
monitoring, performance analytics, and workflow management.

Key Features:
- Complete System Workflow Orchestration
- Performance Attribution and Analytics  
- Health Monitoring and Alerting
- Centralized Logging and Reporting
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import asyncio

logger = logging.getLogger(__name__)


class WorkflowType(Enum):
    """Workflow types"""
    DAILY_DCA = "daily_dca"
    MONTHLY_REBALANCE = "monthly_rebalance"
    FULL_PIPELINE = "full_pipeline"
    DATA_COLLECTION = "data_collection"
    PERFORMANCE_ANALYSIS = "performance_analysis"


class SystemStatus(Enum):
    """System status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    ERROR = "error"
    MAINTENANCE = "maintenance"


@dataclass
class Layer6Config:
    """Configuration for Layer 6 Orchestration"""
    
    # Workflow scheduling
    enable_daily_dca: bool = True
    daily_dca_amount: float = 33.33              # R$33.33 daily (~R$1000 monthly)
    daily_dca_time: str = "14:00"                # 2 PM execution time
    
    enable_monthly_rebalance: bool = True
    monthly_rebalance_day: int = 1               # 1st of month
    rebalance_threshold: float = 0.25            # 25% drift threshold
    
    # Performance analytics
    benchmark_symbol: str = "BTC"
    performance_lookback_days: int = 252         # 1 year
    enable_attribution_analysis: bool = True
    
    # Monitoring settings
    health_check_interval_minutes: int = 30
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'max_drawdown': 0.3,                     # 30% max drawdown alert
        'daily_loss': 0.05,                      # 5% daily loss alert
        'execution_failure_rate': 0.1,          # 10% execution failure rate
        'data_staleness_hours': 6               # 6 hours data staleness
    })
    
    # Reporting settings
    enable_daily_report: bool = True
    enable_monthly_report: bool = True
    report_email: Optional[str] = None
    
    # System settings
    max_concurrent_workflows: int = 3
    workflow_timeout_minutes: int = 60
    enable_performance_tracking: bool = True


@dataclass
class OrchestrationResult:
    """Complete orchestration result"""
    
    # Workflow execution
    workflow_type: WorkflowType = WorkflowType.FULL_PIPELINE
    workflow_status: str = "success"
    workflow_duration: Optional[timedelta] = None
    
    # Layer results
    layer1_result: Optional[Any] = None          # Data collection
    layer2_result: Optional[Any] = None          # Data processing
    layer3_result: Optional[Any] = None          # Intelligence
    layer4_result: Optional[Any] = None          # Optimization
    layer5_result: Optional[Any] = None          # Execution
    
    # Performance metrics
    portfolio_value: float = 0.0
    daily_return: float = 0.0
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    
    # System health
    system_status: SystemStatus = SystemStatus.HEALTHY
    health_alerts: List[str] = field(default_factory=list)
    performance_warnings: List[str] = field(default_factory=list)
    
    # Execution summary
    execution_timestamp: datetime = field(default_factory=datetime.now)
    total_assets_processed: int = 0
    orders_executed: int = 0
    amount_invested: float = 0.0
    
    # Analytics
    performance_attribution: Dict[str, float] = field(default_factory=dict)
    risk_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    orchestration_stats: Dict[str, Any] = field(default_factory=dict)


class Layer6OrchestrationProcessor:
    """
    Layer 6 Orchestration Processor
    
    Central coordinator that orchestrates the complete DCA system workflow,
    provides performance monitoring, and manages system health.
    """
    
    def __init__(self, config: Optional[Layer6Config] = None):
        self.config = config or Layer6Config()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Workflow state
        self._active_workflows = {}
        self._workflow_history = []
        self._system_metrics = {}
        
        # Performance tracking
        self._performance_history = []
        self._portfolio_snapshots = []
        self._last_health_check = None
        
        # Initialize layer processors (would be injected in real system)
        self._layer_processors = {}
        
        self.logger.info(f"🎼 Layer6OrchestrationProcessor initialized")
        self.logger.info(f"  Daily DCA: {'✓' if self.config.enable_daily_dca else '✗'} (R${self.config.daily_dca_amount:.2f})")
        self.logger.info(f"  Monthly Rebalance: {'✓' if self.config.enable_monthly_rebalance else '✗'}")
        self.logger.info(f"  Performance Tracking: {'✓' if self.config.enable_performance_tracking else '✗'}")
    
    async def run_complete_dca_pipeline(self, 
                                      dca_amount: Optional[float] = None) -> OrchestrationResult:
        """
        Run the complete DCA system pipeline
        
        Args:
            dca_amount: Optional DCA investment amount
            
        Returns:
            OrchestrationResult with complete system execution
        """
        dca_amount = dca_amount or self.config.daily_dca_amount
        
        self.logger.info(f"🎼 Running complete DCA pipeline with R${dca_amount:.2f}...")
        
        workflow_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()
        
        result = OrchestrationResult(
            workflow_type=WorkflowType.FULL_PIPELINE,
            execution_timestamp=start_time
        )
        
        try:
            # Track active workflow
            self._active_workflows[workflow_id] = {
                'start_time': start_time,
                'type': WorkflowType.FULL_PIPELINE,
                'amount': dca_amount
            }
            
            # Layer 1: Data Collection
            self.logger.info("📊 Layer 1: Data Collection...")
            layer1_result = await self._execute_layer1()
            result.layer1_result = layer1_result
            
            if not layer1_result or layer1_result.get('error'):
                raise Exception("Layer 1 data collection failed")
            
            # Layer 2: Data Processing  
            self.logger.info("🔄 Layer 2: Data Processing...")
            layer2_result = await self._execute_layer2(layer1_result)
            result.layer2_result = layer2_result
            
            if not layer2_result or layer2_result.get('error'):
                raise Exception("Layer 2 data processing failed")
            
            # Layer 3: Intelligence & Scoring
            self.logger.info("🧠 Layer 3: Intelligence & Scoring...")
            layer3_result = await self._execute_layer3(layer2_result)
            result.layer3_result = layer3_result
            
            if not layer3_result or layer3_result.get('error'):
                raise Exception("Layer 3 intelligence failed")
            
            # Layer 4: Portfolio Optimization
            self.logger.info("📈 Layer 4: Portfolio Optimization...")
            layer4_result = await self._execute_layer4(layer3_result, layer2_result)
            result.layer4_result = layer4_result
            
            if not layer4_result or layer4_result.get('error'):
                raise Exception("Layer 4 optimization failed")
            
            # Layer 5: Execution
            self.logger.info("⚡ Layer 5: Execution...")
            layer5_result = await self._execute_layer5(layer4_result, dca_amount)
            result.layer5_result = layer5_result
            
            # Process results and calculate metrics
            result = self._process_pipeline_results(result)
            result.workflow_duration = datetime.now() - start_time
            result.workflow_status = "success"
            
            # Update tracking
            self._update_performance_tracking(result)
            self._update_system_health()
            
            # Cleanup workflow tracking
            del self._active_workflows[workflow_id]
            self._workflow_history.append(result)
            
            self.logger.info(f"✅ Complete pipeline finished in {result.workflow_duration}")
            
            # 🎨 GENERATE SYSTEM PERFORMANCE VISUALIZATIONS
            try:
                from ..visualization import PortfolioVisualizer
                visualizer = PortfolioVisualizer()
                
                self.logger.info("🎨 Generating system performance visualizations...")
                
                # Collect all results for visualization
                pipeline_results = {
                    'layer1': layer1_result,
                    'layer2': layer2_result,
                    'layer3': layer3_result,
                    'layer4': layer4_result,
                    'layer5': layer5_result,
                    'total_duration': result.workflow_duration
                }
                
                system_plots = visualizer.visualize_system_performance(pipeline_results, "system")
                
                if system_plots:
                    self.logger.info(f"✅ Created {len(system_plots)} system performance visualizations:")
                    for plot_type, file_path in system_plots.items():
                        if file_path:
                            self.logger.info(f"  📊 {plot_type}: {file_path}")
                
                # Create comprehensive dashboard
                all_plots = {
                    'system': system_plots
                }
                
                dashboard_path = visualizer.create_visualization_index(all_plots)
                if dashboard_path:
                    self.logger.info(f"🎯 VISUALIZATION DASHBOARD CREATED: {dashboard_path}")
                    self.logger.info("📊 Open the dashboard in your browser to see all analysis results!")
                        
            except Exception as e:
                self.logger.error(f"Failed to create system performance visualizations: {e}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline execution failed: {e}")
            result.workflow_status = f"failed: {e}"
            result.workflow_duration = datetime.now() - start_time
            result.system_status = SystemStatus.ERROR
            result.health_alerts.append(f"Pipeline failure: {str(e)}")
            
            # Cleanup
            if workflow_id in self._active_workflows:
                del self._active_workflows[workflow_id]
            
            return result
    
    async def run_daily_dca_workflow(self) -> OrchestrationResult:
        """Run daily DCA investment workflow"""
        
        self.logger.info(f"💰 Running daily DCA workflow...")
        
        result = OrchestrationResult(
            workflow_type=WorkflowType.DAILY_DCA,
            execution_timestamp=datetime.now()
        )
        
        if not self.config.enable_daily_dca:
            result.workflow_status = "skipped: daily DCA disabled"
            return result
        
        try:
            # Get latest portfolio intelligence (simplified)
            latest_intelligence = self._get_latest_intelligence()
            
            if not latest_intelligence:
                # Run minimal pipeline to get recommendations
                intelligence_result = await self._run_minimal_intelligence_pipeline()
                latest_intelligence = intelligence_result
            
            # Execute DCA investment
            layer5_result = await self._execute_dca_investment(
                self.config.daily_dca_amount,
                latest_intelligence
            )
            
            result.layer5_result = layer5_result
            result.amount_invested = self.config.daily_dca_amount
            result.orders_executed = layer5_result.get('orders_filled', 0)
            result.workflow_status = "success"
            
            # Update tracking
            self._update_performance_tracking(result)
            
            self.logger.info(f"✅ Daily DCA completed: R${result.amount_invested:.2f} invested")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Daily DCA failed: {e}")
            result.workflow_status = f"failed: {e}"
            result.system_status = SystemStatus.ERROR
            return result
    
    async def run_monthly_rebalance_workflow(self) -> OrchestrationResult:
        """Run monthly portfolio rebalancing workflow"""
        
        self.logger.info(f"⚖️ Running monthly rebalance workflow...")
        
        result = OrchestrationResult(
            workflow_type=WorkflowType.MONTHLY_REBALANCE,
            execution_timestamp=datetime.now()
        )
        
        if not self.config.enable_monthly_rebalance:
            result.workflow_status = "skipped: monthly rebalance disabled"
            return result
        
        try:
            # Check if rebalancing is needed
            current_positions = self._get_current_positions()
            drift_analysis = self._calculate_portfolio_drift(current_positions)
            
            if drift_analysis['max_drift'] < self.config.rebalance_threshold:
                result.workflow_status = f"skipped: drift {drift_analysis['max_drift']:.1%} < {self.config.rebalance_threshold:.1%}"
                return result
            
            # Run full optimization pipeline for rebalancing
            pipeline_result = await self.run_complete_dca_pipeline(0)  # No new investment
            
            # Execute rebalancing trades
            rebalance_trades = self._generate_rebalance_trades(
                current_positions,
                pipeline_result.layer4_result
            )
            
            if rebalance_trades:
                execution_result = await self._execute_rebalance_trades(rebalance_trades)
                result.layer5_result = execution_result
                result.orders_executed = execution_result.get('orders_filled', 0)
            
            result.workflow_status = "success"
            result.performance_warnings.extend(drift_analysis.get('warnings', []))
            
            self.logger.info(f"✅ Monthly rebalance completed: {result.orders_executed} trades executed")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Monthly rebalance failed: {e}")
            result.workflow_status = f"failed: {e}"
            result.system_status = SystemStatus.ERROR
            return result
    
    async def _execute_layer1(self) -> Dict[str, Any]:
        """Execute Layer 1 - Use existing data FIRST, only collect if needed"""
        
        from pathlib import Path
        import os
        from datetime import datetime, timedelta
        
        try:
            # Check existing data first
            data_path = Path("data/altcoin_data")
            
            if data_path.exists():
                data_files = list(data_path.glob("*.csv"))
                
                if len(data_files) > 10:  # We have sufficient data
                    self.logger.info(f"📁 Using existing data: {len(data_files)} files found")
                    
                    # Check if data is recent (within last 24 hours)
                    newest_file = max(data_files, key=lambda f: f.stat().st_mtime)
                    file_age_hours = (datetime.now().timestamp() - newest_file.stat().st_mtime) / 3600
                    
                    if file_age_hours < 24:  # Data is recent
                        self.logger.info(f"📈 Data is recent ({file_age_hours:.1f} hours old) - skipping download")
                        
                        # Return mock Layer 1 result indicating we're using existing data
                        return {
                            'status': 'success',
                            'symbols_collected': len(data_files),
                            'data_points': 0,  # Not downloading new data
                            'collection_duration': 0,  # No collection time
                            'quality_score': 0.95,  # Assume good quality for existing data
                            'data_source': 'existing_files',
                            'ingestion_result': None  # Layer 2 will load files directly
                        }
                    else:
                        self.logger.info(f"📥 Data is stale ({file_age_hours:.1f} hours old) - running incremental update")
                        
            # Only collect data if we don't have it or it's stale
            from ..layer1_data_ingestion import Layer1DataProcessor, Layer1Config
            
            # Initialize real Layer 1 processor for incremental update
            layer1_processor = Layer1DataProcessor()
            
            # Execute real data ingestion
            ingestion_result = await layer1_processor.process_data_ingestion()
            
            # Return real results
            return {
                'status': 'success',
                'symbols_collected': ingestion_result.symbols_successful,
                'data_points': ingestion_result.total_data_points,
                'collection_duration': ingestion_result.processing_duration.total_seconds() if ingestion_result.processing_duration else 0,
                'quality_score': ingestion_result.overall_quality_score,
                'data_source': 'api_collection',
                'ingestion_result': ingestion_result  # Full result for Layer 2
            }
            
        except Exception as e:
            self.logger.error(f"Real Layer 1 execution failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'symbols_collected': 0,
                'data_points': 0,
                'collection_duration': 0,
                'quality_score': 0.0
            }
    
    async def _execute_layer2(self, layer1_result: Dict) -> Dict[str, Any]:
        """Execute Layer 2 data processing using REAL implementation"""
        
        # Import Layer 2 real processor
        from ..layer2_data_processing import process_layer1_data, Layer2Config
        
        try:
            if layer1_result.get('status') != 'success':
                raise Exception("Layer 1 failed - cannot proceed with Layer 2")
            
            # Execute real data processing using Layer 1 results
            # Use the data path where Layer 1 stored the data
            market_data_path = "data"  # This is where Layer 1 stores the CSV files
            processing_result = await process_layer1_data(market_data_path)
            
            return {
                'status': 'success',
                'features_created': len(processing_result.layer3_ready_features.columns) if hasattr(processing_result, 'layer3_ready_features') else 0,
                'assets_processed': layer1_result.get('symbols_collected', 0),
                'processing_duration': processing_result.processing_stats.get('processing_time_seconds', 0),
                'data_quality': processing_result.quality_report.get('overall_score', 0.0),
                'processing_result': processing_result  # Full result for Layer 3
            }
            
        except Exception as e:
            self.logger.error(f"Real Layer 2 execution failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'features_created': 0,
                'assets_processed': 0,
                'processing_duration': 0,
                'data_quality': 0.0
            }
    
    async def _execute_layer3(self, layer2_result: Dict) -> Dict[str, Any]:
        """Execute Layer 3 intelligence using REAL implementation"""
        
        # Import Layer 3 real processor
        from ..layer3_intelligence import Layer3IntelligenceProcessor, Layer3Config
        
        try:
            if layer2_result.get('status') != 'success':
                raise Exception("Layer 2 failed - cannot proceed with Layer 3")
            
            # Initialize real Layer 3 processor
            layer3_processor = Layer3IntelligenceProcessor()
            
            # Extract and convert Layer 2 results to proper format for Layer 3
            processing_result = layer2_result.get('processing_result')
            
            # Convert ProcessingResult to Dict[str, pd.DataFrame] format expected by Layer 3
            market_data = self._convert_processing_result_to_market_data(processing_result)
            
            # Execute real intelligence processing with properly formatted data
            intelligence_result = await layer3_processor.process_intelligence(market_data)
            
            # Extract portfolio recommendations
            portfolio_weights = intelligence_result.recommended_portfolio
            top_assets = list(sorted(portfolio_weights.items(), key=lambda x: x[1], reverse=True)[:4])
            
            return {
                'status': 'success',
                'assets_scored': intelligence_result.assets_processed,
                'top_opportunities': [asset[0] for asset in top_assets],
                'market_regime': intelligence_result.market_regime.regime_type.value if intelligence_result.market_regime else 'unknown',
                'regime_confidence': intelligence_result.market_regime.confidence if intelligence_result.market_regime else 0.0,
                'recommended_portfolio': portfolio_weights,
                'intelligence_result': intelligence_result  # Full result for Layer 4
            }
            
        except Exception as e:
            self.logger.error(f"Real Layer 3 execution failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'assets_scored': 0,
                'top_opportunities': [],
                'market_regime': 'unknown',
                'regime_confidence': 0.0,
                'recommended_portfolio': {}
            }
    
    def _convert_processing_result_to_market_data(self, processing_result) -> Dict[str, pd.DataFrame]:
        """Convert ProcessingResult to Dict[str, pd.DataFrame] format expected by Layer 3"""
        try:
            if not processing_result:
                return {}
            
            # Use layer3_ready_features as the main data source
            if hasattr(processing_result, 'layer3_ready_features') and processing_result.layer3_ready_features is not None:
                features_df = processing_result.layer3_ready_features
            elif hasattr(processing_result, 'transformed_data') and processing_result.transformed_data is not None:
                features_df = processing_result.transformed_data
            else:
                self.logger.warning("No usable data found in ProcessingResult")
                return {}
            
            market_data = {}
            
            # Extract unique asset symbols from column names
            # Column format: SYMBOL_feature (e.g., BTC_close, ETH_returns, etc.)
            asset_columns = {}
            for column in features_df.columns:
                if '_' in column:
                    # Split on first underscore to get symbol
                    parts = column.split('_', 1)
                    if len(parts) >= 2:
                        symbol = parts[0]
                        feature = parts[1]
                        if symbol not in asset_columns:
                            asset_columns[symbol] = []
                        asset_columns[symbol].append(column)
            
            # Create individual DataFrames for each asset
            for symbol, columns in asset_columns.items():
                try:
                    # Get all columns for this asset
                    asset_data = features_df[columns].copy()
                    
                    # Rename columns to remove symbol prefix for cleaner names
                    asset_data.columns = [col.replace(f'{symbol}_', '') for col in asset_data.columns]
                    
                    # Only include assets with sufficient data
                    if len(asset_data.dropna()) > 10:  # Need at least 10 valid rows
                        market_data[symbol] = asset_data
                    
                except Exception as e:
                    self.logger.warning(f"Could not process data for asset {symbol}: {e}")
                    continue
            
            self.logger.info(f"Converted ProcessingResult to market_data format for {len(market_data)} assets")
            return market_data
            
        except Exception as e:
            self.logger.error(f"Failed to convert ProcessingResult to market_data: {e}")
            return {}
    
    async def _execute_layer4(self, layer3_result: Dict, layer2_result: Dict) -> Dict[str, Any]:
        """Execute Layer 4 portfolio optimization using REAL implementation"""
        
        # Import Layer 4 real processor
        from ..layer4_portfolio import Layer4PortfolioProcessor, Layer4Config
        
        try:
            if layer3_result.get('status') != 'success':
                raise Exception("Layer 3 failed - cannot proceed with Layer 4")
            
            # Initialize real Layer 4 processor
            layer4_processor = Layer4PortfolioProcessor()
            
            # Execute real portfolio optimization using Layer 3 results
            intelligence_result = layer3_result.get('intelligence_result')
            
            # Layer 4 also needs market_data - get it from Layer 2 results
            layer2_processing_result = layer2_result.get('processing_result')
            market_data = self._convert_processing_result_to_market_data(layer2_processing_result)
            
            optimization_result = await layer4_processor.optimize_portfolio(intelligence_result, market_data)
            
            return {
                'status': 'success',
                'optimal_weights': optimization_result.optimal_weights,
                'expected_return': optimization_result.expected_return,
                'expected_volatility': optimization_result.expected_volatility,
                'sharpe_ratio': optimization_result.sharpe_ratio,
                'diversification_ratio': optimization_result.diversification_ratio,
                'assets_included': len(optimization_result.optimal_weights),
                'optimization_result': optimization_result  # Full result for Layer 5
            }
            
        except Exception as e:
            self.logger.error(f"Real Layer 4 execution failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'optimal_weights': {},
                'expected_return': 0.0,
                'expected_volatility': 0.0,
                'sharpe_ratio': 0.0,
                'diversification_ratio': 1.0,
                'assets_included': 0
            }
    
    async def _execute_layer5(self, layer4_result: Dict, amount: float) -> Dict[str, Any]:
        """Execute Layer 5 order execution using REAL implementation"""
        
        # Import Layer 5 real processor
        from ..layer5_execution import Layer5ExecutionProcessor, Layer5Config
        
        try:
            if layer4_result.get('status') != 'success':
                raise Exception("Layer 4 failed - cannot proceed with Layer 5")
            
            # Initialize real Layer 5 processor
            layer5_processor = Layer5ExecutionProcessor()
            
            # Execute real DCA investment using Layer 4 results
            optimal_weights = layer4_result.get('optimal_weights', {})
            execution_result = await layer5_processor.execute_dca_investment(amount, optimal_weights)
            
            return {
                'status': 'success',
                'orders_submitted': execution_result.orders_submitted,
                'orders_filled': execution_result.orders_filled,
                'orders_failed': execution_result.orders_failed,
                'total_value_executed': execution_result.total_value_executed,
                'average_slippage': execution_result.average_slippage,
                'execution_time': execution_result.execution_time_seconds,
                'execution_result': execution_result  # Full result for analysis
            }
            
        except Exception as e:
            self.logger.error(f"Real Layer 5 execution failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'orders_submitted': 0,
                'orders_filled': 0,
                'orders_failed': 0,
                'total_value_executed': 0.0,
                'average_slippage': 0.0,
                'execution_time': 0.0
            }
    
    async def _execute_dca_investment(self, amount: float, intelligence: Dict) -> Dict[str, Any]:
        """Execute DCA investment (mock)"""
        
        await asyncio.sleep(0.5)
        
        portfolio = intelligence.get('recommended_portfolio', {})
        
        return {
            'status': 'success',
            'orders_submitted': len(portfolio),
            'orders_filled': len(portfolio),
            'total_value_executed': amount,
            'dca_allocation': portfolio
        }
    
    def _process_pipeline_results(self, result: OrchestrationResult) -> OrchestrationResult:
        """Process and enrich pipeline results"""
        
        # Extract key metrics
        if result.layer1_result:
            result.total_assets_processed = result.layer1_result.get('symbols_collected', 0)
        
        if result.layer4_result:
            result.sharpe_ratio = result.layer4_result.get('sharpe_ratio', 0)
        
        if result.layer5_result:
            result.orders_executed = result.layer5_result.get('orders_filled', 0)
            result.amount_invested = result.layer5_result.get('total_value_executed', 0)
        
        # Calculate performance metrics (mock)
        result.portfolio_value = 10000 + result.amount_invested  # Mock portfolio value
        result.daily_return = np.random.normal(0.002, 0.02)      # Mock daily return
        result.total_return = 0.15                               # Mock total return
        result.max_drawdown = 0.12                              # Mock max drawdown
        
        # Risk metrics
        result.risk_metrics = {
            'var_95': 0.032,
            'volatility': result.layer4_result.get('expected_volatility', 0.6) if result.layer4_result else 0.6,
            'beta': 1.2,
            'correlation_to_btc': 0.75
        }
        
        # Performance attribution (mock)
        result.performance_attribution = {
            'asset_selection': 0.003,
            'market_timing': -0.001,
            'execution': -0.0005,
            'market_return': 0.008
        }
        
        # System health check
        result.system_status = self._assess_system_health(result)
        
        return result
    
    def _assess_system_health(self, result: OrchestrationResult) -> SystemStatus:
        """Assess overall system health"""
        
        alerts = []
        
        # Check execution quality
        if result.layer5_result:
            failure_rate = result.layer5_result.get('orders_failed', 0) / max(result.layer5_result.get('orders_submitted', 1), 1)
            if failure_rate > self.config.alert_thresholds['execution_failure_rate']:
                alerts.append(f"High execution failure rate: {failure_rate:.1%}")
        
        # Check performance
        if result.daily_return < -self.config.alert_thresholds['daily_loss']:
            alerts.append(f"Large daily loss: {result.daily_return:.2%}")
        
        if result.max_drawdown > self.config.alert_thresholds['max_drawdown']:
            alerts.append(f"High drawdown: {result.max_drawdown:.1%}")
        
        # Update alerts
        result.health_alerts.extend(alerts)
        
        # Determine status
        if len(alerts) > 2:
            return SystemStatus.ERROR
        elif len(alerts) > 0:
            return SystemStatus.WARNING
        else:
            return SystemStatus.HEALTHY
    
    def _update_performance_tracking(self, result: OrchestrationResult):
        """Update performance tracking history"""
        
        if not self.config.enable_performance_tracking:
            return
        
        performance_record = {
            'timestamp': result.execution_timestamp,
            'portfolio_value': result.portfolio_value,
            'daily_return': result.daily_return,
            'amount_invested': result.amount_invested,
            'sharpe_ratio': result.sharpe_ratio,
            'workflow_type': result.workflow_type.value
        }
        
        self._performance_history.append(performance_record)
        
        # Keep only recent history
        if len(self._performance_history) > self.config.performance_lookback_days:
            self._performance_history.pop(0)
    
    def _update_system_health(self):
        """Update system health metrics"""
        
        self._last_health_check = datetime.now()
        
        # Calculate system metrics
        self._system_metrics = {
            'active_workflows': len(self._active_workflows),
            'total_workflows_today': len([w for w in self._workflow_history 
                                         if w.execution_timestamp.date() == datetime.now().date()]),
            'avg_execution_time': np.mean([w.workflow_duration.total_seconds() 
                                          for w in self._workflow_history[-10:] 
                                          if w.workflow_duration]) if self._workflow_history else 0,
            'success_rate': np.mean([1 if 'success' in w.workflow_status else 0 
                                    for w in self._workflow_history[-50:]]) if self._workflow_history else 1
        }
    
    def get_system_status_report(self) -> Dict[str, Any]:
        """Get comprehensive system status report"""
        
        # Current system status
        latest_workflow = self._workflow_history[-1] if self._workflow_history else None
        
        report = {
            'system_overview': {
                'status': latest_workflow.system_status.value if latest_workflow else 'unknown',
                'last_execution': latest_workflow.execution_timestamp.isoformat() if latest_workflow else None,
                'active_workflows': len(self._active_workflows),
                'health_check_time': self._last_health_check.isoformat() if self._last_health_check else None
            },
            'performance_summary': {
                'portfolio_value': latest_workflow.portfolio_value if latest_workflow else 0,
                'total_return': latest_workflow.total_return if latest_workflow else 0,
                'sharpe_ratio': latest_workflow.sharpe_ratio if latest_workflow else 0,
                'max_drawdown': latest_workflow.max_drawdown if latest_workflow else 0
            },
            'recent_activity': {
                'workflows_today': len([w for w in self._workflow_history 
                                       if w.execution_timestamp.date() == datetime.now().date()]),
                'amount_invested_today': sum(w.amount_invested for w in self._workflow_history 
                                           if w.execution_timestamp.date() == datetime.now().date()),
                'orders_executed_today': sum(w.orders_executed for w in self._workflow_history 
                                            if w.execution_timestamp.date() == datetime.now().date())
            },
            'system_metrics': self._system_metrics,
            'alerts': latest_workflow.health_alerts if latest_workflow else [],
            'warnings': latest_workflow.performance_warnings if latest_workflow else []
        }
        
        return report
    
    def get_performance_analytics(self) -> Dict[str, Any]:
        """Get detailed performance analytics"""
        
        if not self._performance_history:
            return {'error': 'No performance history available'}
        
        df = pd.DataFrame(self._performance_history)
        
        analytics = {
            'period_summary': {
                'start_date': df['timestamp'].min().isoformat(),
                'end_date': df['timestamp'].max().isoformat(),
                'total_days': len(df),
                'total_invested': df['amount_invested'].sum(),
                'final_portfolio_value': df['portfolio_value'].iloc[-1]
            },
            'return_metrics': {
                'total_return': (df['portfolio_value'].iloc[-1] / df['portfolio_value'].iloc[0]) - 1,
                'annualized_return': df['daily_return'].mean() * 252,
                'annualized_volatility': df['daily_return'].std() * np.sqrt(252),
                'sharpe_ratio': (df['daily_return'].mean() * 252) / (df['daily_return'].std() * np.sqrt(252)),
                'max_drawdown': (df['portfolio_value'] / df['portfolio_value'].expanding().max() - 1).min()
            },
            'workflow_statistics': {
                'total_workflows': len(df),
                'dca_workflows': len(df[df['workflow_type'] == 'daily_dca']),
                'rebalance_workflows': len(df[df['workflow_type'] == 'monthly_rebalance']),
                'pipeline_workflows': len(df[df['workflow_type'] == 'full_pipeline'])
            }
        }
        
        return analytics


# Helper functions for integration
async def run_dca_system_orchestration(dca_amount: Optional[float] = None,
                                      config: Optional[Layer6Config] = None) -> OrchestrationResult:
    """
    Run the complete DCA system orchestration
    
    Args:
        dca_amount: Optional DCA investment amount
        config: Optional Layer 6 configuration
        
    Returns:
        OrchestrationResult with complete system execution
    """
    orchestrator = Layer6OrchestrationProcessor(config)
    return await orchestrator.run_complete_dca_pipeline(dca_amount)


# Layer 6 Orchestration is integrated into the main DCA system
# Use the main CLI for testing: python cli.py run
