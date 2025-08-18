#!/usr/bin/env python3
"""
DCA Automated System - Main System Orchestrator

Unified orchestrator for the complete DCA system that coordinates all layers:
- Layer 1: Data Ingestion & Storage
- Layer 2: Data Processing & Analytics  
- Layer 3: Intelligence & Scoring (future)
- Layer 4: Portfolio Optimization (future)
- Layer 5: Execution & Order Management (future)
- Layer 6: Orchestration & Monitoring (future)

This is the single entry point for the entire DCA system.
"""

import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass

# Layer imports
from src.layer1_data_ingestion import Layer1DataProcessor, Layer1Config, CollectionMode
from src.layer2_data_processing import Layer2DataProcessor, process_layer1_data
from src.layer3_intelligence import Layer3IntelligenceProcessor, Layer3Config
from src.layer4_portfolio import Layer4PortfolioProcessor, Layer4Config
from src.layer5_execution import Layer5ExecutionProcessor, Layer5Config
from src.layer6_orchestration import Layer6OrchestrationProcessor, Layer6Config


@dataclass
class SystemConfig:
    """Unified system configuration"""
    
    # Data paths
    data_path: str = "data"
    
    # Layer 1 settings
    collection_mode: CollectionMode = CollectionMode.INCREMENTAL
    
    # Layer 2 settings
    enable_layer2: bool = True
    feature_selection: bool = True
    max_features_per_asset: int = 100
    
    # Layer 3 settings
    enable_layer3: bool = True
    min_composite_score: float = 0.3
    
    # Layer 4 settings
    enable_layer4: bool = True
    target_return: Optional[float] = None
    max_individual_weight: float = 0.05
    
    # Layer 5 settings
    enable_layer5: bool = True
    max_slippage_tolerance: float = 0.02
    
    # Layer 6 settings
    enable_layer6: bool = True
    enable_orchestration: bool = True
    
    # System settings
    log_level: str = "INFO"
    save_results: bool = True


class DCASystem:
    """
    Main DCA System Orchestrator
    
    Coordinates all layers to provide a complete DCA investment system.
    Single entry point for data collection, processing, and analysis.
    """
    
    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()
        self.logger = self._setup_logging()
        
        # Initialize layers
        self.layer1_processor = None
        self.layer2_processor = None
        self.layer3_intelligence = None
        self.layer4_portfolio = None
        self.layer5_execution = None
        self.layer6_orchestration = None
        
        # System state
        self.last_run_stats = {}
        
    def _setup_logging(self) -> logging.Logger:
        """Setup system-wide logging"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        logger = logging.getLogger("DCASystem")
        logger.info(f"🚀 DCA System initialized with config: {self.config}")
        return logger
    
    async def run_complete_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete DCA pipeline through all 6 layers
        
        Returns:
            Dictionary with results from all layers
        """
        self.logger.info("🎯 Starting complete DCA pipeline (6-layer architecture)...")
        
        # Use Layer 6 orchestration if enabled
        if self.config.enable_layer6 and self.config.enable_orchestration:
            return await self._run_orchestrated_pipeline()
        
        # Otherwise run traditional layer-by-layer execution
        return await self._run_sequential_pipeline()
    
    async def run_data_collection_only(self) -> Dict[str, Any]:
        """Run only Layer 1 data collection"""
        self.logger.info("📊 Running data collection only...")
        
        try:
            layer1_results = await self._run_layer1()
            return {'layer1_results': layer1_results, 'mode': 'data_collection_only'}
        except Exception as e:
            self.logger.error(f"❌ Data collection failed: {e}")
            return {'error': str(e), 'mode': 'data_collection_only'}
    
    async def run_processing_only(self) -> Dict[str, Any]:
        """Run only Layer 2 data processing (assumes Layer 1 data exists)"""
        self.logger.info("🔄 Running data processing only...")
        
        try:
            layer2_results = await self._run_layer2()
            return {'layer2_results': layer2_results, 'mode': 'processing_only'}
        except Exception as e:
            self.logger.error(f"❌ Data processing failed: {e}")
            return {'error': str(e), 'mode': 'processing_only'}
    
    async def _run_layer1(self) -> Optional[Any]:
        """Execute Layer 1 data processing"""
        
        # Initialize Layer 1 processor if not already done
        if self.layer1_processor is None:
            layer1_config = Layer1Config(
                collection_mode=self.config.collection_mode,
                enable_quality_filtering=True,
                enable_data_storage=True
            )
            self.layer1_processor = Layer1DataProcessor(layer1_config)
        
        # Run data ingestion
        ingestion_result = await self.layer1_processor.process_data_ingestion()
        
        self.logger.info(f"Layer 1 completed: {ingestion_result.symbols_successful}/{ingestion_result.symbols_processed} assets processed")
        return ingestion_result
    
    async def _run_layer2(self) -> Optional[Any]:
        """Execute Layer 2 data processing"""
        
        # Use the data path from system config
        data_path = Path(self.config.data_path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {data_path}")
        
        # Process Layer 1 data through Layer 2
        from src.layer2_data_processing import Layer2Config
        
        layer2_config = Layer2Config(
            feature_selection=self.config.feature_selection,
            max_features_per_asset=self.config.max_features_per_asset,
            save_intermediate_results=self.config.save_results
        )
        
        layer2_results = await process_layer1_data(str(data_path), layer2_config)
        
        self.logger.info(f"Layer 2 completed: {layer2_results.processing_stats['features_created']} features generated")
        return layer2_results
    
    async def _run_layer3(self, layer2_results: Any) -> Optional[Any]:
        """Execute Layer 3 intelligence processing"""
        
        if self.layer3_intelligence is None:
            layer3_config = Layer3Config(
                min_composite_score=self.config.min_composite_score
            )
            self.layer3_intelligence = Layer3IntelligenceProcessor(layer3_config)
        
        # Mock market data (in production, would come from Layer 2)
        mock_market_data = {}  # Would be populated from layer2_results
        
        intelligence_results = await self.layer3_intelligence.process_intelligence(mock_market_data)
        
        self.logger.info(f"Layer 3 completed: {intelligence_results.assets_processed} assets analyzed")
        return intelligence_results
    
    async def _run_layer4(self, layer3_results: Any) -> Optional[Any]:
        """Execute Layer 4 portfolio optimization"""
        
        if self.layer4_portfolio is None:
            layer4_config = Layer4Config(
                target_return=self.config.target_return,
                max_individual_weight=self.config.max_individual_weight
            )
            self.layer4_portfolio = Layer4PortfolioProcessor(layer4_config)
        
        # Convert intelligence results to format needed for optimization
        intelligence_data = {
            'asset_scores': getattr(layer3_results, 'asset_scores', {}),
            'asset_classifications': getattr(layer3_results, 'asset_classifications', {}),
            'recommended_portfolio': getattr(layer3_results, 'recommended_portfolio', {})
        }
        
        # Mock market data (in production, would come from Layer 2)
        mock_market_data = {}  # Would be populated from layer2_results
        
        optimization_results = await self.layer4_portfolio.optimize_portfolio(
            intelligence_data, mock_market_data
        )
        
        self.logger.info(f"Layer 4 completed: {optimization_results.assets_included} assets optimized, Sharpe={optimization_results.sharpe_ratio:.3f}")
        return optimization_results
    
    async def _run_layer5(self, layer4_results: Any) -> Optional[Any]:
        """Execute Layer 5 trade execution"""
        
        if self.layer5_execution is None:
            layer5_config = Layer5Config(
                max_slippage_tolerance=self.config.max_slippage_tolerance
            )
            self.layer5_execution = Layer5ExecutionProcessor(layer5_config)
        
        # Extract target portfolio from optimization
        target_portfolio = getattr(layer4_results, 'optimal_weights', {})
        available_capital = 1000.0  # Daily DCA amount
        
        execution_results = await self.layer5_execution.execute_portfolio_changes(
            target_portfolio, {}, available_capital
        )
        
        self.logger.info(f"Layer 5 completed: {execution_results.orders_filled} orders filled, R${execution_results.total_value_executed:.2f} executed")
        return execution_results
    
    def _calculate_pipeline_stats(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall pipeline statistics"""
        
        pipeline_end = datetime.now()
        total_duration = pipeline_end - results['pipeline_start']
        
        stats = {
            'total_duration': total_duration,
            'pipeline_end': pipeline_end,
            'layers_executed': []
        }
        
        # Layer 1 stats
        if results.get('layer1_results'):
            layer1_stats = results['layer1_results']
            stats['layers_executed'].append('layer1')
            stats['layer1'] = {
                'assets_collected': getattr(layer1_stats, 'symbols_successful', 0),
                'collection_mode': self.config.collection_mode.value,
                'data_points': getattr(layer1_stats, 'total_data_points', 0)
            }
        
        # Layer 2 stats  
        if results.get('layer2_results'):
            layer2_stats = results['layer2_results']
            stats['layers_executed'].append('layer2')
            stats['layer2'] = {
                'features_created': layer2_stats.processing_stats.get('features_created', 0),
                'assets_processed': layer2_stats.processing_stats.get('assets_processed', 0),
                'processing_duration': layer2_stats.processing_stats.get('duration', None)
            }
        
        return stats
    
    async def _run_orchestrated_pipeline(self) -> Dict[str, Any]:
        """Run pipeline through Layer 6 orchestration"""
        
        # Initialize Layer 6 orchestrator
        if self.layer6_orchestration is None:
            layer6_config = Layer6Config(
                daily_dca_amount=1000.0 / 30,  # Monthly amount / 30 days
                enable_daily_dca=True,
                enable_monthly_rebalance=True
            )
            self.layer6_orchestration = Layer6OrchestrationProcessor(layer6_config)
        
        # Run complete orchestrated pipeline
        orchestration_result = await self.layer6_orchestration.run_complete_dca_pipeline()
        
        return {
            'orchestration_result': orchestration_result,
            'workflow_type': orchestration_result.workflow_type.value,
            'system_status': orchestration_result.system_status.value,
            'pipeline_stats': orchestration_result.orchestration_stats,
            'performance_metrics': {
                'portfolio_value': orchestration_result.portfolio_value,
                'sharpe_ratio': orchestration_result.sharpe_ratio,
                'total_return': orchestration_result.total_return
            }
        }
    
    async def _run_sequential_pipeline(self) -> Dict[str, Any]:
        """Run traditional sequential pipeline through all layers"""
        
        pipeline_start = datetime.now()
        
        results = {
            'pipeline_start': pipeline_start,
            'layer1_results': None,
            'layer2_results': None,
            'layer3_results': None,
            'layer4_results': None,
            'layer5_results': None,
            'pipeline_stats': {}
        }
        
        try:
            # Layer 1: Data Collection
            self.logger.info("📊 Layer 1: Data Collection...")
            layer1_results = await self._run_layer1()
            results['layer1_results'] = layer1_results
            
            # Layer 2: Data Processing
            if self.config.enable_layer2 and layer1_results:
                self.logger.info("🔄 Layer 2: Data Processing...")
                layer2_results = await self._run_layer2()
                results['layer2_results'] = layer2_results
            
            # Layer 3: Intelligence & Scoring
            if self.config.enable_layer3 and results.get('layer2_results'):
                self.logger.info("🧠 Layer 3: Intelligence & Scoring...")
                layer3_results = await self._run_layer3(results['layer2_results'])
                results['layer3_results'] = layer3_results
            
            # Layer 4: Portfolio Optimization
            if self.config.enable_layer4 and results.get('layer3_results'):
                self.logger.info("📈 Layer 4: Portfolio Optimization...")
                layer4_results = await self._run_layer4(results['layer3_results'])
                results['layer4_results'] = layer4_results
            
            # Layer 5: Execution
            if self.config.enable_layer5 and results.get('layer4_results'):
                self.logger.info("⚡ Layer 5: Execution...")
                layer5_results = await self._run_layer5(results['layer4_results'])
                results['layer5_results'] = layer5_results
            
            # Calculate pipeline statistics
            results['pipeline_stats'] = self._calculate_pipeline_stats(results)
            
            # Save results if configured
            if self.config.save_results:
                await self._save_system_results(results)
            
            self.logger.info("✅ Complete 6-layer DCA pipeline finished successfully!")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ DCA pipeline failed: {e}")
            results['error'] = str(e)
            return results
    
    async def _save_system_results(self, results: Dict[str, Any]):
        """Save system results to disk"""
        
        results_dir = Path(self.config.data_path) / "system_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = results_dir / f"dca_system_run_{timestamp}.json"
        
        # Convert non-serializable objects to strings
        import json
        
        def serialize_results(obj):
            if hasattr(obj, 'to_dict'):
                return obj.to_dict()
            elif hasattr(obj, '__dict__'):
                return {k: serialize_results(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, (datetime, )):
                return obj.isoformat()
            elif hasattr(obj, 'total_seconds'):  # timedelta
                return obj.total_seconds()
            else:
                return str(obj)
        
        try:
            serializable_results = serialize_results(results)
            
            with open(results_file, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            
            self.logger.info(f"System results saved to {results_file}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save system results: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        
        data_path = Path(self.config.data_path)
        
        status = {
            'system_config': self.config.__dict__,
            'data_path_exists': data_path.exists(),
            'layers_available': {
                'layer1': True,  # Data Ingestion & Storage
                'layer2': True,  # Data Processing & Analytics  
                'layer3': True,  # Intelligence & Scoring
                'layer4': True,  # Portfolio Optimization
                'layer5': True,  # Execution & Order Management
                'layer6': True,  # Orchestration & Monitoring
            }
        }
        
        # Check data availability
        if data_path.exists():
            altcoin_data = data_path / "altcoin_data"
            major_crypto_data = data_path / "major_crypto_data"
            
            status['data_available'] = {
                'altcoin_files': len(list(altcoin_data.glob('*.csv'))) if altcoin_data.exists() else 0,
                'major_crypto_files': len(list(major_crypto_data.glob('*.csv'))) if major_crypto_data.exists() else 0,
                'total_assets': 0
            }
            
            status['data_available']['total_assets'] = (
                status['data_available']['altcoin_files'] + 
                status['data_available']['major_crypto_files']
            )
        
        return status


# Convenience functions for external use
async def run_dca_system(config: Optional[SystemConfig] = None) -> Dict[str, Any]:
    """
    Convenience function to run the complete DCA system
    
    Args:
        config: Optional system configuration
        
    Returns:
        Dictionary with results from all layers
    """
    system = DCASystem(config)
    return await system.run_complete_pipeline()


async def collect_market_data(collection_mode: CollectionMode = CollectionMode.INCREMENTAL) -> Dict[str, Any]:
    """
    Convenience function to run only data collection
    
    Args:
        collection_mode: Data collection mode
        
    Returns:
        Dictionary with Layer 1 results
    """
    config = SystemConfig(collection_mode=collection_mode, enable_layer2=False)
    system = DCASystem(config)
    return await system.run_data_collection_only()


if __name__ == "__main__":
    # Quick system test
    async def main():
        print("🚀 DCA System Test")
        print("=" * 30)
        
        # Create system
        config = SystemConfig(
            collection_mode=CollectionMode.INCREMENTAL,
            enable_layer2=True,
            log_level="INFO"
        )
        
        system = DCASystem(config)
        
        # Show system status
        status = system.get_system_status()
        print(f"📊 System Status:")
        print(f"  Data path exists: {status['data_path_exists']}")
        print(f"  Available assets: {status.get('data_available', {}).get('total_assets', 0)}")
        print(f"  Available layers: {[k for k, v in status['layers_available'].items() if v]}")
        
        # Run system (comment out for quick test)
        # results = await system.run_complete_pipeline()
        # print(f"\n✅ System test completed!")
        
    asyncio.run(main())
