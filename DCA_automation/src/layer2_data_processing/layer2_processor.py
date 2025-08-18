#!/usr/bin/env python3
"""
Layer 2 Data Processing Orchestrator

Main orchestrator for Layer 2 data processing that integrates all Layer 2 modules:
- Technical Indicators Engine
- Statistical Analysis Engine  
- Data Transformation Pipeline

Provides clean interface between Layer 1 (data ingestion) and Layer 3 (intelligence).
Processes raw market data into analyzed features ready for portfolio optimization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
from pathlib import Path
import logging
import asyncio
from datetime import datetime, timedelta

# Layer 2 modules
from .technical_indicators import TechnicalIndicatorsEngine, IndicatorConfig
from .statistical_analysis import StatisticalAnalysisEngine, StatisticalConfig
from .data_pipeline import DataTransformationPipeline, PipelineConfig, TransformationResult

# Layer 1 integration
import sys
sys.path.append(str(Path(__file__).parent.parent))
from layer1_data_ingestion.config import get_config as get_layer1_config

logger = logging.getLogger(__name__)


@dataclass
class Layer2Config:
    """Complete configuration for Layer 2 processing"""
    
    # Module configurations
    indicators_config: IndicatorConfig = field(default_factory=IndicatorConfig)
    statistics_config: StatisticalConfig = field(default_factory=StatisticalConfig) 
    pipeline_config: PipelineConfig = field(default_factory=PipelineConfig)
    
    # Processing settings
    batch_size: int = 50  # Process assets in batches
    parallel_processing: bool = True
    save_intermediate_results: bool = True
    
    # Output settings
    output_path: Optional[str] = None  # Where to save processed data
    feature_selection: bool = True  # Apply feature selection
    max_features_per_asset: int = 100  # Limit features per asset


@dataclass
class ProcessingResult:
    """Result of Layer 2 processing"""
    
    # Processed data
    transformed_data: pd.DataFrame
    technical_indicators: pd.DataFrame
    statistical_metrics: pd.DataFrame
    
    # Analysis results
    correlation_analysis: Dict[str, pd.DataFrame]
    risk_metrics: pd.DataFrame
    cointegration_analysis: Dict[str, Any]
    
    # Metadata
    feature_metadata: Dict[str, Any]
    quality_report: Dict[str, Any]
    processing_stats: Dict[str, Any]
    
    # Summary for Layer 3
    layer3_ready_features: pd.DataFrame
    asset_scores_preview: Dict[str, float]


class Layer2DataProcessor:
    """
    Main orchestrator for Layer 2 data processing
    
    Coordinates all Layer 2 modules to transform raw market data from Layer 1
    into processed features and analytics ready for Layer 3 intelligence systems.
    """
    
    def __init__(self, config: Optional[Layer2Config] = None):
        self.config = config or Layer2Config()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize processing engines
        self.indicators_engine = TechnicalIndicatorsEngine(self.config.indicators_config)
        self.statistics_engine = StatisticalAnalysisEngine(self.config.statistics_config)
        self.pipeline = DataTransformationPipeline(self.config.pipeline_config)
        
        # Set output path
        if self.config.output_path is None:
            layer1_config = get_layer1_config()
            self.config.output_path = str(Path(layer1_config.storage.base_path) / "layer2_processed")
        
        # Ensure output directory exists
        Path(self.config.output_path).mkdir(parents=True, exist_ok=True)
    
    async def process_market_data(self, 
                                market_data: Dict[str, pd.DataFrame]) -> ProcessingResult:
        """
        Process complete market data through all Layer 2 systems
        
        Args:
            market_data: Dictionary of {symbol: OHLCV_dataframe} from Layer 1
            
        Returns:
            ProcessingResult with all processed data and analysis
        """
        self.logger.info(f"🔄 Starting Layer 2 processing for {len(market_data)} assets")
        start_time = datetime.now()
        
        processing_stats = {
            'start_time': start_time,
            'assets_processed': 0,
            'features_created': 0,
            'processing_stages': []
        }
        
        # Stage 1: Data Transformation Pipeline
        self.logger.info("📊 Stage 1: Data transformation and cleaning...")
        transformed_data, transformation_results = await self._transform_market_data(market_data)
        processing_stats['processing_stages'].append('data_transformation')
        
        # Stage 2: Technical Indicators
        self.logger.info("📈 Stage 2: Technical indicators calculation...")
        technical_indicators = await self._calculate_technical_indicators(transformed_data)
        processing_stats['processing_stages'].append('technical_indicators')
        
        # Stage 3: Statistical Analysis
        self.logger.info("🔬 Stage 3: Statistical analysis...")
        statistical_results = await self._perform_statistical_analysis(transformed_data)
        processing_stats['processing_stages'].append('statistical_analysis')
        
        # Stage 4: Feature Engineering and Selection
        self.logger.info("🧠 Stage 4: Feature engineering and selection...")
        layer3_features = await self._prepare_layer3_features(
            transformed_data, technical_indicators, statistical_results
        )
        processing_stats['processing_stages'].append('feature_preparation')
        
        # Stage 5: Quality Assessment and Asset Scoring Preview
        self.logger.info("✅ Stage 5: Quality assessment and preview scoring...")
        quality_report, asset_scores = await self._assess_quality_and_preview_scores(layer3_features)
        processing_stats['processing_stages'].append('quality_assessment')
        
        # Finalize processing stats
        processing_stats.update({
            'end_time': datetime.now(),
            'duration': datetime.now() - start_time,
            'assets_processed': len(market_data),
            'features_created': len(layer3_features.columns),
            'memory_usage_mb': layer3_features.memory_usage(deep=True).sum() / 1024 / 1024
        })
        
        # Save results if configured
        if self.config.save_intermediate_results:
            await self._save_processing_results(layer3_features, processing_stats)
        
        # Create comprehensive result
        result = ProcessingResult(
            transformed_data=transformed_data,
            technical_indicators=technical_indicators,
            statistical_metrics=statistical_results['risk_metrics'],
            correlation_analysis=statistical_results['correlation_analysis'],
            risk_metrics=statistical_results['risk_metrics'],
            cointegration_analysis=statistical_results['cointegration_analysis'],
            feature_metadata=transformation_results.feature_metadata,
            quality_report=quality_report,
            processing_stats=processing_stats,
            layer3_ready_features=layer3_features,
            asset_scores_preview=asset_scores
        )
        
        self.logger.info(f"✅ Layer 2 processing completed in {processing_stats['duration']}")
        self.logger.info(f"📊 Generated {processing_stats['features_created']} features for {processing_stats['assets_processed']} assets")
        
        # 🎨 GENERATE VISUALIZATIONS
        try:
            from ..visualization import PortfolioVisualizer
            visualizer = PortfolioVisualizer()
            
            self.logger.info("🎨 Generating Layer 2 visualizations...")
            layer2_plots = visualizer.visualize_layer2_analysis(result, "layer2")
            
            if layer2_plots:
                self.logger.info(f"✅ Created {len(layer2_plots)} Layer 2 visualizations:")
                for plot_type, file_path in layer2_plots.items():
                    if file_path:
                        self.logger.info(f"  📊 {plot_type}: {file_path}")
                        
        except Exception as e:
            self.logger.error(f"Failed to create Layer 2 visualizations: {e}")
        
        return result
    
    async def _transform_market_data(self, 
                                   market_data: Dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, TransformationResult]:
        """Transform and clean raw market data"""
        
        # Combine all market data into single DataFrame with asset prefixes
        combined_data = pd.DataFrame()
        
        for symbol, data in market_data.items():
            # Add symbol prefix to columns
            renamed_data = data.copy()
            renamed_data.columns = [f"{symbol}_{col}" for col in renamed_data.columns]
            
            if combined_data.empty:
                combined_data = renamed_data
            else:
                combined_data = pd.concat([combined_data, renamed_data], axis=1, sort=True)
        
        # Apply transformation pipeline
        transformation_result = self.pipeline.transform_data(combined_data)
        
        return transformation_result.transformed_data, transformation_result
    
    async def _calculate_technical_indicators(self, 
                                           transformed_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for all assets"""
        
        # Group columns by asset
        asset_indicators = pd.DataFrame(index=transformed_data.index)
        
        # Identify unique assets
        assets = set()
        for col in transformed_data.columns:
            if any(ohlcv in col.lower() for ohlcv in ['open', 'high', 'low', 'close', 'volume']):
                asset_name = col.split('_')[0]
                assets.add(asset_name)
        
        # Calculate indicators for each asset
        for asset in assets:
            # Extract OHLCV columns for this asset
            asset_columns = [col for col in transformed_data.columns if col.startswith(f"{asset}_")]
            ohlcv_columns = []
            
            for base_col in ['open', 'high', 'low', 'close', 'volume']:
                matching_cols = [col for col in asset_columns if col.endswith(f"_{base_col}") or col == f"{asset}_{base_col}"]
                if matching_cols:
                    ohlcv_columns.append(matching_cols[0])
            
            if len(ohlcv_columns) >= 4:  # Need at least OHLC
                # Create OHLCV DataFrame for this asset
                asset_data = transformed_data[ohlcv_columns].copy()
                
                # Rename columns to standard OHLCV format
                column_mapping = {}
                for col in asset_data.columns:
                    if 'open' in col.lower():
                        column_mapping[col] = 'open'
                    elif 'high' in col.lower():
                        column_mapping[col] = 'high'
                    elif 'low' in col.lower():
                        column_mapping[col] = 'low'
                    elif 'close' in col.lower():
                        column_mapping[col] = 'close'
                    elif 'volume' in col.lower():
                        column_mapping[col] = 'volume'
                
                asset_data = asset_data.rename(columns=column_mapping)
                
                # Calculate indicators
                try:
                    asset_indicators_data = self.indicators_engine.calculate_all_indicators(asset_data)
                    
                    # Add asset prefix to indicator columns (skip original OHLCV)
                    for col in asset_indicators_data.columns:
                        if col not in ['open', 'high', 'low', 'close', 'volume']:
                            asset_indicators[f"{asset}_{col}"] = asset_indicators_data[col]
                            
                except Exception as e:
                    self.logger.warning(f"Failed to calculate indicators for {asset}: {e}")
                    continue
        
        return asset_indicators
    
    async def _perform_statistical_analysis(self, 
                                          transformed_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis"""
        
        # Extract price data for analysis
        price_columns = [col for col in transformed_data.columns 
                        if 'close' in col.lower() and not any(x in col.lower() for x in ['ratio', 'momentum'])]
        
        if len(price_columns) == 0:
            self.logger.warning("No price columns found for statistical analysis")
            return {}
        
        price_data = transformed_data[price_columns].dropna()
        
        # Returns analysis
        returns_analysis = self.statistics_engine.calculate_returns_analysis(price_data)
        
        # Extract log returns for further analysis
        log_returns_cols = [col for col in returns_analysis.columns if col.endswith('_log_returns')]
        log_returns = returns_analysis[log_returns_cols].dropna()
        
        # Correlation analysis
        correlation_analysis = {}
        if len(log_returns.columns) > 1:
            correlation_analysis = self.statistics_engine.calculate_correlation_analysis(log_returns)
        
        # Risk metrics
        risk_metrics = pd.DataFrame()
        if not log_returns.empty:
            risk_metrics = self.statistics_engine.calculate_risk_metrics(log_returns)
        
        # Cointegration analysis (if enough assets)
        cointegration_analysis = {}
        if len(price_data.columns) >= 2 and len(price_data) >= 90:
            try:
                cointegration_analysis = self.statistics_engine.calculate_cointegration_analysis(price_data)
            except Exception as e:
                self.logger.warning(f"Cointegration analysis failed: {e}")
        
        return {
            'returns_analysis': returns_analysis,
            'correlation_analysis': correlation_analysis,
            'risk_metrics': risk_metrics,
            'cointegration_analysis': cointegration_analysis
        }
    
    async def _prepare_layer3_features(self, 
                                     transformed_data: pd.DataFrame,
                                     technical_indicators: pd.DataFrame,
                                     statistical_results: Dict[str, Any]) -> pd.DataFrame:
        """Prepare final feature set for Layer 3 intelligence systems"""
        
        # Combine all features
        all_features = pd.concat([
            transformed_data,
            technical_indicators,
            statistical_results.get('returns_analysis', pd.DataFrame()),
            statistical_results.get('risk_metrics', pd.DataFrame())
        ], axis=1)
        
        # Remove duplicate columns
        all_features = all_features.loc[:, ~all_features.columns.duplicated()]
        
        # Feature selection if enabled
        if self.config.feature_selection:
            all_features = await self._select_features(all_features)
        
        # Final cleanup
        all_features = all_features.replace([np.inf, -np.inf], np.nan)
        all_features = all_features.fillna(method='ffill', limit=3)
        all_features = all_features.dropna(how='all')
        
        return all_features
    
    async def _select_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature selection to reduce dimensionality"""
        
        # Simple feature selection based on:
        # 1. Remove features with too many NaNs
        # 2. Remove features with zero variance
        # 3. Limit features per asset if configured
        
        # Remove high-NaN features
        nan_threshold = 0.8
        features_df = features_df.loc[:, features_df.isnull().mean() < nan_threshold]
        
        # Remove zero-variance features
        numeric_features = features_df.select_dtypes(include=[np.number])
        non_zero_var_features = numeric_features.loc[:, numeric_features.var() > 0]
        
        # Keep non-numeric features
        non_numeric_features = features_df.select_dtypes(exclude=[np.number])
        
        features_df = pd.concat([non_zero_var_features, non_numeric_features], axis=1)
        
        # Limit features per asset if configured
        if self.config.max_features_per_asset > 0:
            features_df = self._limit_features_per_asset(features_df)
        
        self.logger.info(f"Feature selection: {features_df.shape[1]} features selected")
        return features_df
    
    def _limit_features_per_asset(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Limit number of features per asset"""
        
        # Group features by asset
        asset_features = {}
        ungrouped_features = []
        
        for col in features_df.columns:
            parts = col.split('_')
            if len(parts) > 1:
                asset = parts[0]
                if asset not in asset_features:
                    asset_features[asset] = []
                asset_features[asset].append(col)
            else:
                ungrouped_features.append(col)
        
        # Select top features per asset (by variance for now)
        selected_columns = ungrouped_features.copy()
        
        for asset, feature_list in asset_features.items():
            if len(feature_list) > self.config.max_features_per_asset:
                # Calculate variance for each feature
                feature_variances = features_df[feature_list].var().sort_values(ascending=False)
                top_features = feature_variances.head(self.config.max_features_per_asset).index.tolist()
                selected_columns.extend(top_features)
            else:
                selected_columns.extend(feature_list)
        
        return features_df[selected_columns]
    
    async def _assess_quality_and_preview_scores(self, 
                                               layer3_features: pd.DataFrame) -> tuple[Dict[str, Any], Dict[str, float]]:
        """Assess data quality and generate preview asset scores"""
        
        # Quality assessment
        quality_report = {
            'feature_count': len(layer3_features.columns),
            'observation_count': len(layer3_features),
            'missing_data_ratio': layer3_features.isnull().sum().sum() / (layer3_features.shape[0] * layer3_features.shape[1]),
            'memory_usage_mb': layer3_features.memory_usage(deep=True).sum() / 1024 / 1024,
            'date_range': {
                'start': layer3_features.index.min() if isinstance(layer3_features.index, pd.DatetimeIndex) else None,
                'end': layer3_features.index.max() if isinstance(layer3_features.index, pd.DatetimeIndex) else None
            }
        }
        
        # Preview asset scores (simple momentum-based scoring for demonstration)
        asset_scores = {}
        
        # Find momentum features for scoring
        momentum_features = [col for col in layer3_features.columns if 'momentum' in col.lower()]
        
        if momentum_features:
            for feature in momentum_features:
                asset_name = feature.split('_')[0]
                if asset_name not in asset_scores:
                    # Simple scoring: average recent momentum
                    recent_momentum = layer3_features[feature].tail(30).mean()
                    asset_scores[asset_name] = float(recent_momentum) if not pd.isna(recent_momentum) else 0.0
        
        return quality_report, asset_scores
    
    async def _save_processing_results(self, 
                                     layer3_features: pd.DataFrame,
                                     processing_stats: Dict[str, Any]):
        """Save processing results to disk"""
        
        output_path = Path(self.config.output_path)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        try:
            # Save main features dataset
            features_file = output_path / f"layer3_features_{timestamp}.parquet"
            layer3_features.to_parquet(features_file)
            
            # Save processing statistics
            stats_file = output_path / f"processing_stats_{timestamp}.json"
            import json
            
            # Convert non-serializable objects
            serializable_stats = processing_stats.copy()
            for key, value in serializable_stats.items():
                if isinstance(value, (datetime, timedelta)):
                    serializable_stats[key] = str(value)
            
            with open(stats_file, 'w') as f:
                json.dump(serializable_stats, f, indent=2)
            
            self.logger.info(f"Saved processing results to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save processing results: {e}")


async def process_layer1_data(market_data_path: str, 
                            config: Optional[Layer2Config] = None) -> ProcessingResult:
    """Process market data from Layer 1 data files"""
    processor = Layer2DataProcessor(config)
    
    # Load market data from Layer 1 files
    data_path = Path(market_data_path)
    market_data = {}
    
    # Load data from altcoin_data and major_crypto_data directories
    for subdir in ['altcoin_data', 'major_crypto_data']:
        subdir_path = data_path / subdir
        if subdir_path.exists():
            for csv_file in subdir_path.glob('*_data.csv'):
                symbol = csv_file.name.replace('_data.csv', '')
                try:
                    df = pd.read_csv(csv_file, index_col='timestamp', parse_dates=True)
                    if len(df) > 0:
                        market_data[symbol] = df
                except Exception as e:
                    logger.warning(f"Failed to load {csv_file}: {e}")
    
    if not market_data:
        raise ValueError(f"No market data found in {market_data_path}")
    
    logger.info(f"Loaded market data for {len(market_data)} assets from {market_data_path}")
    return await processor.process_market_data(market_data)


# Layer 2 is now integrated into the main DCA system
# Use the main CLI for testing: python cli.py process
