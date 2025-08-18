#!/usr/bin/env python3
"""
Module 2.3: Data Transformation Pipeline

Data transformation, normalization, and feature engineering pipeline.
Cleans, normalizes, and prepares data from Layer 1 for Layer 3 intelligence systems.

Features:
- Data Normalization: Currency conversion, price adjustments, standardization
- Outlier Handling: Winsorization, anomaly detection, data cleaning
- Feature Engineering: Rolling statistics, momentum factors, regime indicators
- Data Alignment: Timestamp synchronization, missing data handling
- Production-optimized with efficient transformations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class OutlierMethod(Enum):
    """Methods for outlier detection and handling"""
    WINSORIZE = "winsorize"
    Z_SCORE = "z_score"
    IQR = "iqr"
    ISOLATION_FOREST = "isolation_forest"


class NormalizationMethod(Enum):
    """Methods for data normalization"""
    STANDARDIZE = "standardize"  # Z-score normalization
    MINMAX = "minmax"           # Min-max scaling
    ROBUST = "robust"           # Robust scaling (median and IQR)
    NONE = "none"               # No normalization


@dataclass
class PipelineConfig:
    """Configuration for data transformation pipeline"""
    
    # Data Quality
    max_missing_ratio: float = 0.1  # Maximum missing data ratio (10%)
    min_observations: int = 30  # Minimum observations per asset
    # SMART TIERED PROCESSING
    core_assets: int = 20      # Deep analysis - major cryptocurrencies
    satellite_assets: int = 50  # Moderate analysis - promising altcoins
    diversification_assets: int = 100  # Basic analysis - for diversification
    max_assets_to_process: int = 0  # Keep all data, but process intelligently
    
    # Outlier Handling
    outlier_method: OutlierMethod = OutlierMethod.WINSORIZE
    winsorize_limits: Tuple[float, float] = (0.01, 0.99)  # 1% and 99% percentiles
    z_score_threshold: float = 3.0
    
    # Normalization
    price_normalization: NormalizationMethod = NormalizationMethod.NONE
    returns_normalization: NormalizationMethod = NormalizationMethod.STANDARDIZE
    volume_normalization: NormalizationMethod = NormalizationMethod.ROBUST
    
    # Feature Engineering (Performance-optimized - reduced windows)
    momentum_windows: List[int] = field(default_factory=lambda: [10, 20])  # Reduced from 4 to 2
    volatility_windows: List[int] = field(default_factory=lambda: [20])     # Reduced from 3 to 1  
    volume_windows: List[int] = field(default_factory=lambda: [10])         # Reduced from 2 to 1
    
    # Data Alignment
    frequency: str = '1D'  # Target frequency for alignment
    interpolation_method: str = 'linear'  # Method for filling gaps
    forward_fill_limit: int = 3  # Max days to forward fill
    
    # Currency Conversion (for international data)
    base_currency: str = 'BRL'
    convert_currencies: bool = True


@dataclass
class TransformationResult:
    """Result of data transformation pipeline"""
    transformed_data: pd.DataFrame
    feature_metadata: Dict[str, Any]
    quality_report: Dict[str, Any]
    outliers_removed: Dict[str, int]
    transformations_applied: List[str]


class DataTransformationPipeline:
    """
    Production-ready data transformation pipeline
    
    Transforms raw market data into clean, normalized features ready for
    Layer 3 intelligence systems and portfolio optimization.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Track transformations
        self.transformations_applied = []
        self.feature_metadata = {}
        self.quality_report = {}
        
    def transform_data(self, raw_data: pd.DataFrame, 
                      metadata: Optional[Dict[str, Any]] = None) -> TransformationResult:
        """
        Apply complete data transformation pipeline
        
        Args:
            raw_data: Raw OHLCV data from Layer 1
            metadata: Optional metadata about the data
            
        Returns:
            TransformationResult with transformed data and metadata
        """
        self.logger.info(f"Starting data transformation pipeline for {len(raw_data.columns)} columns, {len(raw_data)} rows")
        
        # Reset transformation tracking
        self.transformations_applied = []
        self.feature_metadata = {}
        self.quality_report = {}
        
        # Start with copy of raw data
        data = raw_data.copy()
        
        # 1. Data Quality Assessment
        data = self._assess_data_quality(data)
        
        # 1.5. PERFORMANCE: Filter to top assets by volume
        data = self._filter_top_assets_by_volume(data)
        
        # 2. Data Alignment and Resampling
        data = self._align_and_resample_data(data)
        
        # 3. Missing Data Handling
        data = self._handle_missing_data(data)
        
        # 4. Outlier Detection and Treatment
        outliers_removed = self._handle_outliers(data)
        
        # 5. Currency Conversion (if needed)
        if self.config.convert_currencies:
            data = self._convert_currencies(data)
        
        # 6. Feature Engineering
        data = self._engineer_features(data)
        
        # 7. Data Normalization
        data = self._normalize_data(data)
        
        # 8. Final Quality Check
        final_quality = self._final_quality_check(data)
        
        self.logger.info(f"Transformation pipeline completed. Output: {len(data.columns)} features, {len(data)} rows")
        
        return TransformationResult(
            transformed_data=data,
            feature_metadata=self.feature_metadata,
            quality_report=self.quality_report,
            outliers_removed=outliers_removed,
            transformations_applied=self.transformations_applied
        )
    
    def _assess_data_quality(self, data: pd.DataFrame) -> pd.DataFrame:
        """Assess and report data quality issues"""
        self.logger.info("Assessing data quality...")
        
        quality_report = {}
        
        for column in data.columns:
            series = data[column]
            
            # Basic statistics
            total_obs = len(series)
            missing_obs = series.isna().sum()
            missing_ratio = missing_obs / total_obs
            
            # Data range
            if series.dtype in ['float64', 'int64']:
                data_min = series.min()
                data_max = series.max()
                data_std = series.std()
            else:
                data_min = data_max = data_std = np.nan
            
            quality_report[column] = {
                'total_observations': total_obs,
                'missing_observations': missing_obs,
                'missing_ratio': missing_ratio,
                'data_min': data_min,
                'data_max': data_max,
                'data_std': data_std,
                'dtype': str(series.dtype)
            }
            
            # Flag problematic columns
            if missing_ratio > self.config.max_missing_ratio:
                self.logger.warning(f"Column {column} has {missing_ratio:.1%} missing data (threshold: {self.config.max_missing_ratio:.1%})")
            
            if total_obs < self.config.min_observations:
                self.logger.warning(f"Column {column} has only {total_obs} observations (minimum: {self.config.min_observations})")
        
        self.quality_report['initial_assessment'] = quality_report
        self.transformations_applied.append('data_quality_assessment')
        
        return data
    
    def _filter_top_assets_by_volume(self, data: pd.DataFrame) -> pd.DataFrame:
        """Filter to top 150 cryptocurrencies by market cap"""
        from .top_crypto_assets import filter_assets_by_market_cap, get_asset_tier
        
        # Extract all unique asset symbols from column names
        all_assets = set()
        for col in data.columns:
            # Extract asset symbol from column names like 'BTC_close', 'ETH_volume', etc.
            if '_' in col:
                asset = col.split('_')[0]
                all_assets.add(asset)
        
        all_assets = list(all_assets)
        self.logger.info(f"🔍 Found {len(all_assets)} unique assets in data")
        
        # Filter to top 150 by market cap
        top_assets = filter_assets_by_market_cap(all_assets)
        self.logger.info(f"📈 Filtered to {len(top_assets)} assets from top 150 by market cap")
        
        if len(top_assets) == 0:
            self.logger.warning("No assets match top 150 crypto by market cap - keeping original data")
            return data
        
        # Filter columns to keep only top market cap assets
        columns_to_keep = []
        for col in data.columns:
            # Keep non-asset columns (like datetime index columns)
            if '_' not in col or col.split('_')[0] in top_assets:
                columns_to_keep.append(col)
        
        filtered_data = data[columns_to_keep]
        
        # Log asset tiers
        tier_counts = {}
        for asset in top_assets:
            tier = get_asset_tier(asset)
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        
        self.logger.info(f"🎯 MARKET CAP FILTERING: {len(all_assets)} → {len(top_assets)} assets")
        self.logger.info(f"📊 Asset breakdown: {dict(tier_counts)}")
        self.logger.info(f"🔢 Columns reduced: {len(data.columns)} → {len(filtered_data.columns)}")
        
        return filtered_data
    
    def _align_and_resample_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Align timestamps and resample to target frequency"""
        self.logger.info(f"Aligning data to {self.config.frequency} frequency...")
        
        if not isinstance(data.index, pd.DatetimeIndex):
            self.logger.warning("Data index is not DatetimeIndex, attempting to convert...")
            if 'timestamp' in data.columns:
                data.index = pd.to_datetime(data['timestamp'])
                data.drop('timestamp', axis=1, inplace=True)
            elif 'date' in data.columns:
                data.index = pd.to_datetime(data['date'])
                data.drop('date', axis=1, inplace=True)
            else:
                self.logger.error("Cannot find timestamp or date column for indexing")
                return data
        
        # Resample to target frequency
        if self.config.frequency != 'None':
            # Define aggregation methods for different column types
            agg_methods = {}
            for column in data.columns:
                if 'open' in column.lower():
                    agg_methods[column] = 'first'
                elif 'close' in column.lower():
                    agg_methods[column] = 'last'
                elif 'high' in column.lower():
                    agg_methods[column] = 'max'
                elif 'low' in column.lower():
                    agg_methods[column] = 'min'
                elif 'volume' in column.lower():
                    agg_methods[column] = 'sum'
                else:
                    agg_methods[column] = 'last'  # Default for other columns
            
            # Resample
            try:
                data = data.resample(self.config.frequency).agg(agg_methods)
            except Exception as e:
                self.logger.warning(f"Resampling failed: {e}, proceeding without resampling")
        
        self.transformations_applied.append('data_alignment')
        return data
    
    def _handle_missing_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing data through interpolation and forward filling"""
        self.logger.info("Handling missing data...")
        
        initial_missing = data.isna().sum().sum()
        
        # Forward fill first (limited)
        if self.config.forward_fill_limit > 0:
            data = data.fillna(method='ffill', limit=self.config.forward_fill_limit)
        
        # Interpolate numeric columns
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            if data[column].isna().any():
                data[column] = data[column].interpolate(method=self.config.interpolation_method)
        
        # Remove columns with too much missing data
        columns_to_remove = []
        for column in data.columns:
            missing_ratio = data[column].isna().sum() / len(data)
            if missing_ratio > self.config.max_missing_ratio:
                columns_to_remove.append(column)
        
        if columns_to_remove:
            self.logger.warning(f"Removing columns with excessive missing data: {columns_to_remove}")
            data = data.drop(columns=columns_to_remove)
        
        final_missing = data.isna().sum().sum()
        self.logger.info(f"Missing data reduced from {initial_missing} to {final_missing}")
        
        self.transformations_applied.append('missing_data_handling')
        return data
    
    def _handle_outliers(self, data: pd.DataFrame) -> Dict[str, int]:
        """Detect and handle outliers in the data"""
        self.logger.info(f"Handling outliers using {self.config.outlier_method.value} method...")
        
        outliers_removed = {}
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            series = data[column].dropna()
            original_count = len(series)
            
            if self.config.outlier_method == OutlierMethod.WINSORIZE:
                # Winsorization
                lower_limit = series.quantile(self.config.winsorize_limits[0])
                upper_limit = series.quantile(self.config.winsorize_limits[1])
                
                data[column] = data[column].clip(lower=lower_limit, upper=upper_limit)
                
            elif self.config.outlier_method == OutlierMethod.Z_SCORE:
                # Z-score based outlier removal
                z_scores = np.abs((series - series.mean()) / series.std())
                outlier_mask = z_scores > self.config.z_score_threshold
                data.loc[outlier_mask, column] = np.nan
                
            elif self.config.outlier_method == OutlierMethod.IQR:
                # IQR based outlier removal
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outlier_mask = (data[column] < lower_bound) | (data[column] > upper_bound)
                data.loc[outlier_mask, column] = np.nan
            
            # Count outliers handled
            final_count = data[column].count()
            outliers_removed[column] = original_count - final_count
        
        total_outliers = sum(outliers_removed.values())
        self.logger.info(f"Handled {total_outliers} outliers across all columns")
        
        self.transformations_applied.append('outlier_handling')
        return outliers_removed
    
    def _convert_currencies(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert currencies to base currency (placeholder for now)"""
        # This would integrate with a currency conversion service
        # For now, assume data is already in BRL
        self.logger.info(f"Currency conversion to {self.config.base_currency} (placeholder)")
        self.transformations_applied.append('currency_conversion')
        return data
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate portfolio theory fundamentals: returns, volatility, correlations"""
        self.logger.info("Calculating portfolio fundamentals (returns, volatility, correlations)...")

        # Identify close price columns for each asset
        price_columns = [col for col in data.columns if col.endswith('_close')]

        if not price_columns:
            self.logger.warning("No close price columns found - skipping portfolio calculations")
            return data

        self.logger.info(f"🚀 Processing {len(price_columns)} assets for portfolio theory - OPTIMIZED for large datasets")
        
        # 1. Calculate Daily Returns for each asset
        for price_col in price_columns:
            asset = price_col.replace('_close', '')
            data[f'{asset}_returns'] = data[price_col].pct_change()
            
            # 2. Calculate Rolling Volatility (risk measure)
            data[f'{asset}_volatility_20d'] = data[f'{asset}_returns'].rolling(20).std() * np.sqrt(252)  # Annualized
            
            # 3. Calculate Sharpe Ratio (risk-adjusted return)
            mean_return = data[f'{asset}_returns'].rolling(60).mean() * 252  # Annualized
            data[f'{asset}_sharpe_60d'] = mean_return / data[f'{asset}_volatility_20d']
        
                # 4. Create Returns Matrix for Portfolio Optimization (OPTIMIZED)
        returns_columns = [col for col in data.columns if col.endswith('_returns')]
        if len(returns_columns) > 1:
            self.logger.info(f"🔢 Calculating portfolio matrices for {len(returns_columns)} assets...")
            
            returns_matrix = data[returns_columns].dropna()

            if len(returns_matrix) > 30:  # Need sufficient data
                # 5. Calculate Correlation Matrix (OPTIMIZED - no individual correlations stored)
                self.logger.info("📊 Computing correlation matrix...")
                correlation_matrix = returns_matrix.corr()
                
                # Skip storing individual correlations to avoid 82k+ features
                # Only store the matrix itself for portfolio optimization

                # 6. Store covariance matrix for Markowitz optimization  
                self.logger.info("📈 Computing covariance matrix...")
                covariance_matrix = returns_matrix.cov() * 252  # Annualized

                self.feature_metadata['portfolio_matrices'] = {
                    'correlation_matrix': correlation_matrix,
                    'covariance_matrix': covariance_matrix,
                    'returns_matrix': returns_matrix,
                    'assets': [col.replace('_returns', '') for col in returns_columns]
                }

                # Calculate summary statistics instead of storing all pairs
                corr_values = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)]
                avg_correlation = corr_values.mean()
                max_correlation = corr_values.max()
                min_correlation = corr_values.min()
                
                # Store only summary correlation features
                data['portfolio_avg_correlation'] = avg_correlation
                data['portfolio_max_correlation'] = max_correlation  
                data['portfolio_min_correlation'] = min_correlation
                data['portfolio_correlation_std'] = corr_values.std()

                self.logger.info(f"✅ Portfolio matrices calculated for {len(returns_columns)} assets")
                self.logger.info(f"📊 Average correlation: {avg_correlation:.3f} (min: {min_correlation:.3f}, max: {max_correlation:.3f})")
            else:
                self.logger.warning(f"Insufficient data for portfolio calculations ({len(returns_matrix)} rows)")
        
        self.transformations_applied.append('portfolio_fundamentals')
        return data
    
    def _normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize data according to configuration"""
        self.logger.info("Normalizing data...")
        
        # Separate different types of features for different normalization
        price_columns = [col for col in data.columns if any(x in col.lower() for x in ['close', 'price']) 
                        and not any(x in col.lower() for x in ['ratio', 'momentum', 'volatility'])]
        volume_columns = [col for col in data.columns if 'volume' in col.lower() and 'ratio' not in col.lower()]
        return_columns = [col for col in data.columns if 'momentum' in col.lower() or 'return' in col.lower()]
        ratio_columns = [col for col in data.columns if 'ratio' in col.lower()]
        
        # Apply normalization based on column type
        normalization_applied = {}
        
        # Price normalization
        if self.config.price_normalization != NormalizationMethod.NONE:
            for column in price_columns:
                data[column] = self._apply_normalization(data[column], self.config.price_normalization)
            normalization_applied['price'] = self.config.price_normalization.value
        
        # Volume normalization
        if self.config.volume_normalization != NormalizationMethod.NONE:
            for column in volume_columns:
                data[column] = self._apply_normalization(data[column], self.config.volume_normalization)
            normalization_applied['volume'] = self.config.volume_normalization.value
        
        # Returns normalization
        if self.config.returns_normalization != NormalizationMethod.NONE:
            for column in return_columns:
                data[column] = self._apply_normalization(data[column], self.config.returns_normalization)
            normalization_applied['returns'] = self.config.returns_normalization.value
        
        self.feature_metadata['normalization_applied'] = normalization_applied
        self.transformations_applied.append('data_normalization')
        return data
    
    # Helper methods
    
    def _identify_asset_columns(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """Identify columns belonging to different assets"""
        assets = {}
        
        # Simple heuristic: group columns by common prefixes
        for column in data.columns:
            parts = column.split('_')
            if len(parts) > 1:
                asset_name = parts[0]
                if asset_name not in assets:
                    assets[asset_name] = []
                assets[asset_name].append(column)
        
        return assets
    
    def _create_cross_asset_features(self, data: pd.DataFrame, asset_columns: Dict[str, List[str]]) -> pd.DataFrame:
        """Create features that compare different assets"""
        asset_names = list(asset_columns.keys())
        
        # Price correlations
        price_columns = {}
        for asset in asset_names:
            asset_price_cols = [col for col in asset_columns[asset] if 'close' in col.lower() or 'price' in col.lower()]
            if asset_price_cols:
                price_columns[asset] = asset_price_cols[0]
        
        # Rolling correlations between assets
        if len(price_columns) >= 2:
            asset_pairs = [(a1, a2) for i, a1 in enumerate(asset_names) for a2 in asset_names[i+1:]]
            
            for asset1, asset2 in asset_pairs:
                if asset1 in price_columns and asset2 in price_columns:
                    returns1 = data[price_columns[asset1]].pct_change()
                    returns2 = data[price_columns[asset2]].pct_change()
                    
                    # Rolling correlation
                    correlation = returns1.rolling(window=30).corr(returns2)
                    data[f'{asset1}_{asset2}_correlation_30d'] = correlation
        
        return data
    
    def _create_regime_indicators(self, data: pd.DataFrame, price_columns: List[str]) -> pd.DataFrame:
        """Create market regime indicators"""
        for column in price_columns:
            returns = data[column].pct_change()
            
            # Volatility regime (high/low based on rolling percentiles)
            vol_30d = returns.rolling(window=30).std()
            vol_percentile = vol_30d.rolling(window=252).rank(pct=True)  # 1-year ranking
            data[f'{column}_vol_regime'] = np.where(vol_percentile > 0.8, 2,  # High vol
                                                   np.where(vol_percentile < 0.2, 0, 1))  # Low, Normal vol
            
            # Trend regime (up/down/sideways)
            ma_short = data[column].rolling(window=20).mean()
            ma_long = data[column].rolling(window=50).mean()
            
            trend_signal = np.where(ma_short > ma_long * 1.02, 1,     # Uptrend
                                  np.where(ma_short < ma_long * 0.98, -1, 0))  # Downtrend, Sideways
            data[f'{column}_trend_regime'] = trend_signal
        
        return data
    
    def _apply_normalization(self, series: pd.Series, method: NormalizationMethod) -> pd.Series:
        """Apply specified normalization method to a series"""
        if method == NormalizationMethod.STANDARDIZE:
            return (series - series.mean()) / series.std()
        elif method == NormalizationMethod.MINMAX:
            return (series - series.min()) / (series.max() - series.min())
        elif method == NormalizationMethod.ROBUST:
            median = series.median()
            mad = (series - median).abs().median()  # Median Absolute Deviation
            return (series - median) / (1.4826 * mad)  # 1.4826 factor for consistency with std
        else:
            return series
    
    def _final_quality_check(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform final quality check on transformed data"""
        self.logger.info("Performing final quality check...")
        
        quality_check = {
            'final_shape': data.shape,
            'final_missing_data': data.isna().sum().sum(),
            'infinite_values': np.isinf(data.select_dtypes(include=[np.number])).sum().sum(),
            'column_dtypes': data.dtypes.to_dict(),
            'date_range': {
                'start': data.index.min() if isinstance(data.index, pd.DatetimeIndex) else None,
                'end': data.index.max() if isinstance(data.index, pd.DatetimeIndex) else None
            }
        }
        
        # Check for problematic values
        numeric_data = data.select_dtypes(include=[np.number])
        quality_check['suspicious_values'] = {
            'infinite_count': np.isinf(numeric_data).sum().sum(),
            'nan_count': numeric_data.isna().sum().sum(),
            'extreme_values': (np.abs(numeric_data) > 1000).sum().sum()  # Arbitrary threshold
        }
        
        self.quality_report['final_quality'] = quality_check
        return quality_check


# Data Transformation Pipeline is integrated into the main DCA system
# Use the main CLI for testing: python cli.py run
