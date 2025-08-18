"""
Module 1.3: Data Quality Manager

Comprehensive data quality assurance system for the DCA platform.
Ensures data integrity, completeness, and accuracy across all data sources.

Quality Checks:
- Schema validation
- Missing data detection
- Outlier detection and handling
- Cross-source validation
- Time series continuity
- Data freshness monitoring
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import warnings
from scipy import stats
import asyncio

from .market_data_collector import MarketDataPoint, AssetMetadata

logger = logging.getLogger(__name__)


class DataQualityLevel(Enum):
    """Data quality assessment levels"""
    EXCELLENT = "excellent"
    GOOD = "good" 
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNACCEPTABLE = "unacceptable"


@dataclass
class QualityCheck:
    """Individual quality check result"""
    check_name: str
    passed: bool
    score: float  # 0.0 to 1.0
    message: str
    severity: str  # 'critical', 'warning', 'info'
    details: Dict[str, Any]


@dataclass
class DataQualityReport:
    """Comprehensive data quality assessment"""
    symbol: str
    overall_score: float
    quality_level: DataQualityLevel
    checks: List[QualityCheck]
    timestamp: datetime
    recommendations: List[str]
    is_usable: bool


class DataQualityManager:
    """
    Main data quality orchestrator
    Performs comprehensive quality checks on market data and metadata
    """
    
    def __init__(self, min_acceptable_score: float = 0.7):
        self.min_acceptable_score = min_acceptable_score
        
        # Quality thresholds
        self.quality_thresholds = {
            'completeness_min': 0.95,      # 95% data completeness required
            'outlier_max_pct': 0.05,       # Max 5% outliers allowed
            'price_jump_threshold': 0.50,   # 50% single-period jump threshold
            'volume_spike_threshold': 10.0, # 10x volume spike threshold  
            'staleness_max_hours': 24,     # Max 24 hours data staleness
            'min_data_points': 30,         # Minimum data points for analysis
        }
        
        self.validators = [
            self._validate_schema,
            self._validate_completeness,
            self._validate_price_continuity,
            self._validate_volume_patterns,
            self._validate_outliers,
            self._validate_freshness,
            self._validate_consistency
        ]
        
        logger.info("Data Quality Manager initialized")
    
    async def assess_ohlcv_quality(self, ohlcv_data: Dict[str, List[MarketDataPoint]]) -> Dict[str, DataQualityReport]:
        """
        Comprehensive quality assessment for OHLCV data
        
        Args:
            ohlcv_data: Dictionary of symbol -> MarketDataPoint list
            
        Returns:
            Dictionary of symbol -> DataQualityReport
        """
        logger.info(f"Starting quality assessment for {len(ohlcv_data)} symbols")
        
        quality_reports = {}
        
        # Process each symbol
        for symbol, data_points in ohlcv_data.items():
            if not data_points:
                quality_reports[symbol] = self._create_empty_report(symbol, "No data points available")
                continue
            
            # Convert to DataFrame for analysis
            df = self._datapoints_to_dataframe(data_points)
            
            if df.empty:
                quality_reports[symbol] = self._create_empty_report(symbol, "Empty DataFrame after conversion")
                continue
            
            # Run all quality checks
            checks = []
            for validator in self.validators:
                try:
                    check_result = await validator(symbol, df, data_points)
                    checks.append(check_result)
                except Exception as e:
                    logger.error(f"Quality check failed for {symbol}: {e}")
                    checks.append(QualityCheck(
                        check_name=validator.__name__,
                        passed=False,
                        score=0.0,
                        message=f"Check failed: {e}",
                        severity='critical',
                        details={}
                    ))
            
            # Calculate overall quality
            quality_reports[symbol] = self._compile_quality_report(symbol, checks)
        
        # Summary logging
        total_symbols = len(quality_reports)
        usable_symbols = sum(1 for report in quality_reports.values() if report.is_usable)
        avg_score = np.mean([report.overall_score for report in quality_reports.values()])
        
        logger.info(f"Quality assessment completed:")
        logger.info(f"  • Total symbols assessed: {total_symbols}")
        logger.info(f"  • Usable symbols: {usable_symbols} ({usable_symbols/total_symbols*100:.1f}%)")
        logger.info(f"  • Average quality score: {avg_score:.3f}")
        
        return quality_reports
    
    async def assess_metadata_quality(self, metadata: Dict[str, AssetMetadata]) -> Dict[str, DataQualityReport]:
        """Quality assessment for asset metadata"""
        logger.info(f"Assessing metadata quality for {len(metadata)} assets")
        
        quality_reports = {}
        
        for symbol, meta in metadata.items():
            checks = []
            
            # Check metadata completeness
            checks.append(await self._validate_metadata_completeness(symbol, meta))
            
            # Check metadata consistency
            checks.append(await self._validate_metadata_consistency(symbol, meta))
            
            # Check metadata freshness
            checks.append(await self._validate_metadata_freshness(symbol, meta))
            
            # Compile report
            quality_reports[symbol] = self._compile_quality_report(symbol, checks)
        
        return quality_reports
    
    def _datapoints_to_dataframe(self, data_points: List[MarketDataPoint]) -> pd.DataFrame:
        """Convert MarketDataPoint list to DataFrame for analysis"""
        try:
            df = pd.DataFrame([
                {
                    'timestamp': point.timestamp,
                    'open': point.open_price,
                    'high': point.high_price,
                    'low': point.low_price,
                    'close': point.close_price,
                    'volume': point.volume,
                    'quote_volume': point.quote_volume
                }
                for point in data_points
            ])
            
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error converting data points to DataFrame: {e}")
            return pd.DataFrame()
    
    async def _validate_schema(self, symbol: str, df: pd.DataFrame, data_points: List[MarketDataPoint]) -> QualityCheck:
        """Validate data schema and structure"""
        try:
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                return QualityCheck(
                    check_name="schema_validation",
                    passed=False,
                    score=0.0,
                    message=f"Missing required columns: {missing_columns}",
                    severity='critical',
                    details={'missing_columns': missing_columns}
                )
            
            # Check data types
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
            type_issues = []
            
            for col in numeric_columns:
                if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                    type_issues.append(col)
            
            if type_issues:
                return QualityCheck(
                    check_name="schema_validation",
                    passed=False,
                    score=0.3,
                    message=f"Non-numeric data in columns: {type_issues}",
                    severity='warning',
                    details={'type_issues': type_issues}
                )
            
            return QualityCheck(
                check_name="schema_validation",
                passed=True,
                score=1.0,
                message="Schema validation passed",
                severity='info',
                details={'columns': list(df.columns)}
            )
            
        except Exception as e:
            return QualityCheck(
                check_name="schema_validation",
                passed=False,
                score=0.0,
                message=f"Schema validation error: {e}",
                severity='critical',
                details={}
            )
    
    async def _validate_completeness(self, symbol: str, df: pd.DataFrame, data_points: List[MarketDataPoint]) -> QualityCheck:
        """Validate data completeness (missing values)"""
        try:
            if df.empty:
                return QualityCheck(
                    check_name="completeness_check",
                    passed=False,
                    score=0.0,
                    message="Empty dataset",
                    severity='critical',
                    details={}
                )
            
            total_cells = df.size
            missing_cells = df.isnull().sum().sum()
            completeness_ratio = 1 - (missing_cells / total_cells)
            
            # Check minimum data points
            min_points = self.quality_thresholds['min_data_points']
            if len(df) < min_points:
                return QualityCheck(
                    check_name="completeness_check",
                    passed=False,
                    score=0.2,
                    message=f"Insufficient data points: {len(df)} < {min_points}",
                    severity='warning',
                    details={'data_points': len(df), 'minimum_required': min_points}
                )
            
            # Check completeness threshold
            completeness_threshold = self.quality_thresholds['completeness_min']
            
            passed = completeness_ratio >= completeness_threshold
            score = min(1.0, completeness_ratio / completeness_threshold)
            
            return QualityCheck(
                check_name="completeness_check",
                passed=passed,
                score=score,
                message=f"Data completeness: {completeness_ratio:.3f} ({100*completeness_ratio:.1f}%)",
                severity='warning' if not passed else 'info',
                details={
                    'completeness_ratio': completeness_ratio,
                    'missing_cells': missing_cells,
                    'total_cells': total_cells,
                    'data_points': len(df)
                }
            )
            
        except Exception as e:
            return QualityCheck(
                check_name="completeness_check",
                passed=False,
                score=0.0,
                message=f"Completeness check error: {e}",
                severity='critical',
                details={}
            )
    
    async def _validate_price_continuity(self, symbol: str, df: pd.DataFrame, data_points: List[MarketDataPoint]) -> QualityCheck:
        """Validate price continuity and detect unreasonable price jumps"""
        try:
            if len(df) < 2:
                return QualityCheck(
                    check_name="price_continuity",
                    passed=True,
                    score=1.0,
                    message="Insufficient data for continuity check",
                    severity='info',
                    details={}
                )
            
            # Check OHLC consistency within each period
            ohlc_issues = 0
            for idx, row in df.iterrows():
                if not (row['low'] <= row['open'] <= row['high'] and
                       row['low'] <= row['close'] <= row['high']):
                    ohlc_issues += 1
            
            ohlc_consistency = 1 - (ohlc_issues / len(df))
            
            # Check price jumps between periods
            price_returns = df['close'].pct_change().dropna()
            jump_threshold = self.quality_thresholds['price_jump_threshold']
            
            extreme_jumps = (np.abs(price_returns) > jump_threshold).sum()
            jump_rate = extreme_jumps / len(price_returns) if len(price_returns) > 0 else 0
            
            # Check for negative prices (critical error)
            negative_prices = (df[['open', 'high', 'low', 'close']] <= 0).any(axis=1).sum()
            
            if negative_prices > 0:
                return QualityCheck(
                    check_name="price_continuity",
                    passed=False,
                    score=0.0,
                    message=f"Found {negative_prices} periods with non-positive prices",
                    severity='critical',
                    details={'negative_prices': negative_prices}
                )
            
            # Overall score
            continuity_score = (ohlc_consistency * 0.7) + ((1 - jump_rate) * 0.3)
            passed = continuity_score >= 0.8 and ohlc_issues == 0
            
            return QualityCheck(
                check_name="price_continuity",
                passed=passed,
                score=continuity_score,
                message=f"Price continuity: {continuity_score:.3f} (OHLC issues: {ohlc_issues}, jumps: {extreme_jumps})",
                severity='warning' if not passed else 'info',
                details={
                    'ohlc_issues': ohlc_issues,
                    'extreme_jumps': extreme_jumps,
                    'jump_rate': jump_rate,
                    'max_jump': np.abs(price_returns).max() if len(price_returns) > 0 else 0
                }
            )
            
        except Exception as e:
            return QualityCheck(
                check_name="price_continuity",
                passed=False,
                score=0.0,
                message=f"Price continuity check error: {e}",
                severity='critical',
                details={}
            )
    
    async def _validate_volume_patterns(self, symbol: str, df: pd.DataFrame, data_points: List[MarketDataPoint]) -> QualityCheck:
        """Validate volume patterns and detect anomalies"""
        try:
            if 'volume' not in df.columns or df['volume'].isnull().all():
                return QualityCheck(
                    check_name="volume_patterns",
                    passed=False,
                    score=0.3,
                    message="No volume data available",
                    severity='warning',
                    details={}
                )
            
            volumes = df['volume'].dropna()
            
            if len(volumes) == 0:
                return QualityCheck(
                    check_name="volume_patterns",
                    passed=False,
                    score=0.0,
                    message="All volume values are null",
                    severity='critical',
                    details={}
                )
            
            # Check for negative volumes
            negative_volumes = (volumes < 0).sum()
            if negative_volumes > 0:
                return QualityCheck(
                    check_name="volume_patterns",
                    passed=False,
                    score=0.0,
                    message=f"Found {negative_volumes} negative volume values",
                    severity='critical',
                    details={'negative_volumes': negative_volumes}
                )
            
            # Check for zero volumes (acceptable but noteworthy)
            zero_volumes = (volumes == 0).sum()
            zero_volume_rate = zero_volumes / len(volumes)
            
            # Check for volume spikes
            volume_median = volumes.median()
            spike_threshold = self.quality_thresholds['volume_spike_threshold']
            
            volume_spikes = (volumes > volume_median * spike_threshold).sum()
            spike_rate = volume_spikes / len(volumes)
            
            # Overall volume quality score
            volume_score = 1.0
            if zero_volume_rate > 0.2:  # More than 20% zero volumes
                volume_score *= 0.7
            if spike_rate > 0.1:  # More than 10% volume spikes
                volume_score *= 0.8
            
            passed = volume_score >= 0.6 and negative_volumes == 0
            
            return QualityCheck(
                check_name="volume_patterns",
                passed=passed,
                score=volume_score,
                message=f"Volume quality: {volume_score:.3f} (zeros: {zero_volume_rate:.2%}, spikes: {spike_rate:.2%})",
                severity='warning' if not passed else 'info',
                details={
                    'zero_volumes': zero_volumes,
                    'zero_volume_rate': zero_volume_rate,
                    'volume_spikes': volume_spikes,
                    'spike_rate': spike_rate,
                    'median_volume': volume_median
                }
            )
            
        except Exception as e:
            return QualityCheck(
                check_name="volume_patterns",
                passed=False,
                score=0.0,
                message=f"Volume pattern check error: {e}",
                severity='critical',
                details={}
            )
    
    async def _validate_outliers(self, symbol: str, df: pd.DataFrame, data_points: List[MarketDataPoint]) -> QualityCheck:
        """Detect and assess outliers using statistical methods"""
        try:
            if len(df) < 10:
                return QualityCheck(
                    check_name="outlier_detection",
                    passed=True,
                    score=1.0,
                    message="Insufficient data for outlier detection",
                    severity='info',
                    details={}
                )
            
            # Calculate returns for outlier detection
            returns = df['close'].pct_change().dropna()
            
            if len(returns) == 0:
                return QualityCheck(
                    check_name="outlier_detection",
                    passed=True,
                    score=1.0,
                    message="No returns available for outlier detection",
                    severity='info',
                    details={}
                )
            
            # Z-score based outlier detection
            z_scores = np.abs(stats.zscore(returns))
            z_outliers = (z_scores > 3).sum()  # More than 3 standard deviations
            
            # IQR based outlier detection
            q1 = returns.quantile(0.25)
            q3 = returns.quantile(0.75)
            iqr = q3 - q1
            iqr_outliers = ((returns < (q1 - 1.5 * iqr)) | (returns > (q3 + 1.5 * iqr))).sum()
            
            # Overall outlier rate
            outlier_count = max(z_outliers, iqr_outliers)
            outlier_rate = outlier_count / len(returns)
            
            max_outlier_rate = self.quality_thresholds['outlier_max_pct']
            passed = outlier_rate <= max_outlier_rate
            
            # Score based on outlier rate
            if outlier_rate == 0:
                score = 1.0
            else:
                score = max(0.0, 1.0 - (outlier_rate / max_outlier_rate))
            
            return QualityCheck(
                check_name="outlier_detection",
                passed=passed,
                score=score,
                message=f"Outlier rate: {outlier_rate:.3f} ({outlier_count}/{len(returns)})",
                severity='warning' if not passed else 'info',
                details={
                    'outlier_count': outlier_count,
                    'outlier_rate': outlier_rate,
                    'z_outliers': z_outliers,
                    'iqr_outliers': iqr_outliers,
                    'return_std': returns.std(),
                    'return_mean': returns.mean()
                }
            )
            
        except Exception as e:
            return QualityCheck(
                check_name="outlier_detection",
                passed=False,
                score=0.0,
                message=f"Outlier detection error: {e}",
                severity='critical',
                details={}
            )
    
    async def _validate_freshness(self, symbol: str, df: pd.DataFrame, data_points: List[MarketDataPoint]) -> QualityCheck:
        """Validate data freshness (how recent the data is)"""
        try:
            if df.empty:
                return QualityCheck(
                    check_name="data_freshness",
                    passed=False,
                    score=0.0,
                    message="No data available for freshness check",
                    severity='critical',
                    details={}
                )
            
            # Get latest data point
            latest_timestamp = df.index.max()
            current_time = datetime.now()
            
            # Handle timezone-naive timestamps
            if latest_timestamp.tzinfo is None:
                latest_timestamp = latest_timestamp.replace(tzinfo=current_time.tzinfo)
            
            staleness_hours = (current_time - latest_timestamp).total_seconds() / 3600
            max_staleness = self.quality_thresholds['staleness_max_hours']
            
            passed = staleness_hours <= max_staleness
            
            # Score based on freshness
            if staleness_hours <= max_staleness / 2:
                score = 1.0  # Very fresh
            elif staleness_hours <= max_staleness:
                score = 0.8  # Acceptable
            else:
                score = max(0.0, 1.0 - (staleness_hours / (max_staleness * 2)))
            
            return QualityCheck(
                check_name="data_freshness",
                passed=passed,
                score=score,
                message=f"Data age: {staleness_hours:.1f} hours (latest: {latest_timestamp})",
                severity='warning' if not passed else 'info',
                details={
                    'latest_timestamp': latest_timestamp,
                    'staleness_hours': staleness_hours,
                    'max_allowed_hours': max_staleness
                }
            )
            
        except Exception as e:
            return QualityCheck(
                check_name="data_freshness",
                passed=False,
                score=0.0,
                message=f"Freshness check error: {e}",
                severity='critical',
                details={}
            )
    
    async def _validate_consistency(self, symbol: str, df: pd.DataFrame, data_points: List[MarketDataPoint]) -> QualityCheck:
        """Validate internal data consistency"""
        try:
            consistency_issues = []
            
            # Check timestamp ordering
            if not df.index.is_monotonic_increasing:
                consistency_issues.append("Timestamps not in chronological order")
            
            # Check for duplicate timestamps
            duplicate_timestamps = df.index.duplicated().sum()
            if duplicate_timestamps > 0:
                consistency_issues.append(f"Found {duplicate_timestamps} duplicate timestamps")
            
            # Check for consistent data intervals (if enough data)
            if len(df) > 5:
                intervals = df.index.to_series().diff().dropna()
                if len(intervals.unique()) > 3:  # Allow some variation in intervals
                    consistency_issues.append("Inconsistent time intervals between data points")
            
            # Overall consistency score
            if not consistency_issues:
                score = 1.0
                passed = True
                message = "Data consistency validation passed"
                severity = 'info'
            else:
                score = max(0.0, 1.0 - (len(consistency_issues) * 0.3))
                passed = score >= 0.6
                message = f"Consistency issues found: {', '.join(consistency_issues)}"
                severity = 'warning'
            
            return QualityCheck(
                check_name="consistency_check",
                passed=passed,
                score=score,
                message=message,
                severity=severity,
                details={
                    'issues': consistency_issues,
                    'duplicate_timestamps': duplicate_timestamps,
                    'monotonic_timestamps': df.index.is_monotonic_increasing
                }
            )
            
        except Exception as e:
            return QualityCheck(
                check_name="consistency_check",
                passed=False,
                score=0.0,
                message=f"Consistency check error: {e}",
                severity='critical',
                details={}
            )
    
    async def _validate_metadata_completeness(self, symbol: str, metadata: AssetMetadata) -> QualityCheck:
        """Validate metadata completeness"""
        try:
            required_fields = ['symbol', 'name', 'is_active', 'exchange', 'source']
            optional_fields = ['market_cap_usd', 'daily_volume_brl', 'listing_date', 'sector']
            
            missing_required = []
            for field in required_fields:
                value = getattr(metadata, field, None)
                if value is None or value == '':
                    missing_required.append(field)
            
            present_optional = []
            for field in optional_fields:
                value = getattr(metadata, field, None)
                if value is not None:
                    present_optional.append(field)
            
            if missing_required:
                return QualityCheck(
                    check_name="metadata_completeness",
                    passed=False,
                    score=0.0,
                    message=f"Missing required fields: {missing_required}",
                    severity='critical',
                    details={'missing_required': missing_required}
                )
            
            # Score based on optional field completeness
            completeness_score = len(present_optional) / len(optional_fields)
            
            return QualityCheck(
                check_name="metadata_completeness",
                passed=True,
                score=0.7 + (0.3 * completeness_score),  # Base 70% + bonus for optional fields
                message=f"Metadata completeness: {completeness_score:.2%} optional fields",
                severity='info',
                details={
                    'present_optional': present_optional,
                    'missing_optional': [f for f in optional_fields if f not in present_optional]
                }
            )
            
        except Exception as e:
            return QualityCheck(
                check_name="metadata_completeness",
                passed=False,
                score=0.0,
                message=f"Metadata completeness check error: {e}",
                severity='critical',
                details={}
            )
    
    async def _validate_metadata_consistency(self, symbol: str, metadata: AssetMetadata) -> QualityCheck:
        """Validate metadata consistency and reasonableness"""
        try:
            issues = []
            
            # Check symbol consistency
            if metadata.symbol != symbol:
                issues.append(f"Symbol mismatch: expected {symbol}, got {metadata.symbol}")
            
            # Check market cap reasonableness (if present)
            if metadata.market_cap_usd is not None:
                if metadata.market_cap_usd < 0:
                    issues.append("Negative market cap")
                elif metadata.market_cap_usd > 1e12:  # $1T seems unreasonable for small caps
                    issues.append("Unreasonably large market cap")
            
            # Check volume reasonableness (if present)
            if metadata.daily_volume_brl is not None:
                if metadata.daily_volume_brl < 0:
                    issues.append("Negative daily volume")
            
            # Check listing date reasonableness (if present)
            if metadata.listing_date is not None:
                if metadata.listing_date > datetime.now():
                    issues.append("Future listing date")
                elif metadata.listing_date < datetime(2009, 1, 1):  # Before Bitcoin
                    issues.append("Unreasonably early listing date")
            
            if not issues:
                return QualityCheck(
                    check_name="metadata_consistency",
                    passed=True,
                    score=1.0,
                    message="Metadata consistency validation passed",
                    severity='info',
                    details={}
                )
            else:
                score = max(0.0, 1.0 - (len(issues) * 0.25))
                return QualityCheck(
                    check_name="metadata_consistency",
                    passed=score >= 0.6,
                    score=score,
                    message=f"Consistency issues: {', '.join(issues)}",
                    severity='warning',
                    details={'issues': issues}
                )
                
        except Exception as e:
            return QualityCheck(
                check_name="metadata_consistency",
                passed=False,
                score=0.0,
                message=f"Metadata consistency check error: {e}",
                severity='critical',
                details={}
            )
    
    async def _validate_metadata_freshness(self, symbol: str, metadata: AssetMetadata) -> QualityCheck:
        """Validate metadata freshness (not implemented for basic metadata)"""
        return QualityCheck(
            check_name="metadata_freshness",
            passed=True,
            score=1.0,
            message="Metadata freshness check not applicable",
            severity='info',
            details={}
        )
    
    def _create_empty_report(self, symbol: str, reason: str) -> DataQualityReport:
        """Create a quality report for empty/invalid data"""
        return DataQualityReport(
            symbol=symbol,
            overall_score=0.0,
            quality_level=DataQualityLevel.UNACCEPTABLE,
            checks=[QualityCheck(
                check_name="data_availability",
                passed=False,
                score=0.0,
                message=reason,
                severity='critical',
                details={}
            )],
            timestamp=datetime.now(),
            recommendations=[f"No usable data available for {symbol}: {reason}"],
            is_usable=False
        )
    
    def _compile_quality_report(self, symbol: str, checks: List[QualityCheck]) -> DataQualityReport:
        """Compile individual checks into comprehensive quality report"""
        try:
            if not checks:
                return self._create_empty_report(symbol, "No quality checks performed")
            
            # Calculate weighted overall score
            critical_checks = [c for c in checks if c.severity == 'critical']
            warning_checks = [c for c in checks if c.severity == 'warning']
            info_checks = [c for c in checks if c.severity == 'info']
            
            # Critical failures result in very low score
            if any(not check.passed for check in critical_checks):
                overall_score = min(0.3, np.mean([c.score for c in checks]))
            else:
                # Weight different severity levels
                scores = []
                weights = []
                
                for check in checks:
                    if check.severity == 'critical':
                        weights.append(0.4)  # Critical checks get 40% weight
                    elif check.severity == 'warning':
                        weights.append(0.35)  # Warning checks get 35% weight
                    else:
                        weights.append(0.25)  # Info checks get 25% weight
                    scores.append(check.score)
                
                overall_score = np.average(scores, weights=weights)
            
            # Determine quality level
            if overall_score >= 0.9:
                quality_level = DataQualityLevel.EXCELLENT
            elif overall_score >= 0.8:
                quality_level = DataQualityLevel.GOOD
            elif overall_score >= self.min_acceptable_score:
                quality_level = DataQualityLevel.ACCEPTABLE
            elif overall_score >= 0.4:
                quality_level = DataQualityLevel.POOR
            else:
                quality_level = DataQualityLevel.UNACCEPTABLE
            
            # Generate recommendations
            recommendations = []
            failed_critical = [c for c in critical_checks if not c.passed]
            failed_warning = [c for c in warning_checks if not c.passed]
            
            if failed_critical:
                recommendations.append(f"CRITICAL: Fix {len(failed_critical)} critical data issues before use")
            
            if failed_warning:
                recommendations.append(f"WARNING: Address {len(failed_warning)} data quality warnings")
            
            if overall_score < self.min_acceptable_score:
                recommendations.append("Data quality below minimum threshold - consider excluding from analysis")
            
            if not recommendations:
                recommendations.append("Data quality is acceptable for analysis")
            
            # Determine if data is usable
            is_usable = (overall_score >= self.min_acceptable_score and 
                        not any(not check.passed for check in critical_checks))
            
            return DataQualityReport(
                symbol=symbol,
                overall_score=overall_score,
                quality_level=quality_level,
                checks=checks,
                timestamp=datetime.now(),
                recommendations=recommendations,
                is_usable=is_usable
            )
            
        except Exception as e:
            logger.error(f"Error compiling quality report for {symbol}: {e}")
            return self._create_empty_report(symbol, f"Error compiling quality report: {e}")
    
    def filter_by_quality(self, data: Dict[str, any], quality_reports: Dict[str, DataQualityReport], 
                         min_score: Optional[float] = None) -> Dict[str, any]:
        """Filter data based on quality scores"""
        min_score = min_score or self.min_acceptable_score
        
        filtered_data = {}
        for symbol, report in quality_reports.items():
            if report.overall_score >= min_score and report.is_usable and symbol in data:
                filtered_data[symbol] = data[symbol]
        
        logger.info(f"Quality filtering: {len(filtered_data)}/{len(data)} symbols passed (min_score={min_score})")
        return filtered_data
    
    def get_quality_summary(self, quality_reports: Dict[str, DataQualityReport]) -> Dict[str, Any]:
        """Generate summary statistics for quality reports"""
        if not quality_reports:
            return {}
        
        scores = [r.overall_score for r in quality_reports.values()]
        usable_count = sum(1 for r in quality_reports.values() if r.is_usable)
        
        quality_levels = {}
        for level in DataQualityLevel:
            quality_levels[level.value] = sum(1 for r in quality_reports.values() 
                                            if r.quality_level == level)
        
        return {
            'total_symbols': len(quality_reports),
            'usable_symbols': usable_count,
            'usable_rate': usable_count / len(quality_reports),
            'average_score': np.mean(scores),
            'median_score': np.median(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'quality_levels': quality_levels,
            'critical_failures': sum(1 for r in quality_reports.values() 
                                   if any(c.severity == 'critical' and not c.passed for c in r.checks))
        }

