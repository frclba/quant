#!/usr/bin/env python3
"""
Production Configuration Management for DCA Data Ingestion Layer 1

Centralizes all configuration, supports environment variables, and provides
validation for production deployment.
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import logging


@dataclass
class APIConfig:
    """API configuration settings"""
    rate_limit_seconds: float = 1.1  # Seconds between requests
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_backoff_factor: float = 2.0
    max_concurrent_requests: int = 5


@dataclass
class StorageConfig:
    """Data storage configuration"""
    base_path: str = ""  # Will be set by DataConfig
    major_crypto_subdir: str = "major_crypto_data"
    altcoin_subdir: str = "altcoin_data"
    logs_subdir: str = "logs"
    backup_enabled: bool = True
    compression_enabled: bool = True
    retention_days: int = 730


@dataclass  
class DataConfig:
    """Data collection configuration"""
    symbol_universe_file: str = "mb_symbols_20250817_184346.json"
    default_history_days: int = 365
    incremental_buffer_days: int = 2  # Extra days to avoid gaps
    min_data_points_threshold: int = 10
    data_quality_min_score: float = 0.7


@dataclass
class MonitoringConfig:
    """Monitoring and alerting configuration"""
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    progress_report_interval: int = 50  # Report progress every N symbols
    metrics_enabled: bool = True
    alerts_enabled: bool = False  # For future implementation


@dataclass
class ProductionConfig:
    """Main production configuration container"""
    api: APIConfig
    storage: StorageConfig
    data: DataConfig
    monitoring: MonitoringConfig
    
    def __post_init__(self):
        """Set up derived paths after initialization"""
        if not self.storage.base_path:
            # Default to data directory relative to this file
            self.storage.base_path = str(
                Path(__file__).parent.parent.parent / "data"
            )
    
    @classmethod
    def from_env(cls) -> 'ProductionConfig':
        """Create configuration from environment variables"""
        
        # API Configuration
        api_config = APIConfig(
            rate_limit_seconds=float(os.getenv('DCA_API_RATE_LIMIT', '1.1')),
            timeout_seconds=int(os.getenv('DCA_API_TIMEOUT', '30')),
            max_retries=int(os.getenv('DCA_API_MAX_RETRIES', '3')),
            max_concurrent_requests=int(os.getenv('DCA_API_MAX_CONCURRENT', '5'))
        )
        
        # Storage Configuration  
        storage_config = StorageConfig(
            base_path=os.getenv('DCA_DATA_PATH', ''),
            retention_days=int(os.getenv('DCA_DATA_RETENTION_DAYS', '730')),
            backup_enabled=os.getenv('DCA_BACKUP_ENABLED', 'true').lower() == 'true'
        )
        
        # Data Configuration
        data_config = DataConfig(
            default_history_days=int(os.getenv('DCA_DEFAULT_HISTORY_DAYS', '365')),
            min_data_points_threshold=int(os.getenv('DCA_MIN_DATA_POINTS', '10')),
            data_quality_min_score=float(os.getenv('DCA_MIN_QUALITY_SCORE', '0.7'))
        )
        
        # Monitoring Configuration
        monitoring_config = MonitoringConfig(
            log_level=os.getenv('DCA_LOG_LEVEL', 'INFO'),
            progress_report_interval=int(os.getenv('DCA_PROGRESS_INTERVAL', '50')),
            metrics_enabled=os.getenv('DCA_METRICS_ENABLED', 'true').lower() == 'true'
        )
        
        return cls(
            api=api_config,
            storage=storage_config, 
            data=data_config,
            monitoring=monitoring_config
        )
    
    @classmethod
    def from_file(cls, config_path: Path) -> 'ProductionConfig':
        """Load configuration from JSON file"""
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path) as f:
            config_data = json.load(f)
        
        return cls(
            api=APIConfig(**config_data.get('api', {})),
            storage=StorageConfig(**config_data.get('storage', {})),
            data=DataConfig(**config_data.get('data', {})),
            monitoring=MonitoringConfig(**config_data.get('monitoring', {}))
        )
    
    def save_to_file(self, config_path: Path):
        """Save configuration to JSON file"""
        config_data = {
            'api': asdict(self.api),
            'storage': asdict(self.storage),
            'data': asdict(self.data),
            'monitoring': asdict(self.monitoring)
        }
        
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def validate(self) -> List[str]:
        """Validate configuration and return any errors"""
        errors = []
        
        # Validate paths exist
        base_path = Path(self.storage.base_path)
        if not base_path.exists():
            errors.append(f"Base data path does not exist: {base_path}")
        
        # Validate symbol universe file
        symbol_file = base_path / self.data.symbol_universe_file
        if not symbol_file.exists():
            errors.append(f"Symbol universe file not found: {symbol_file}")
        
        # Validate numeric ranges
        if self.api.rate_limit_seconds <= 0:
            errors.append("API rate limit must be positive")
        
        if self.data.data_quality_min_score < 0 or self.data.data_quality_min_score > 1:
            errors.append("Data quality minimum score must be between 0 and 1")
        
        # Validate log level
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.monitoring.log_level not in valid_log_levels:
            errors.append(f"Invalid log level: {self.monitoring.log_level}")
        
        return errors
    
    def setup_logging(self):
        """Set up logging based on configuration"""
        logging.basicConfig(
            level=getattr(logging, self.monitoring.log_level),
            format=self.monitoring.log_format,
            force=True  # Override any existing configuration
        )
        
        # Create logger for this module
        logger = logging.getLogger(__name__)
        logger.info(f"Logging configured at {self.monitoring.log_level} level")
        
        return logger


# Global configuration instance
_config_instance: Optional[ProductionConfig] = None


def get_config() -> ProductionConfig:
    """Get the global configuration instance"""
    global _config_instance
    
    if _config_instance is None:
        # Try to load from environment, fall back to defaults
        _config_instance = ProductionConfig.from_env()
        
        # Validate configuration
        errors = _config_instance.validate()
        if errors:
            logger = logging.getLogger(__name__)
            logger.warning("Configuration validation warnings:")
            for error in errors:
                logger.warning(f"  - {error}")
    
    return _config_instance


def set_config(config: ProductionConfig):
    """Set the global configuration instance"""
    global _config_instance
    _config_instance = config


def reset_config():
    """Reset configuration (mainly for testing)"""
    global _config_instance
    _config_instance = None


if __name__ == "__main__":
    # Demo/test the configuration
    config = get_config()
    print("🔧 Production Configuration:")
    print(f"  📁 Data Path: {config.storage.base_path}")
    print(f"  ⏱️  API Rate Limit: {config.api.rate_limit_seconds}s")
    print(f"  📊 Default History: {config.data.default_history_days} days")
    print(f"  📋 Log Level: {config.monitoring.log_level}")
    
    # Validation
    errors = config.validate()
    if errors:
        print("\n❌ Configuration Issues:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("\n✅ Configuration is valid")
