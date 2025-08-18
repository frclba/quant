"""
Layer 6: Orchestration & Monitoring

System coordination, performance analytics, and monitoring.
Provides centralized orchestration and comprehensive system observability.

Main Components:
- SystemOrchestrator: Workflow coordination
- PerformanceAnalytics: Return attribution and analysis
- MonitoringSystem: Health checks and alerting
- Layer6OrchestrationProcessor: Main coordinator
"""

from .system_orchestrator import SystemOrchestrator, OrchestrationConfig, WorkflowResult
from .performance_analytics import PerformanceAnalytics, AnalyticsConfig, PerformanceReport
from .monitoring_system import MonitoringSystem, MonitoringConfig, SystemHealth
from .layer6_processor import Layer6OrchestrationProcessor, Layer6Config, OrchestrationResult

__all__ = [
    # Main processor
    'Layer6OrchestrationProcessor',
    'Layer6Config',
    'OrchestrationResult',
    
    # Core systems
    'SystemOrchestrator',
    'PerformanceAnalytics', 
    'MonitoringSystem',
    
    # Configuration classes
    'OrchestrationConfig',
    'AnalyticsConfig',
    'MonitoringConfig',
    
    # Data structures
    'WorkflowResult',
    'PerformanceReport',
    'SystemHealth'
]
