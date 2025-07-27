"""Metrics utilities for performance tracking and monitoring."""

import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from collections import defaultdict, deque
import numpy as np

from .constants import PerformanceConstants, LoggingConstants
from .logger import get_logger

logger = get_logger(__name__)


class MetricsCollector:
    """Utility class for collecting and aggregating metrics."""
    
    def __init__(self, max_history: int = PerformanceConstants.METRICS_HISTORY_SIZE):
        """Initialize metrics collector.
        
        Args:
            max_history: Maximum number of historical metrics to keep.
        """
        self.max_history = max_history
        self.metrics_history = defaultdict(lambda: deque(maxlen=max_history))
        self.counters = defaultdict(int)
        self.timers = {}
        self.start_time = time.time()
    
    def record_metric(self, name: str, value: Union[int, float], 
                     timestamp: Optional[float] = None) -> None:
        """Record a metric value.
        
        Args:
            name: Metric name.
            value: Metric value.
            timestamp: Timestamp (uses current time if None).
        """
        if timestamp is None:
            timestamp = time.time()
        
        self.metrics_history[name].append({
            "value": float(value),
            "timestamp": timestamp
        })
        
        logger.debug(f"Recorded metric {name}: {value}")
    
    def increment_counter(self, name: str, amount: int = 1) -> None:
        """Increment a counter metric.
        
        Args:
            name: Counter name.
            amount: Amount to increment.
        """
        self.counters[name] += amount
        self.record_metric(f"{name}_total", self.counters[name])
    
    def start_timer(self, name: str) -> None:
        """Start a timer.
        
        Args:
            name: Timer name.
        """
        self.timers[name] = time.time()
    
    def stop_timer(self, name: str) -> Optional[float]:
        """Stop a timer and record the duration.
        
        Args:
            name: Timer name.
            
        Returns:
            Duration in seconds or None if timer not found.
        """
        if name not in self.timers:
            logger.warning(f"Timer {name} not found")
            return None
        
        duration = time.time() - self.timers[name]
        self.record_metric(f"{name}_duration", duration)
        del self.timers[name]
        
        return duration
    
    def get_metric_stats(self, name: str, window_seconds: Optional[float] = None) -> Dict[str, float]:
        """Get statistics for a metric.
        
        Args:
            name: Metric name.
            window_seconds: Time window to consider (all history if None).
            
        Returns:
            Dictionary with metric statistics.
        """
        if name not in self.metrics_history:
            return {}
        
        values = []
        current_time = time.time()
        
        for entry in self.metrics_history[name]:
            if window_seconds is None or (current_time - entry["timestamp"]) <= window_seconds:
                values.append(entry["value"])
        
        if not values:
            return {}
        
        return {
            "count": len(values),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99))
        }
    
    def get_rate(self, counter_name: str, window_seconds: float = 60.0) -> float:
        """Get rate for a counter metric.
        
        Args:
            counter_name: Counter name.
            window_seconds: Time window for rate calculation.
            
        Returns:
            Rate per second.
        """
        metric_name = f"{counter_name}_total"
        if metric_name not in self.metrics_history:
            return 0.0
        
        current_time = time.time()
        values_in_window = []
        
        for entry in self.metrics_history[metric_name]:
            if (current_time - entry["timestamp"]) <= window_seconds:
                values_in_window.append(entry)
        
        if len(values_in_window) < 2:
            return 0.0
        
        # Calculate rate based on first and last values in window
        first_entry = values_in_window[0]
        last_entry = values_in_window[-1]
        
        value_diff = last_entry["value"] - first_entry["value"]
        time_diff = last_entry["timestamp"] - first_entry["timestamp"]
        
        return value_diff / time_diff if time_diff > 0 else 0.0
    
    def get_summary(self, window_seconds: Optional[float] = None) -> Dict[str, Any]:
        """Get summary of all metrics.
        
        Args:
            window_seconds: Time window to consider (all history if None).
            
        Returns:
            Dictionary with metrics summary.
        """
        summary = {
            "collection_start_time": self.start_time,
            "current_time": time.time(),
            "uptime_seconds": time.time() - self.start_time,
            "metrics": {},
            "counters": dict(self.counters)
        }
        
        # Add metric statistics
        for metric_name in self.metrics_history:
            stats = self.get_metric_stats(metric_name, window_seconds)
            if stats:
                summary["metrics"][metric_name] = stats
        
        return summary
    
    def export_metrics(self, file_path: Path) -> None:
        """Export metrics to JSON file.
        
        Args:
            file_path: Path to export file.
        """
        summary = self.get_summary()
        
        # Convert deques to lists for JSON serialization
        export_data = {
            "summary": summary,
            "raw_metrics": {}
        }
        
        for name, history in self.metrics_history.items():
            export_data["raw_metrics"][name] = list(history)
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Metrics exported to: {file_path}")


class PerformanceMonitor:
    """Performance monitoring utilities."""
    
    def __init__(self):
        """Initialize performance monitor."""
        self.metrics_collector = MetricsCollector()
    
    def monitor_prediction_performance(self, func):
        """Decorator to monitor prediction performance.
        
        Args:
            func: Function to monitor.
            
        Returns:
            Decorated function.
        """
        def wrapper(*args, **kwargs):
            self.metrics_collector.start_timer("prediction")
            self.metrics_collector.increment_counter("predictions")
            
            try:
                result = func(*args, **kwargs)
                self.metrics_collector.increment_counter("predictions_success")
                return result
            except Exception as e:
                self.metrics_collector.increment_counter("predictions_error")
                logger.error(f"Prediction error: {e}")
                raise
            finally:
                duration = self.metrics_collector.stop_timer("prediction")
                if duration:
                    self.metrics_collector.record_metric("prediction_latency", duration)
        
        return wrapper
    
    def monitor_training_performance(self, func):
        """Decorator to monitor training performance.
        
        Args:
            func: Function to monitor.
            
        Returns:
            Decorated function.
        """
        def wrapper(*args, **kwargs):
            self.metrics_collector.start_timer("training")
            self.metrics_collector.increment_counter("training_sessions")
            
            try:
                result = func(*args, **kwargs)
                self.metrics_collector.increment_counter("training_success")
                return result
            except Exception as e:
                self.metrics_collector.increment_counter("training_error")
                logger.error(f"Training error: {e}")
                raise
            finally:
                duration = self.metrics_collector.stop_timer("training")
                if duration:
                    self.metrics_collector.record_metric("training_duration", duration)
        
        return wrapper
    
    def record_model_metrics(self, metrics: Dict[str, float], 
                           model_name: str = "default") -> None:
        """Record model performance metrics.
        
        Args:
            metrics: Dictionary of metrics to record.
            model_name: Name of the model.
        """
        for metric_name, value in metrics.items():
            full_name = f"model_{model_name}_{metric_name}"
            self.metrics_collector.record_metric(full_name, value)
    
    def record_data_metrics(self, data_stats: Dict[str, Any]) -> None:
        """Record data processing metrics.
        
        Args:
            data_stats: Dictionary of data statistics.
        """
        if "num_samples" in data_stats:
            self.metrics_collector.record_metric("data_samples", data_stats["num_samples"])
        
        if "num_features" in data_stats:
            self.metrics_collector.record_metric("data_features", data_stats["num_features"])
        
        if "anomaly_rate" in data_stats:
            self.metrics_collector.record_metric("data_anomaly_rate", data_stats["anomaly_rate"])
        
        if "processing_time" in data_stats:
            self.metrics_collector.record_metric("data_processing_time", data_stats["processing_time"])
    
    def get_performance_report(self, window_minutes: int = 60) -> Dict[str, Any]:
        """Generate performance report.
        
        Args:
            window_minutes: Time window in minutes.
            
        Returns:
            Performance report.
        """
        window_seconds = window_minutes * 60
        summary = self.metrics_collector.get_summary(window_seconds)
        
        # Calculate rates
        prediction_rate = self.metrics_collector.get_rate("predictions", window_seconds)
        error_rate = self.metrics_collector.get_rate("predictions_error", window_seconds)
        
        # Get latency statistics
        latency_stats = self.metrics_collector.get_metric_stats(
            "prediction_latency", window_seconds
        )
        
        report = {
            "time_window_minutes": window_minutes,
            "performance": {
                "prediction_rate_per_second": round(prediction_rate, 2),
                "error_rate_per_second": round(error_rate, 2),
                "success_rate_percent": self._calculate_success_rate(window_seconds),
                "latency_stats": latency_stats
            },
            "resource_usage": self._get_resource_usage(),
            "summary": summary
        }
        
        return report
    
    def _calculate_success_rate(self, window_seconds: float) -> float:
        """Calculate success rate percentage.
        
        Args:
            window_seconds: Time window in seconds (unused but kept for interface consistency).
            
        Returns:
            Success rate as percentage.
        """
        # Note: Currently calculates overall rate, could be enhanced to use window_seconds
        total_requests = self.metrics_collector.counters.get("predictions", 0)
        successful_requests = self.metrics_collector.counters.get("predictions_success", 0)
        
        if total_requests == 0:
            return 100.0
        
        return round((successful_requests / total_requests) * 100, 2)
    
    def _get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage.
        
        Returns:
            Resource usage information.
        """
        usage = {}
        
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            
            # Memory usage
            memory_info = process.memory_info()
            usage["memory"] = {
                "rss_mb": round(memory_info.rss / (1024 * 1024), 2),
                "vms_mb": round(memory_info.vms / (1024 * 1024), 2),
                "percent": round(process.memory_percent(), 2)
            }
            
            # CPU usage
            usage["cpu"] = {
                "percent": round(process.cpu_percent(), 2),
                "num_threads": process.num_threads()
            }
            
        except ImportError:
            usage["message"] = "psutil not available for resource monitoring"
        except Exception as e:
            usage["error"] = str(e)
        
        return usage


class AlertManager:
    """Alert management for metrics thresholds."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        """Initialize alert manager.
        
        Args:
            metrics_collector: Metrics collector instance.
        """
        self.metrics_collector = metrics_collector
        self.thresholds = {}
        self.alert_history = deque(maxlen=100)
    
    def set_threshold(self, metric_name: str, threshold: float, 
                     condition: str = "greater") -> None:
        """Set alert threshold for a metric.
        
        Args:
            metric_name: Name of the metric.
            threshold: Threshold value.
            condition: Condition for alert ('greater', 'less', 'equal').
        """
        self.thresholds[metric_name] = {
            "threshold": threshold,
            "condition": condition
        }
        
        logger.info(f"Set alert threshold for {metric_name}: {condition} {threshold}")
    
    def check_alerts(self) -> List[Dict[str, Any]]:
        """Check for metric threshold violations.
        
        Returns:
            List of active alerts.
        """
        active_alerts = []
        
        for metric_name, threshold_config in self.thresholds.items():
            if metric_name not in self.metrics_collector.metrics_history:
                continue
            
            # Get latest value
            history = self.metrics_collector.metrics_history[metric_name]
            if not history:
                continue
            
            latest_value = history[-1]["value"]
            threshold = threshold_config["threshold"]
            condition = threshold_config["condition"]
            
            # Check condition
            alert_triggered = (
                (condition == "greater" and latest_value > threshold) or
                (condition == "less" and latest_value < threshold) or
                (condition == "equal" and abs(latest_value - threshold) < 1e-6)
            )
            
            if alert_triggered:
                alert = {
                    "metric_name": metric_name,
                    "current_value": latest_value,
                    "threshold": threshold,
                    "condition": condition,
                    "timestamp": time.time(),
                    "severity": self._determine_severity(metric_name, latest_value, threshold)
                }
                
                active_alerts.append(alert)
                self.alert_history.append(alert)
                
                logger.warning(f"Alert triggered for {metric_name}: {latest_value} {condition} {threshold}")
        
        return active_alerts
    
    def _determine_severity(self, metric_name: str, value: float, threshold: float) -> str:
        """Determine alert severity.
        
        Args:
            metric_name: Name of the metric.
            value: Current value.
            threshold: Threshold value.
            
        Returns:
            Severity level ('low', 'medium', 'high', 'critical').
        """
        # Simple severity determination based on how much threshold is exceeded
        if "error" in metric_name.lower():
            return "high"
        elif "latency" in metric_name.lower() or "duration" in metric_name.lower():
            ratio = value / threshold if threshold > 0 else 1
            if ratio > 2.0:
                return "critical"
            elif ratio > 1.5:
                return "high"
            elif ratio > 1.2:
                return "medium"
            else:
                return "low"
        else:
            return "medium"
    
    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get alert summary for specified time period.
        
        Args:
            hours: Number of hours to look back.
            
        Returns:
            Alert summary.
        """
        cutoff_time = time.time() - (hours * 3600)
        recent_alerts = [
            alert for alert in self.alert_history
            if alert["timestamp"] >= cutoff_time
        ]
        
        # Count by severity
        severity_counts = defaultdict(int)
        metric_counts = defaultdict(int)
        
        for alert in recent_alerts:
            severity_counts[alert["severity"]] += 1
            metric_counts[alert["metric_name"]] += 1
        
        return {
            "time_period_hours": hours,
            "total_alerts": len(recent_alerts),
            "alerts_by_severity": dict(severity_counts),
            "alerts_by_metric": dict(metric_counts),
            "recent_alerts": recent_alerts[-10:]  # Last 10 alerts
        }
