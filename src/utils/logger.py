"""Logging utilities for the NIDS system."""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
import time


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.
        
        Args:
            record: Log record to format.
            
        Returns:
            JSON formatted log string.
        """
        log_entry = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(record.created)),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add extra fields if they exist
        if hasattr(record, 'extra_data'):
            log_entry.update(record.extra_data)
            
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
            
        return json.dumps(log_entry)


class ContextFilter(logging.Filter):
    """Add contextual information to log records."""
    
    def __init__(self, context: Optional[Dict[str, Any]] = None):
        """Initialize context filter.
        
        Args:
            context: Context data to add to log records.
        """
        super().__init__()
        self.context = context or {}
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add context to log record.
        
        Args:
            record: Log record to filter.
            
        Returns:
            True (always pass the record).
        """
        for key, value in self.context.items():
            setattr(record, key, value)
        return True


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    json_format: bool = False,
    max_bytes: int = 10485760,  # 10MB
    backup_count: int = 5,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """Set up logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Path to log file. If None, only console logging.
        json_format: Whether to use JSON formatting.
        max_bytes: Maximum bytes per log file before rotation.
        backup_count: Number of backup files to keep.
        context: Additional context to add to all log records.
    """
    # Clear existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set root logger level
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Create formatters
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    
    # Add context filter if provided
    if context:
        context_filter = ContextFilter(context)
        console_handler.addFilter(context_filter)
    
    root_logger.addHandler(console_handler)
    
    # File handler (if log_file provided)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)  # Always debug level for files
        file_handler.setFormatter(formatter)
        
        if context:
            file_handler.addFilter(context_filter)
        
        root_logger.addHandler(file_handler)
    
    # Suppress some noisy loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)


def get_logger(
    name: str,
    level: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
) -> logging.Logger:
    """Get a logger instance with optional configuration.
    
    Args:
        name: Logger name (usually __name__).
        level: Logging level for this logger.
        context: Additional context for this logger.
        
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    
    if level:
        logger.setLevel(getattr(logging, level.upper()))
    
    if context:
        # Add a context filter to this specific logger
        context_filter = ContextFilter(context)
        logger.addFilter(context_filter)
    
    return logger


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = None
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class.
        
        Returns:
            Logger instance.
        """
        if self._logger is None:
            self._logger = get_logger(
                f"{self.__class__.__module__}.{self.__class__.__name__}"
            )
        return self._logger
    
    def log_method_call(self, method_name: str, **kwargs) -> None:
        """Log method call with arguments.
        
        Args:
            method_name: Name of the method being called.
            **kwargs: Method arguments to log.
        """
        self.logger.debug(
            f"Calling {method_name}",
            extra={'extra_data': {'method': method_name, 'args': kwargs}}
        )
    
    def log_performance(self, operation: str, duration: float, **metrics) -> None:
        """Log performance metrics.
        
        Args:
            operation: Name of the operation.
            duration: Duration in seconds.
            **metrics: Additional performance metrics.
        """
        performance_data = {
            'operation': operation,
            'duration_seconds': duration,
            **metrics
        }
        
        self.logger.info(
            f"Performance: {operation} completed in {duration:.3f}s",
            extra={'extra_data': performance_data}
        )


def log_function_call(func):
    """Decorator to log function calls.
    
    Args:
        func: Function to decorate.
        
    Returns:
        Decorated function.
    """
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.debug(f"{func.__name__} completed in {duration:.3f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"{func.__name__} failed after {duration:.3f}s: {e}")
            raise
    
    return wrapper


def log_errors(func):
    """Decorator to log function errors.
    
    Args:
        func: Function to decorate.
        
    Returns:
        Decorated function.
    """
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(
                f"Error in {func.__name__}: {e}",
                exc_info=True,
                extra={'extra_data': {'function': func.__name__, 'error': str(e)}}
            )
            raise
    
    return wrapper


# Default logger for the package
default_logger = get_logger(__name__)


def configure_default_logging():
    """Configure default logging for the package."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    setup_logging(
        level="INFO",
        log_file=log_dir / "nids.log",
        json_format=False
    )
