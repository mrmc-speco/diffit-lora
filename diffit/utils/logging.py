"""
Logging Setup for DiffiT

Centralized logging configuration.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    name: Optional[str] = None,
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
):
    """
    Setup logging configuration and return a logger
    
    Args:
        name: Logger name (if None, configures root logger and returns it)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        format_string: Optional custom format string
    
    Returns:
        Logger instance
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Get numeric level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format=format_string,
        handlers=[]
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_formatter = logging.Formatter(format_string)
    console_handler.setFormatter(console_formatter)
    
    # File handler (if specified)
    handlers = [console_handler]
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_formatter = logging.Formatter(format_string)
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)
    
    # Apply handlers to root logger
    root_logger = logging.getLogger()
    root_logger.handlers = handlers
    
    # Set specific logger levels
    logging.getLogger('pytorch_lightning').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    
    # Return the appropriate logger
    if name:
        return logging.getLogger(name)
    else:
        return logging.getLogger()


def get_logger(name: str):
    """
    Get a logger with default configuration
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # If logger has no handlers, set up basic configuration
    if not logger.handlers and not logging.getLogger().handlers:
        setup_logging()
    
    return logger
