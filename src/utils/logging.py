"""Logging utilities for the application."""

import logging
import os
import sys
from datetime import datetime

def setup_logger(name, log_file=None, level=logging.INFO):
    """Set up a logger with the specified name and level."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Create a default logger
default_logger = setup_logger(
    'rag_system',
    log_file=f'logs/rag_system_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
)

def get_logger(name=None):
    """Get a logger with the specified name."""
    if name:
        return logging.getLogger(name)
    return default_logger