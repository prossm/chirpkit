"""
Logging configuration for ChirpKit with custom formatting.
"""
import logging
import sys
from typing import Optional


class ChirpKitFormatter(logging.Formatter):
    """Custom formatter that adds the >>chirp: prefix to all log messages."""
    
    def format(self, record):
        # Get the original formatted message
        message = super().format(record)
        # Add the chirp prefix
        return f">>chirp: {message}"


def setup_chirpkit_logging(level: str = "INFO", 
                          format_string: Optional[str] = None) -> None:
    """
    Set up ChirpKit logging with custom formatting.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string (optional)
    """
    if format_string is None:
        format_string = "%(levelname)s - %(name)s - %(message)s"
    
    # Create and configure the formatter
    formatter = ChirpKitFormatter(format_string)
    
    # Get the root logger for chirpkit
    logger = logging.getLogger('chirpkit')
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove any existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Prevent propagation to avoid duplicate messages
    logger.propagate = False


def get_chirpkit_logger(name: str) -> logging.Logger:
    """
    Get a logger with the chirpkit prefix for the given module name.
    
    Args:
        name: Usually __name__ from the calling module
        
    Returns:
        Configured logger instance
    """
    # Ensure the base logger is set up
    base_logger = logging.getLogger('chirpkit')
    if not base_logger.handlers:
        setup_chirpkit_logging()
    
    # Return a child logger
    return logging.getLogger(f'chirpkit.{name}')