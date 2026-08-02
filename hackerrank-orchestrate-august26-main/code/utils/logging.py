"""Logging configuration compliant with AGENTS.md"""
import logging
import os
from datetime import datetime

# Create logs directory if it doesn't exist
logs_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'logs')
os.makedirs(logs_dir, exist_ok=True)

# Log file path - fixed name as required
log_file = os.path.join(logs_dir, 'log.txt')

# Configure logger
logger = logging.getLogger('notification_router')
logger.setLevel(logging.DEBUG)

# Prevent duplicate handlers
if not logger.handlers:
    # File handler
    file_handler = logging.FileHandler(log_file, mode='w')  # Overwrite each run
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # INFO level for console to reduce noise
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# Export the logger for use in other modules
def get_logger(name: str = None):
    """Get a logger instance"""
    if name:
        return logging.getLogger(f'notification_router.{name}')
    return logger

# For backward compatibility
setup_logger = get_logger