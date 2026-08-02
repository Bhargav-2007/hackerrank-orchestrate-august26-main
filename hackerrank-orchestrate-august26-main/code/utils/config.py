"""
Configuration module for the Message Notification Router.
"""

from pathlib import Path

# Base path for the dataset
DATASET_PATH = Path(__file__).parent.parent / "dataset"

# You can also add other configuration parameters here, such as:
# - Model paths
# - Thresholds
# - API keys (to be loaded from environment variables)