"""
Logging utility for the Message Notification Router.
Sets up logging to both console and file as required by AGENTS.md.
"""

import logging
import os
from pathlib import Path

def setup_logging():
    """Set up logging configuration."""
    # Create logs directory if it doesn't exist
    log_dir = Path.home() / "hackerrank_orchestrate_august26"
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / "agent.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Also log to the mandatory chat transcript log as per AGENTS.md
    transcript_log = log_dir / "log.txt"
    # We'll handle the transcript logging in the main agent loop, but we can also set up a separate handler if needed.
    # For now, we'll just note that the main agent must log to that file.
    # We'll create a separate logger for the transcript if needed, but the main agent will handle it.
    
    # Log the start of the session
    logging.info("Logging initialized")

def log_transcript(message: str):
    """Log a message to the transcript log file as required by AGENTS.md."""
    log_dir = Path.home() / "hackerrank_orchestrate_august26"
    log_dir.mkdir(exist_ok=True)
    transcript_log = log_dir / "log.txt"
    
    from datetime import datetime
    timestamp = datetime.now().isoformat()
    log_entry = f"## [{timestamp}] {message}\n"
    
    with open(transcript_log, "a", encoding="utf-8") as f:
        f.write(log_entry)