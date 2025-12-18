"""Logging utility functions for the pipeline."""

import sys
import datetime

def get_timestamp() -> str:
    """Get current timestamp string."""
    return datetime.datetime.now().strftime("%H:%M:%S")

def print_log(message: str, end: str = "\n", flush: bool = True) -> None:
    """Print message to stdout with flush capability."""
    print(message, end=end, flush=flush)

def log_pipeline(message: str) -> None:
    """Log a pipeline stage message."""
    print_log(f"[{get_timestamp()}] ⚙️  {message}")

def log_stats(message: str) -> None:
    """Log a statistics message."""
    print_log(f"[{get_timestamp()}] 📊 {message}")

def log_done(message: str) -> None:
    """Log a completion message."""
    print_log(f"[{get_timestamp()}] ✅ {message}")

def log_error(message: str) -> None:
    """Log an error message."""
    print_log(f"[{get_timestamp()}] ❌ {message}")

def log_warning(message: str) -> None:
    """Log a warning message."""
    print_log(f"[{get_timestamp()}] ⚠️  {message}")