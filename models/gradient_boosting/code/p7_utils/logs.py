# utils.py

import time
import logging
import threading


def setup_logging():
    """Configure logging for the application."""
    # Get the root logger
    log = logging.getLogger()
    log.setLevel(logging.INFO)  # Set the minimum level to INFO

    # If handlers already exist, clear them to avoid duplicate outputs
    if log.hasHandlers():
        log.handlers.clear()

    # Create a stream handler to output to the console (or notebook cell)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(handler)
    return log


# Initialize a logger for this module
logger = logging.getLogger(__name__)


class Timer:
    """
    Context manager for timing code execution steps with data collection.
    """

    timing_data = {}
    _lock = threading.Lock()

    def __init__(self, step_name: str):
        self.step_name = step_name

    def __enter__(self):
        self.start = time.perf_counter()
        logger.info(f"\nStarting step: {self.step_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self.start
        logger.info(f"Completed step: {self.step_name} in {elapsed:.2f} seconds")
        with Timer._lock:
            Timer.timing_data[self.step_name] = elapsed

    @classmethod
    def get_timing_data(cls):
        """Return the collected timing data."""
        with cls._lock:
            return dict(cls.timing_data)
