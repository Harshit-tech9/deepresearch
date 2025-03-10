"""Performance metrics utilities."""

import time
import functools
import psutil
import torch
from contextlib import contextmanager

class PerformanceTracker:
    """Track performance metrics for various operations."""
    
    def __init__(self, name="default"):
        """Initialize the performance tracker."""
        self.name = name
        self.start_time = None
        self.end_time = None
        self.cpu_percent_start = None
        self.cpu_percent_end = None
        self.memory_start = None
        self.memory_end = None
        self.gpu_memory_start = None
        self.gpu_memory_end = None
        self.gpu_available = False
    
    def start(self):
        """Start tracking performance."""
        self.start_time = time.time()
        self.cpu_percent_start = psutil.cpu_percent(interval=0.1)
        self.memory_start = psutil.virtual_memory().percent
        
        # Safely check for GPU and track memory
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                self.gpu_memory_start = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
                self.gpu_available = True
                print(f"GPU tracking enabled. Initial memory: {self.gpu_memory_start:.4f} GB")
            else:
                self.gpu_memory_start = 0
                print("GPU not available for tracking")
        except Exception as e:
            print(f"Warning: Failed to track GPU memory at start: {str(e)}")
            self.gpu_memory_start = 0
    
    def stop(self):
        """Stop tracking performance."""
        # Safely track GPU memory
        try:
            if self.gpu_available and torch.cuda.is_available():
                torch.cuda.synchronize()
                self.gpu_memory_end = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
                print(f"Final GPU memory: {self.gpu_memory_end:.4f} GB")
            else:
                self.gpu_memory_end = 0
        except Exception as e:
            print(f"Warning: Failed to track GPU memory at end: {str(e)}")
            self.gpu_memory_end = 0
        
        self.end_time = time.time()
        self.cpu_percent_end = psutil.cpu_percent(interval=0.1)
        self.memory_end = psutil.virtual_memory().percent
    
    def get_metrics(self):
        """Get the tracked performance metrics."""
        if self.start_time is None or self.end_time is None:
            return {"error": "Performance tracking not completed."}
        
        metrics = {
            "name": self.name,
            "elapsed_time": self.end_time - self.start_time,
            "cpu_percent_change": self.cpu_percent_end - self.cpu_percent_start,
            "memory_percent_change": self.memory_end - self.memory_start,
        }
        
        # Only add GPU metrics if tracking is available
        if self.gpu_available:
            metrics["gpu_memory_change_gb"] = self.gpu_memory_end - self.gpu_memory_start
        else:
            metrics["gpu_status"] = "GPU tracking not available"
        
        return metrics

def performance_monitor(func):
    """Decorator for monitoring function performance."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tracker = PerformanceTracker(func.__name__)
        tracker.start()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            tracker.stop()
            metrics = tracker.get_metrics()
            print(f"Performance metrics for {func.__name__}:")
            for key, value in metrics.items():
                if key != "name":
                    print(f"  {key}: {value}")
    return wrapper

# Properly implement track_performance_context
@contextmanager
def track_performance_context(name="operation"):
    """Context manager for tracking performance."""
    tracker = PerformanceTracker(name)
    tracker.start()
    try:
        yield tracker
    finally:
        tracker.stop()
        metrics = tracker.get_metrics()
        print(f"Performance metrics for {name}:")
        for key, value in metrics.items():
            if key != "name":
                print(f"  {key}: {value}")

# Simplified function to avoid any confusion
def track_performance(func):
    """Simple decorator for tracking performance."""
    return performance_monitor(func)