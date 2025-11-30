import time
import threading
import psutil
import numpy as np
import pynvml

class ResourceMonitor:
    def __init__(self, interval=0.1):
        self.interval = interval
        self.stop_event = threading.Event()
        self.cpu_log = []
        self.gpu_util_log = []
        self.gpu_mem_log = []
        self.thread = None
        
        # Initialize NVIDIA drivers
        try:
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0) # Logic for GPU 0
            self.has_gpu = True
        except Exception as e:
            print(f"⚠️ GPU monitoring failed (running CPU only): {e}")
            self.has_gpu = False

    def _monitor_loop(self):
        while not self.stop_event.is_set():
            # 1. Measure CPU (System-wide utilization)
            self.cpu_log.append(psutil.cpu_percent(interval=None))
            
            # 2. Measure GPU
            if self.has_gpu:
                try:
                    # GPU Utility (%)
                    util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                    self.gpu_util_log.append(util.gpu)
                    
                    # GPU Memory (MB)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                    self.gpu_mem_log.append(mem.used / 1024 / 1024) 
                except:
                    pass
            
            time.sleep(self.interval)

    def __enter__(self):
        # Start the recording thread
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Stop the recording thread
        self.stop_event.set()
        self.thread.join()
        if self.has_gpu:
            try:
                pynvml.nvmlShutdown()
            except:
                pass

    def get_stats(self):
        return {
            "avg_cpu_util": np.mean(self.cpu_log) if self.cpu_log else 0,
            "max_cpu_util": np.max(self.cpu_log) if self.cpu_log else 0,
            "avg_gpu_util": np.mean(self.gpu_util_log) if self.gpu_util_log else 0,
            "max_gpu_util": np.max(self.gpu_util_log) if self.gpu_util_log else 0,
            "avg_gpu_mem_mb": np.mean(self.gpu_mem_log) if self.gpu_mem_log else 0
        }