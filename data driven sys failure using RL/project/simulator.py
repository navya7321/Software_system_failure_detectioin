import psutil
import random
from datetime import datetime
from models import Metric
from collections import deque

# Maintain rolling windows for adaptive detection
CPU_WINDOW = deque(maxlen=60)
MEM_WINDOW = deque(maxlen=60)


def read_current_metrics():
    cpu = float(psutil.cpu_percent(interval=0.1))
    memory = float(psutil.virtual_memory().percent)
    status = 'Healthy'
    # Introduce random transient spikes to simulate occasional issues
    if random.random() < 0.05:
        cpu = min(100.0, cpu + random.uniform(20, 50))
    if random.random() < 0.05:
        memory = min(100.0, memory + random.uniform(20, 50))
    # Update windows
    CPU_WINDOW.append(cpu)
    MEM_WINDOW.append(memory)
    # Adaptive thresholding using mean + 2*std when enough history exists
    import numpy as _np
    if len(CPU_WINDOW) >= 10 and len(MEM_WINDOW) >= 10:
        cpu_mean, cpu_std = _np.mean(CPU_WINDOW), _np.std(CPU_WINDOW)
        mem_mean, mem_std = _np.mean(MEM_WINDOW), _np.std(MEM_WINDOW)
        cpu_thresh = cpu_mean + 2 * max(cpu_std, 1e-3)
        mem_thresh = mem_mean + 2 * max(mem_std, 1e-3)
        if cpu >= cpu_thresh or memory >= mem_thresh:
            status = 'Failed'
    else:
        if cpu >= 90.0 or memory >= 90.0:
            status = 'Failed'
    return cpu, memory, status


def simulate_failure_metric():
    # Force a failure metric
    cpu = random.uniform(90.0, 100.0)
    memory = random.uniform(90.0, 100.0)
    return Metric(timestamp=datetime.utcnow(), cpu=cpu, memory=memory, status='Failed')
