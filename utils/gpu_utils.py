# common/gpu_utils.py

import subprocess
import json
import torch
import time



def get_gpu_stats():
    try:
        result = subprocess.run([
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,nounits,noheader"
        ], capture_output=True, text=True)

        gpu_lines = result.stdout.strip().split('\n')
        stats = []
        for line in gpu_lines:
            index, name, total, used, free, util = [x.strip() for x in line.split(',')]
            stats.append({
                "index": int(index),
                "name": name,
                "memory_total": int(total),
                "memory_used": int(used),
                "memory_free": int(free),
                "utilization_gpu": int(util),
            })

        # Get active processes
        proc_result = subprocess.run([
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,nounits,noheader"
        ], capture_output=True, text=True)

        process_lines = proc_result.stdout.strip().split('\n')
        processes = []
        for line in process_lines:
            if line:
                pid, name, mem = [x.strip() for x in line.split(',')]
                processes.append({
                    "pid": int(pid),
                    "name": name,
                    "memory": int(mem)
                })

        return {
            "gpus": stats,
            "processes": processes
        }

    except Exception as e:
        return {"error": str(e)}

def wait_for_vram(required_gb: float = 5.0, poll_interval: float = 0.5, timeout: float = 30.0) -> bool:
    """
    Blocks execution until at least `required_gb` GPU memory is free.

    Args:
        required_gb: GB of VRAM required to continue.
        poll_interval: Time between VRAM checks (in seconds).
        timeout: Max wait time (in seconds).

    Returns:
        True if memory became available, False if timeout occurred.
    """
    start = time.time()
    while time.time() - start < timeout:
        free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
        free_gb = free / 1024**3

        if free_gb >= required_gb:
            print(f"[VRAM-MUTEX] Sufficient VRAM free: {free_gb:.2f} GB (needed: {required_gb} GB)")
            return True

        print(f"[VRAM-MUTEX] Waiting for VRAM... Free: {free_gb:.2f} GB (needed: {required_gb} GB)")
        time.sleep(poll_interval)

    print(f"[VRAM-MUTEX] Timeout reached. VRAM never freed to {required_gb} GB.")
    return False