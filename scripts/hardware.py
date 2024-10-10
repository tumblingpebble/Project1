import platform
import psutil
import os
import multiprocessing
import subprocess
import sklearn
print(sklearn.__version__)


def get_gpu_info():
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name,memory.free', '--format=csv,noheader,nounits'],
            encoding='utf-8'
        )
        gpus = []
        for line in result.strip().split('\n'):
            name, memory_free = line.split(', ')
            gpus.append({'name': name, 'memory_free_MB': int(memory_free)})
        return gpus
    except Exception as e:
        print(f"Error retrieving GPU information: {e}")
        return None

gpu_info = get_gpu_info()
if gpu_info:
    for idx, gpu in enumerate(gpu_info):
        print(f"GPU {idx}: {gpu['name']} with {gpu['memory_free_MB']} MB free memory")
else:
    print("No NVIDIA GPU detected or unable to retrieve GPU information.")


# CPU information
print(f"Processor: {platform.processor()}")
print(f"Number of CPU Cores: {multiprocessing.cpu_count()}")
print(f"CPU Frequency: {psutil.cpu_freq().current:.2f} MHz")

# Get the number of physical cores
physical_cores = psutil.cpu_count(logical=False)
print(f"Physical CPU cores: {physical_cores}")

# Get the number of logical cores (including hyper-threading)
logical_cores = psutil.cpu_count(logical=True)
print(f"Logical CPU cores: {logical_cores}")

# Memory information
mem = psutil.virtual_memory()
print(f"Total memory: {mem.total / (1024 ** 3):.2f} GB")
print(f"Available memory: {mem.available / (1024 ** 3):.2f} GB")
print(f"Memory usage: {mem.percent}%")

# Disk information
disk = psutil.disk_usage('/')
print(f"Disk total: {disk.total / (1024 ** 3):.2f} GB")
print(f"Disk used: {disk.used / (1024 ** 3):.2f} GB")
print(f"Disk free: {disk.free / (1024 ** 3):.2f} GB")
print(f"Disk usage: {disk.percent}%")

# Swap memory information
swap = psutil.swap_memory()
print(f"Total swap: {swap.total / (1024 ** 3):.2f} GB")
print(f"Swap used: {swap.used / (1024 ** 3):.2f} GB")
print(f"Swap free: {swap.free / (1024 ** 3):.2f} GB")
print(f"Swap usage: {swap.percent}%")

# GPU information (if applicable)
try:
    # nvidia-smi command output
    gpu_info = os.popen('nvidia-smi').read()
    print("GPU information:\n", gpu_info)
except Exception as e:
    print("No GPU detected or unable to run nvidia-smi")

# Checking current resource usage for CPU, memory, and disk
cpu_usage = psutil.cpu_percent(interval=1)
print(f"Current CPU usage: {cpu_usage}%")

memory_usage = psutil.virtual_memory().percent
print(f"Current memory usage: {memory_usage}%")

disk_usage = psutil.disk_usage('/').percent
print(f"Current disk usage: {disk_usage}%")
