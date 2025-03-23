import psutil
import time
import os
import numpy as np
import pandas as pd

def monitor_performance(func):
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024
        cpu_before = psutil.cpu_percent(interval=0.1)
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        mem_after = process.memory_info().rss / 1024 / 1024
        cpu_after = psutil.cpu_percent(interval=0.1)
        execution_time = end_time - start_time
        memory_usage = mem_after - mem_before
        cpu_usage = cpu_after - cpu_before
        
        print(f"\nPerformance Metrics for {func.__name__}:")
        print(f"Execution time: {execution_time:.6f} seconds")
        print(f"Memory usage: {memory_usage:.2f} MB")
        print(f"CPU usage: {cpu_usage:.2f}%")
        
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        log_entry = (
            f"\n[{timestamp}] Function: {func.__name__}\n"
            f"Execution time: {execution_time:.6f} seconds\n"
            f"Memory usage: {memory_usage:.2f} MB\n"
            f"CPU usage: {cpu_usage:.2f}%\n"
            f"{'-'*50}\n"
        )
        with open('Performances.txt', 'a') as f:
            f.write(log_entry)
        return result
    return wrapper

def compare_array_operations():
    """
    Compare the performance of summing 1M numbers using a regular list, a numpy array, and a pandas Series.
    Also report the data type used in each operation.
    """
    size = 10**6
    data_list = list(range(size))
    data_np = np.arange(size)
    data_pd = pd.Series(data_list)
    
    # List sum
    start = time.time()
    sum_list = sum(data_list)
    time_list = time.time() - start
    
    # Numpy sum
    start = time.time()
    sum_np = np.sum(data_np)
    time_np = time.time() - start
    
    # Pandas sum
    start = time.time()
    sum_pd = data_pd.sum()
    time_pd = time.time() - start

    messages = [
        f"List sum time: {time_list:.6f} seconds using type {type(data_list)}",
        f"Numpy sum time: {time_np:.6f} seconds using type {type(data_np)}",
        f"Pandas sum time: {time_pd:.6f} seconds using type {type(data_pd)}"
    ]
    print("\n".join(messages))
    return (sum_list, sum_np, sum_pd)