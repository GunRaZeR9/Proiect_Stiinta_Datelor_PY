import psutil
import time
import os

def monitor_performance(func):
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        
        mem_before = process.memory_info().rss / 1024 / 1024
        cpu_before = psutil.cpu_percent()
        mem_after = process.memory_info().rss / 1024 / 1024
        cpu_after = psutil.cpu_percent()

        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        memory_usage = mem_after - mem_before
        cpu_usage = cpu_after - cpu_before
        
        print(f"\nPerformance Metrics for {func.__name__}:")
        print(f"Execution time: {execution_time:.2f} seconds")
        print(f"Memory usage: {memory_usage:.2f} MB")
        print(f"CPU usage: {cpu_usage:.2f}%")
        
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        log_entry = (
            f"\n[{timestamp}] Function: {func.__name__}\n"
            f"Execution time: {execution_time:.2f} seconds\n"
            f"Memory usage: {memory_usage:.2f} MB\n"
            f"CPU usage: {cpu_usage:.2f}%\n"
            f"{'-'*50}"
        )
        with open('Performances.txt', 'a') as f:
            f.write(log_entry)
        
        return result
    return wrapper