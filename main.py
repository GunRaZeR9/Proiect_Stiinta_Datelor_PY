import psutil
import time
from data_loader import DataLoader
from classifications import Classifications
import os
from performance_monitor import monitor_performance

@monitor_performance
def load_and_process_data():
    loader = DataLoader("D:/Facultate/Haller", nrows=4000000, random_state=42)
    df = loader.load_csv("2019-Nov.csv")
    
    if df is not None:
        stats = loader.get_basic_stats(df)
        print("\nBasic Statistics:")
        for key, value in stats.items():
            if key == 'missing_values':
                print(f"\n{key}:")
                for col, count in value.items():
                    print(f"  {col}: {count}")
            else:
                print(f"{key}: {value}")
        
        print("\nCleaning data...")
        return loader.clean_data(df)
    return None


def main():
    load_and_process_data()

if __name__ == "__main__":
    main()