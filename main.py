import psutil
import time
from data_loader import DataLoader
from classifications import Classifications
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

@monitor_performance
def load_and_process_data():
    loader = DataLoader("D:/Projects/Proiect_Stiinta_Datelor_PY", nrows=10000, random_state=42)
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

@monitor_performance
def main():
    df_cleaned = load_and_process_data()
    
    if df_cleaned is not None:
        classifier = Classifications(df_cleaned)
        
        print("\nEvent Type Distribution:")
        print(classifier.event_type_analysis())
        
        print("\nProduct Popularity Analysis:")
        print(classifier.product_popularity_analysis())
        
        print("\nHourly Sales Analysis:")
        print(classifier.hourly_sales_analysis())

        print("\nPrice Statistics:")
        print(classifier.calculate_price_statistics())

        print("\nDetected Price Outliers:")
        print(classifier.detect_price_outliers())

        print("\nPrice Correlation Analysis:")
        print(classifier.price_correlation_analysis())

        print("\nPrice Based Clustering Results:")
        print(classifier.price_based_clustering().head())

        print("\nVisualizing Price Trends...")
        classifier.visualize_price_trends()

        print("\nComparing Price Analysis Methods:")
        classifier.compare_price_analysis()

        plots = ['price_segments', 'time_patterns', 'popular_categories', 
                'brand_engagement', 'price_distribution', 'event_types', 
                'hourly_sales']
        
        for plot_type in plots:
            print(f"\nGenerating {plot_type} plot...")
            classifier.plot_results(plot_type)

        #print("\n3D Graph...")
        #classifier.visualize_price_trends_3d()


if __name__ == "__main__":
    main()