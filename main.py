import time
import psutil
import pandas as pd
import numpy as np
from data_loader import DataLoader
from classifications import Classifications
from performance_monitor import monitor_performance, compare_array_operations

def pipeline(loader, csv_filename, numeric_columns, db_name, collection_name, client_string):
    # 1. Upload CSV to MongoDB (if needed) and retrieve data.
    loader.upload_csv_to_mongodb(csv_filename, db_name, collection_name, client_uri=client_string)
    retrieved_data = loader.retrieve_data_from_mongodb(db_name, collection_name, client_uri=client_string)
    if retrieved_data is None:
        retrieved_data = []
    print(f"\nRetrieved {len(retrieved_data)} records from MongoDB.")
    
    # 2. Load CSV locally.
    df = loader.load_csv(csv_filename)
    if df is None:
        print("Failed to load data.")
        return None
    
    # 3. Use only necessary numeric columns for performance/description.
    # Here, numeric_columns should be a list of columns you know will be used in the description.
    loader.plot_description_performance(df, numeric_columns)
    
    # 4. Print basic statistics.
    stats = loader.get_basic_stats(df)
    print("\nBasic Statistics:\n", pd.DataFrame([stats]).T)
    
    # 5. Clean the data.
    df_cleaned = loader.clean_data(df)
    
    # 6. (Optional) If you want to restrict the working dataset to only a subset (e.g. numeric columns)
    # you can select only those columns:
    # df_cleaned = df_cleaned[numeric_columns + other_required_columns]
    
    return df_cleaned

@monitor_performance
def load_process_upload_and_retrieve():
    loader = DataLoader("D:/Facultate/Haller/PROIECT_STIINTA_DATELOR_PY")
    csv_filename = "job_descriptions.csv"
    db_name = "Proiect"
    collection_name = "Job_Data"
    client_string = "mongodb://localhost:27017/"
    
    # Define the columns you need in this pipeline.
    numeric_columns = ["latitude", "longitude", "Company Size", "Preference_numeric"]
    
    # Run the pipeline.
    cleaned_df = pipeline(loader, csv_filename, numeric_columns, db_name, collection_name, client_string)
    return cleaned_df

def main():
    compare_array_operations()  # Run any performance or utility test if needed.
    df = load_process_upload_and_retrieve()
    if df is None:
        print("Pipeline failed, exiting.")
        return
    
    # Pass the cleaned data to the Classification methods.
    cls = Classifications(df)
    cls.cluster_salary_by_geolocation()
    cls.plot_event_time_trends()
    cls.plot_job_posting_weekday_trends()
    cls.plot_qualifications_skills_match()

if __name__ == "__main__":
    main()