import time
import psutil
import pandas as pd
import numpy as np
from data_loader import DataLoader
from classifications import Classifications
from performance_monitor import monitor_performance, compare_array_operations

@monitor_performance
def load_process_upload_and_retrieve():
    # Initialize DataLoader with the data directory
    loader = DataLoader("D:/Facultate/Haller/PROIECT_STIINTA_DATELOR_PY")
    
    # Define CSV file, MongoDB parameters, and connection string.
    csv_filename = "job_descriptions.csv"
    db_name = "Proiect"
    collection_name = "Job_Data"
    client_string = "mongodb://localhost:27017/"
    
    # Upload CSV to MongoDB
    loader.upload_csv_to_mongodb(csv_filename, db_name, collection_name, client_uri=client_string)
    
    # Retrieve data from MongoDB
    retrieved_data = loader.retrieve_data_from_mongodb(db_name, collection_name, client_uri=client_string)
    if retrieved_data is None:
        retrieved_data = []
    print(f"\nRetrieved {len(retrieved_data)} records from MongoDB.")
    
    # Load CSV locally and process it with vectorized calls.
    df = loader.load_csv(csv_filename)
    numeric_columns = ["latitude", "longitude", "Company Size", "Preference_numeric"]
    loader.plot_data_description(df, numeric_columns)
    stats = loader.get_basic_stats(df)
    print("\nBasic Statistics:\n", pd.DataFrame([stats]).T)
    cleaned_df = loader.clean_data(df)
    return cleaned_df

def main():
    # Compare sum operations performance
    compare_array_operations()
    
    # Process data and get the cleaned DataFrame.
    df = load_process_upload_and_retrieve()
    
    cls = Classifications(df)

    cls.cluster_salary_by_geolocation()
    # Plot job posting trends over time (by month)
    cls.plot_event_time_trends()
    
    # Plot job posting trends by weekday
    cls.plot_job_posting_weekday_trends()
    
    # Plot the frequency of top 10 qualifications
    cls.plot_qualifications_skills_match()

    

if __name__ == "__main__":
    main()