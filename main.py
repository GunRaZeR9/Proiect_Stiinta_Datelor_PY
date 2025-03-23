import time
import psutil
import pandas as pd
import numpy as np
from data_loader import DataLoader
from classifications import Classifications
from performance_monitor import monitor_performance, compare_array_operations

@monitor_performance
def load_process_upload_and_retrieve():
    loader = DataLoader("D:/Facultate/Haller/PROIECT_STIINTA_DATELOR_PY")
    csv_filename = "job_descriptions.csv"
    db_name = "Proiect"
    collection_name = "Job_Data"
    client_string = "mongodb://localhost:27017/"

    loader.upload_csv_to_mongodb(csv_filename, db_name, collection_name, client_uri=client_string)
    retrieved_data = loader.retrieve_data_from_mongodb(db_name, collection_name, client_uri=client_string)
    if retrieved_data is None:
        retrieved_data = []
    print(f"\nRetrieved {len(retrieved_data)} records from MongoDB.")

    df = loader.load_csv(csv_filename)
    numeric_columns = ["latitude", "longitude", "Company Size", "Preference_numeric"]
    
    
    loader.plot_description_performance(df, numeric_columns)
    
    stats = loader.get_basic_stats(df)
    print("\nBasic Statistics:\n", pd.DataFrame([stats]).T)
    cleaned_df = loader.clean_data(df)
    return cleaned_df

def main():
    compare_array_operations()
    df = load_process_upload_and_retrieve()
    
    cls = Classifications(df)
    cls.cluster_salary_by_geolocation()
    cls.plot_event_time_trends()
    cls.plot_job_posting_weekday_trends()
    cls.plot_qualifications_skills_match()

if __name__ == "__main__":
    main()