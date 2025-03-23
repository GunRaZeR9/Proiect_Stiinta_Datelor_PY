import pandas as pd
from pathlib import Path
import numpy as np
from pymongo import MongoClient
from datetime import datetime
import matplotlib.pyplot as plt

class DataLoader:
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        
    def load_csv(self, filename):
        file_path = self.data_path / filename
        try:
            df = pd.read_csv(file_path)
            
            # Convert 'Preference' to numeric values if present.
            if "Preference" in df.columns:
                mapping = {"Male": 0, "Female": 1, "Both": 2}
                df["Preference_numeric"] = df["Preference"].map(mapping)
            
            print("\nData Preview (first 5 rows):")
            print(df.head())
            
            print("\nDetailed Data Description:")
            # Use to_markdown() for a nicely formatted table output in the terminal.
            print(df.describe(include='all').to_markdown())
            
            print(f"\nSuccessfully loaded data from {filename}")
            return df
            
        except Exception as e:
            print(f"Error loading {filename}: {str(e)}")
            return None
    
    def get_basic_stats(self, df):
        return {
            'num_rows': len(df),
            'num_columns': len(df.columns),
            'columns': list(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'dtypes': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2
        }
    
    def clean_data(self, df):
        print("\nData shape before cleaning:", df.shape)
        df = df.drop_duplicates()
        print("Shape after removing duplicates:", df.shape)
        df = df.dropna()
        print("Shape after removing missing values:", df.shape)
        return df

    def load_all_csvs(self, clean=False):
        dfs = {}
        for file in self.data_path.glob("*.csv"):
            print(f"\nLoading file: {file.name}")
            df = self.load_csv(file.name)
            if df is not None and clean:
                df = self.clean_data(df)
            dfs[file.name] = df
        return dfs

    def upload_csv_to_mongodb(self, filename, db_name, collection_name, client_uri="mongodb://localhost:27017/"):

        client = MongoClient(client_uri)
        db = client[db_name]
        meta_collection = db["uploaded_files"]
        
        # Check if the file was already uploaded
        if meta_collection.find_one({"filename": filename}):
            print(f"{filename} is already uploaded. Skipping upload.")
            client.close()
            return
        
        df = self.load_csv(filename)
        if df is not None:
            records = df.to_dict("records")
            if records:
                db[collection_name].insert_many(records)
                # Record the upload metadata
                meta_collection.insert_one({
                    "filename": filename,
                    "uploaded_at": datetime.utcnow()
                })
                print(f"Uploaded data from {filename} to MongoDB collection '{collection_name}'.")
            else:
                print(f"No records found in {filename} to upload.")
        client.close()

    def retrieve_data_from_mongodb(self, db_name, collection_name, query={}, client_uri="mongodb://localhost:27017/"):

        client = MongoClient(client_uri)
        db = client[db_name]
        data = list(db[collection_name].find(query))
        client.close()
        return data

    def plot_description_performance(self, df, numeric_columns):
        """
        Measures and plots performance (execution time and memory) for computing descriptive
        statistics using Pandas, NumPy, and a pure Python Array approach. It also displays the
        descriptive statistics computed by each method.
        """
        import time
        import tracemalloc

        # ------------------
        # Pandas measurement
        tracemalloc.start()
        start_time = time.time()
        desc_pandas = df[numeric_columns].describe().round(2)
        pandas_time = time.time() - start_time
        pandas_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        pandas_mem_usage = (pandas_mem[1] - pandas_mem[0]) / (1024 * 1024)
        print(f"Pandas: {pandas_time:.4f} sec, {pandas_mem_usage:.4f} MB")

        # ------------------
        # NumPy measurement
        tracemalloc.start()
        start_time = time.time()
        data_numpy = df[numeric_columns].to_numpy()
        desc_numpy = {
            "count": np.count_nonzero(~np.isnan(data_numpy), axis=0),
            "mean": np.nanmean(data_numpy, axis=0),
            "std": np.nanstd(data_numpy, axis=0),
            "min": np.nanmin(data_numpy, axis=0),
            "25%": np.nanpercentile(data_numpy, 25, axis=0),
            "50%": np.nanpercentile(data_numpy, 50, axis=0),
            "75%": np.nanpercentile(data_numpy, 75, axis=0),
            "max": np.nanmax(data_numpy, axis=0),
        }
        numpy_time = time.time() - start_time
        numpy_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        numpy_mem_usage = (numpy_mem[1] - numpy_mem[0]) / (1024 * 1024)
        print(f"NumPy: {numpy_time:.4f} sec, {numpy_mem_usage:.4f} MB")

        # ------------------
        # Array (pure Python list) measurement
        data_list = df[numeric_columns].values.tolist()
        tracemalloc.start()
        start_time = time.time()
        # Transform rows into columns.
        data_list_numeric = [
            [float(row[i]) if isinstance(row[i], (int, float)) or 
             (isinstance(row[i], str) and row[i].replace('.', '', 1).isdigit())
             else float("nan")
             for row in data_list
            ] for i in range(len(numeric_columns))
        ]
        desc_list = {
            "count": [len([x for x in col if not np.isnan(x)]) for col in data_list_numeric],
            "mean": [np.nanmean(col) for col in data_list_numeric],
            "std": [np.nanstd(col) for col in data_list_numeric],
            "min": [np.nanmin(col) for col in data_list_numeric],
            "25%": [np.nanpercentile(col, 25) for col in data_list_numeric],
            "50%": [np.nanpercentile(col, 50) for col in data_list_numeric],
            "75%": [np.nanpercentile(col, 75) for col in data_list_numeric],
            "max": [np.nanmax(col) for col in data_list_numeric],
        }
        list_time = time.time() - start_time
        list_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        list_mem_usage = (list_mem[1] - list_mem[0]) / (1024 * 1024)
        print(f"Array: {list_time:.4f} sec, {list_mem_usage:.4f} MB")

        # ------------------
        # Create and display the performance results table.
        results = pd.DataFrame({
            "Metodă": ["Pandas", "NumPy", "Array"],
            "Timp de rulare (sec)": [pandas_time, numpy_time, list_time],
            "Memorie utilizată (MB)": [pandas_mem_usage, numpy_mem_usage, list_mem_usage]
        })
        print("Compararea Performanței:")
        print(results)

        fig_perf, ax_perf = plt.subplots(figsize=(8, 2))
        ax_perf.axis('tight')
        ax_perf.axis('off')
        table_perf = ax_perf.table(
            cellText=results.values,
            colLabels=results.columns,
            loc='center'
        )
        table_perf.auto_set_font_size(False)
        table_perf.set_fontsize(10)
        plt.title("Compararea Performanței: Pandas vs NumPy vs Array", pad=20)
        plt.show()

        # ------------------
        # Display Descriptive Statistics for each method.
        # For consistency, the NumPy and Array results are transformed to a DataFrame with the same format as Pandas.
        
        # Pandas Descriptive Statistics are already in the right format.
        fig_pandas, ax_pandas = plt.subplots(figsize=(10, desc_pandas.shape[0]*0.6 + 2))
        ax_pandas.axis('tight')
        ax_pandas.axis('off')
        table_pandas = ax_pandas.table(
            cellText=desc_pandas.values,
            rowLabels=desc_pandas.index,
            colLabels=desc_pandas.columns,
            loc='center'
        )
        table_pandas.auto_set_font_size(False)
        table_pandas.set_fontsize(10)
        plt.title("Descrierea Datelor (Pandas)", pad=20)
        plt.show()

        # NumPy Descriptive Statistics
        desc_numpy_df = pd.DataFrame(desc_numpy, index=numeric_columns).T.round(2)
        fig_numpy, ax_numpy = plt.subplots(figsize=(10, desc_numpy_df.shape[0]*0.6 + 2))
        ax_numpy.axis('tight')
        ax_numpy.axis('off')
        table_numpy = ax_numpy.table(
            cellText=desc_numpy_df.values,
            rowLabels=desc_numpy_df.index,
            colLabels=desc_numpy_df.columns,
            loc='center'
        )
        table_numpy.auto_set_font_size(False)
        table_numpy.set_fontsize(10)
        plt.title("Descrierea Datelor (NumPy)", pad=20)
        plt.show()

        # Array Descriptive Statistics
        desc_list_df = pd.DataFrame(desc_list, index=numeric_columns).T.round(2)
        fig_list, ax_list = plt.subplots(figsize=(10, desc_list_df.shape[0]*0.6 + 2))
        ax_list.axis('tight')
        ax_list.axis('off')
        table_list = ax_list.table(
            cellText=desc_list_df.values,
            rowLabels=desc_list_df.index,
            colLabels=desc_list_df.columns,
            loc='center'
        )
        table_list.auto_set_font_size(False)
        table_list.set_fontsize(10)
        plt.title("Descrierea Datelor (Array)", pad=20)
        plt.show()