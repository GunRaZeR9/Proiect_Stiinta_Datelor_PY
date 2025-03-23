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

    def plot_data_description(self, df, numeric_columns):
        """
        Plots a table of the Pandas describe() output for the specified numeric columns.
        """
        # Compute the descriptive statistics.
        desc = df[numeric_columns].describe().round(2)
        
        # Create a figure and hide axes.
        fig, ax = plt.subplots(figsize=(10, desc.shape[0]*0.6 + 2))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table.
        table = ax.table(
            cellText=desc.values,
            rowLabels=desc.index,
            colLabels=desc.columns,
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        fig.tight_layout()
        plt.title("Data Description (Pandas)", pad=20)
        plt.show()