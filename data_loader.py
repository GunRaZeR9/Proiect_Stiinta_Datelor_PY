import pandas as pd
from pathlib import Path
import numpy as np

class DataLoader:
    def __init__(self, data_path, nrows=None, random_state=42):
        self.data_path = Path(data_path)
        self.nrows = nrows
        self.random_state = random_state
        
    def load_csv(self, filename):
        file_path = self.data_path / filename
        try:
            total_rows = sum(1 for _ in open(file_path)) - 1
            
            if self.nrows:
                np.random.seed(self.random_state)
                skip_idx = sorted(np.random.choice(
                    range(1, total_rows + 1), 
                    total_rows - self.nrows, 
                    replace=False
                ))
                df = pd.read_csv(file_path, skiprows=skip_idx)
            else:
                df = pd.read_csv(file_path)
            
            print("\nData Preview (first 5 rows):")
            print(df.head())
            print("\nDetailed Data Description:")
            print(df.describe(include='all'))
            print(f"\nSuccessfully loaded data")
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