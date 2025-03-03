import time
import tracemalloc
import pandas as pd
import numpy as np
import csv

from sklearn.preprocessing import StandardScaler


class PerformanceComparison:
    def __init__(self, df):
        self.df = df
        self.scaler = StandardScaler()
        self.df['price'] = pd.to_datetime(self.df['price'])
        self.df_pandas = None
        self.data_numpy = None
        self.data_list = None

    def load_data_pandas(self):
        """Load data using Pandas and measure performance."""
        tracemalloc.start()
        start_time = time.time()

        # Load data and ensure the 'price' column is numeric
        self.df_pandas = pd.read_csv(self.file_path)
        self.df_pandas['price'] = pd.to_numeric(self.df_pandas['price'], errors='coerce')  # Convert to numeric

        elapsed_time = time.time() - start_time
        pandas_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        pandas_mem_usage = (pandas_mem[1] - pandas_mem[0]) / (1024 * 1024)

        print(f"Pandas: {elapsed_time:.4f} sec, {pandas_mem_usage:.4f} MB")

    def load_data_numpy(self):
        """Load data using NumPy and measure performance."""
        tracemalloc.start()
        start_time = time.time()

        with open(self.file_path, "r", encoding="utf-8") as f:
            header = next(f)
            self.data_numpy = np.fromiter(f, dtype="U100", count=-1)

        elapsed_time = time.time() - start_time
        numpy_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        numpy_mem_usage = (numpy_mem[1] - numpy_mem[0]) / (1024 * 1024)

        print(f"NumPy: {elapsed_time:.4f} sec, {numpy_mem_usage:.4f} MB")

    def load_data_list(self):
        """Load data using CSV reader and measure performance."""
        tracemalloc.start()
        start_time = time.time()

        with open(self.file_path, newline="", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            self.data_list = [row for row in reader]

        elapsed_time = time.time() - start_time
        list_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        list_mem_usage = (list_mem[1] - list_mem[0]) / (1024 * 1024)

        print(f"Array: {elapsed_time:.4f} sec, {list_mem_usage:.4f} MB")

    def describe_data(self, numerical_columns):
        """Describe numerical columns using different methods and measure performance."""
        results = {}

        # Pandas
        tracemalloc.start()
        start_time = time.time()
        desc_pandas = self.df_pandas[numerical_columns].describe()
        elapsed_time = time.time() - start_time
        pandas_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        pandas_mem_usage = (pandas_mem[1] - pandas_mem[0]) / (1024 * 1024)

        print(f"Pandas Describe: {elapsed_time:.4f} sec, {pandas_mem_usage:.4f} MB")
        results['Pandas'] = desc_pandas

        # NumPy
        tracemalloc.start()
        start_time = time.time()
        data_numpy = self.df_pandas[numerical_columns].to_numpy()
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
        elapsed_time = time.time() - start_time
        numpy_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        numpy_mem_usage = (numpy_mem[1] - numpy_mem[0]) / (1024 * 1024)

        print(f"NumPy Describe: {elapsed_time:.4f} sec, {numpy_mem_usage:.4f} MB")
        results['NumPy'] = desc_numpy
        tracemalloc.start()
        start_time = time.time()
        # Convert the data_list to numeric form for analysis
        data_list_numeric = [
            [float(row[i]) if row[i].replace(".", "", 1).isdigit() else float("nan") for row in self.data_list]
            for i in range(len(numerical_columns))
        ]

        desc_list = {
            "count": [len([x for x in col if not np.isnan(x)]) for col in data_list_numeric],
            "mean": [sum(col) / len(col) for col in data_list_numeric],
            "std": [np.std(col, ddof=0) for col in data_list_numeric],  # Population std deviation
            "min": [min(col) for col in data_list_numeric],
            "25%": [sorted(col)[int(len(col) * 0.25)] for col in data_list_numeric],
            "50%": [sorted(col)[int(len(col) * 0.50)] for col in data_list_numeric],
            "75%": [sorted(col)[int(len(col) * 0.75)] for col in data_list_numeric],
            "max": [max(col) for col in data_list_numeric],
        }

        elapsed_time = time.time() - start_time
        list_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        list_mem_usage = (list_mem[1] - list_mem[0]) / (1024 * 1024)

        print(f"Array Describe: {elapsed_time:.4f} sec, {list_mem_usage:.4f} MB")
        results['Array'] = desc_list

        return results

    def one_hot_encode(self):
        """Perform One-Hot Encoding using Pandas and measure performance."""
        categorical_columns = self.df_pandas.select_dtypes(include=[object]).columns.tolist()
        results = {}

        # One-Hot Encoding for Pandas
        tracemalloc.start()
        start_time = time.time()
        df_onehot = pd.get_dummies(self.df_pandas, columns=categorical_columns, drop_first=True)
        elapsed_time = time.time() - start_time
        pandas_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        pandas_mem_usage = (pandas_mem[1] - pandas_mem[0]) / (1024 * 1024)

        print(f"One-Hot Encoding Pandas: {elapsed_time:.4f} sec, {pandas_mem_usage:.4f} MB")
        results['Pandas'] = df_onehot.head()

        tracemalloc.start()
        start_time = time.time()

        # Ensure 'class' is one of the categorical columns, if applicable
        if 'class' in self.df_pandas.columns:
            class_numpy = self.df_pandas['class'].to_numpy().reshape(-1, 1)
        else:
            class_numpy = None

        unique_values = [np.unique(self.df_pandas[col]) for col in categorical_columns]
        encoded_numpy = np.hstack([
            (self.df_pandas[col].to_numpy()[:, None] == unique_values[i]).astype(int) for i, col in
            enumerate(categorical_columns)
        ])

        if class_numpy is not None:
            encoded_numpy = np.hstack((class_numpy, encoded_numpy))

        onehot_columns = [f"{categorical_columns[i]}_{val}" for i in range(len(categorical_columns)) for val in
                          unique_values[i]]

        elapsed_time = time.time() - start_time
        numpy_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        numpy_mem_usage = (numpy_mem[1] - numpy_mem[0]) / (1024 * 1024)

        print(f"One-Hot Encoding NumPy: {elapsed_time:.4f} sec, {numpy_mem_usage:.4f} MB")
        results['NumPy'] = pd.DataFrame(encoded_numpy, columns=onehot_columns).head()

        results['Array'] = None  # Will implement later if needed

        return results