import time
import tracemalloc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D


class Classifications:
    def __init__(self, df):
        self.df = df
        self.scaler = StandardScaler()
        self.df['event_time'] = pd.to_datetime(self.df['event_time'])
        self.df['price'] = pd.to_numeric(self.df['price'], errors='coerce')
    def analyze_user_behavior(self):

        user_stats = self.df.groupby('user_id').agg({
            'event_type': 'count',
            'product_id': 'nunique',
            'price': ['mean', 'sum']
        }).reset_index()
        return user_stats
    
    def categorize_price_segments(self):

        self.df['price_category'] = pd.qcut(self.df['price'], 
                                          q=4, 
                                          labels=['Budget', 'Medium', 'High', 'Premium'])
        return self.df['price_category'].value_counts()
    
    def identify_popular_categories(self):

        return self.df['category_code'].value_counts().head(10)
    
    def analyze_brand_engagement(self):

        brand_metrics = self.df.groupby('brand').agg({
            'event_type': 'count',
            'user_id': 'nunique',
            'price': 'mean'
        }).reset_index()
        return brand_metrics.sort_values('event_type', ascending=False).head(10)
    
    def cluster_users_by_behavior(self):

        user_features = self.df.groupby('user_id').agg({
            'event_type': 'count',
            'price': ['mean', 'sum'],
            'product_id': 'nunique'
        }).fillna(0)
        
        scaled_features = self.scaler.fit_transform(user_features)
        kmeans = KMeans(n_clusters=4, random_state=42)
        clusters = kmeans.fit_predict(scaled_features)
        
        return clusters, silhouette_score(scaled_features, clusters)
    
    def session_duration_analysis(self):

        session_data = self.df.groupby('user_session').agg({
            'event_time': lambda x: (max(x) - min(x)).total_seconds()
        })
        session_data['duration_minutes'] = session_data['event_time'] / 3600
        return session_data.describe()
    
    def analyze_time_patterns(self):

        self.df['hour'] = pd.to_datetime(self.df['event_time']).dt.hour
        return self.df.groupby('hour')['event_type'].count()
    
    def category_price_analysis(self):

        return self.df.groupby('category_code')['price'].agg(['mean', 'min', 'max', 'std'])
    
    def user_loyalty_classification(self):

        user_activity = self.df.groupby('user_id').agg({
            'event_time': lambda x: (x.max() - x.min()).days,
            'event_type': 'count'
        })
        
        user_activity['loyalty_score'] = (
            0.7 * (user_activity['event_time'] / user_activity['event_time'].max()) +
            0.3 * (user_activity['event_type'] / user_activity['event_type'].max())
        )
        
        user_activity['loyalty_category'] = pd.qcut(
            user_activity['loyalty_score'],
            q=3,
            labels=['Low', 'Medium', 'High']
        )
        
        return user_activity
    
    def plot_results(self, plot_type):

        plt.figure(figsize=(12, 6))
        
        if plot_type == 'price_segments':
            results = self.categorize_price_segments()
            results.plot(kind='bar')
            plt.title('Product Price Segments Distribution')
            
        elif plot_type == 'time_patterns':
            results = self.analyze_time_patterns()
            plt.plot(results.index, results.values)
            plt.title('Event Distribution by Hour')
            plt.xlabel('Hour of Day')
            plt.ylabel('Number of Events')
            
        elif plot_type == 'popular_categories':
            results = self.identify_popular_categories()
            results.plot(kind='barh')
            plt.title('Top 10 Popular Categories')
            
        elif plot_type == 'brand_engagement':
            results = self.analyze_brand_engagement()
            sns.barplot(data=results.head(10), x='event_type', y='brand')
            plt.title('Top 10 Brands by Event Count')
            
        elif plot_type == 'price_distribution':
            sns.boxplot(data=self.df, x='price_category', y='price')
            plt.title('Price Distribution by Segment')
            plt.yscale('log')
            
        elif plot_type == 'event_types':
            results = self.event_type_analysis()
            sns.barplot(x=results.index, y=results.values)
            plt.title('Distribution of Event Types')
            plt.xticks(rotation=45)
            
        elif plot_type == 'hourly_sales':
            results = self.hourly_sales_analysis()
            plt.plot(results.index, results['price'], marker='o')
            plt.title('Hourly Sales Distribution')
            plt.xlabel('Hour of Day')
            plt.ylabel('Total Sales Value')
            
        plt.tight_layout()
        plt.show()
    
    def event_type_analysis(self):
        """Analyze distribution of different event types"""
        event_dist = self.df['event_type'].value_counts()
        return event_dist

    def user_session_analysis(self):
        """Analyze user session patterns"""
        session_stats = self.df.groupby('user_session').agg({
            'user_id': 'first',
            'event_type': 'count',
            'price': 'sum',
            'product_id': 'nunique'
        }).reset_index()
        return session_stats

    def product_popularity_analysis(self):
        """Analyze product popularity"""
        product_stats = self.df.groupby('product_id').agg({
            'event_type': 'count',
            'user_id': 'nunique',
            'price': 'mean'
        }).sort_values('event_type', ascending=False)
        return product_stats.head(20)

    def hourly_sales_analysis(self):
        """Analyze sales patterns by hour"""
        purchase_data = self.df[self.df['event_type'] == 'purchase']
        purchase_data['hour'] = purchase_data['event_time'].dt.hour
        hourly_sales = purchase_data.groupby('hour').agg({
            'price': 'sum',
            'product_id': 'count'
        })
        return hourly_sales

    def monitor_memory_usage(self, method_to_run, *args, **kwargs):
        """Monitor memory usage of a specific method."""
        tracemalloc.start()
        result = method_to_run(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        current_mb = current / 10 ** 6
        peak_mb = peak / 10 ** 6

        print(f"Current memory usage: {current_mb:.2f} MB; Peak: {peak_mb:.2f} MB")

        return result

    """By Price"""

    def load_data_pandas(self):
        """Load data using Pandas and measure performance."""
        tracemalloc.start()
        start_time = time.time()
        self.df['price'] = pd.to_numeric(self.df['price'], errors='coerce')

        elapsed_time = time.time() - start_time
        pandas_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        pandas_mem_usage = (pandas_mem[1] - pandas_mem[0]) / (1024 * 1024)

        print(f"Pandas Load: {elapsed_time:.4f} sec, {pandas_mem_usage:.4f} MB")

    def load_data_numpy(self):
        """Load data using NumPy and measure performance."""
        tracemalloc.start()
        start_time = time.time()
        elapsed_time = time.time() - start_time
        numpy_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        numpy_mem_usage = (numpy_mem[1] - numpy_mem[0]) / (1024 * 1024)

        print(f"NumPy Load: {elapsed_time:.4f} sec, {numpy_mem_usage:.4f} MB")

    def load_data_list(self):
        """Load data using CSV reader and measure performance."""
        tracemalloc.start()
        start_time = time.time()

        elapsed_time = time.time() - start_time
        list_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        list_mem_usage = (list_mem[1] - list_mem[0]) / (1024 * 1024)

        print(f"Array Load: {elapsed_time:.4f} sec, {list_mem_usage:.4f} MB")

    def analyze_price_statistics(self):
        """Calculate basic statistics for the price column."""
        if 'price' in self.df.columns:
            price_stats = {
                "mean": self.df['price'].mean(),
                "std_dev": self.df['price'].std(),
                "min": self.df['price'].min(),
                "max": self.df['price'].max(),
                "count": self.df['price'].count(),
                "missing_values": self.df['price'].isnull().sum()
            }
            return price_stats
        else:
            print("Price column not found.")
            return None

    def describe_data(self):
        """Describe numerical columns using different methods and measure performance."""
        results = {}
        # Pandas
        tracemalloc.start()
        start_time = time.time()
        desc_pandas = self.df[self.df['price']].describe()
        elapsed_time = time.time() - start_time
        pandas_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        pandas_mem_usage = (pandas_mem[1] - pandas_mem[0]) / (1024 * 1024)

        print(f"Pandas Describe: {elapsed_time:.4f} sec, {pandas_mem_usage:.4f} MB")
        results['Pandas'] = desc_pandas

        # NumPy
        tracemalloc.start()
        start_time = time.time()
        data_numpy = self.df[self.df['price']].to_numpy()
        desc_numpy = {
            "count": np.count_nonzero(~np.isnan(data_numpy), axis=0),
            "mean": np.nanmean(data_numpy, axis=0),
            "std": np.nanstd(data_numpy,  axis=0),
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

        # List
        tracemalloc.start()
        start_time = time.time()
        data_list_numeric = [
            [float(row[i]) if row[i].replace(".", "", 1).isdigit() else float("nan") for row in self.data_list]
            for i in range(len(self.df['price']))
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
        categorical_columns = self.df.select_dtypes(include=[object]).columns.tolist()
        results = {}

        # One-Hot Encoding for Pandas
        tracemalloc.start()
        start_time = time.time()
        df_onehot = pd.get_dummies(self.df, columns=categorical_columns, drop_first=True)
        elapsed_time = time.time() - start_time
        pandas_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        pandas_mem_usage = (pandas_mem[1] - pandas_mem[0]) / (1024 * 1024)

        print(f"One-Hot Encoding Pandas: {elapsed_time:.4f} sec, {pandas_mem_usage:.4f} MB")
        results['Pandas'] = df_onehot.head()

        return results

    def calculate_price_statistics(self):
        """Calculate basic statistics of the price column"""
        mean_price = np.mean(self.df['price'])
        median_price = np.median(self.df['price'])
        std_price = np.std(self.df['price'])
        return {
            'mean': mean_price,
            'median': median_price,
            'std_dev': std_price
        }

    def detect_price_outliers(self):
        """Identify price outliers using the IQR method"""
        Q1 = np.percentile(self.df['price'], 25)
        Q3 = np.percentile(self.df['price'], 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = self.df[(self.df['price'] < lower_bound) | (self.df['price'] > upper_bound)]
        return outliers[['product_id', 'price']]

    def price_correlation_analysis(self):
        """Analyze correlation between price and other numeric variables"""
        numeric_df = self.df.select_dtypes(include=['float64', 'int64'])

        if 'price' in numeric_df.columns:
            correlation_matrix = numeric_df.corr()
            return correlation_matrix['price'].sort_values(ascending=False)
        else:
            return "Price column is not present in the numeric data."

    def price_based_clustering(self):
        """Cluster products based on price using KMeans"""
        price_data = self.df[['price']].fillna(0)
        kmeans = KMeans(n_clusters=3, random_state=42)
        price_clusters = kmeans.fit_predict(price_data)
        self.df['price_cluster'] = price_clusters
        return self.df[['product_id', 'price', 'price_cluster']]

    def visualize_price_trends(self):
        """Visualize trends of average price over time with individual points."""

        plt.figure(figsize=(12, 6))

        all_months = self.df['event_time'].dt.to_period("M").unique()

        for month in all_months:
            monthly_prices = self.df[self.df['event_time'].dt.to_period("M") == month]['price']

            print(f"{month}: {monthly_prices.tolist()}")

            if not monthly_prices.empty:
                month_timestamp = month.to_timestamp()

                jittered_x_values = [month_timestamp + pd.Timedelta(days=i) for i in range(len(monthly_prices))]
                plt.scatter(jittered_x_values, monthly_prices, color='blue', alpha=0.5)

                #mean price for this month
                mean_price = monthly_prices.mean()
                #mean price slightly for visibility
                plt.scatter(month_timestamp, mean_price + 10, color='red', marker='x', s=100)  # Red X for mean price

        #overall average price trend for each month
        price_trends = self.df.groupby(self.df['event_time'].dt.to_period("M"))['price'].mean().reset_index()
        price_trends['event_time'] = price_trends['event_time'].dt.to_timestamp()  # Convert Period to Timestamp

        print("Average price trends:\n", price_trends)

        plt.plot(price_trends['event_time'], price_trends['price'], marker='o',
                 linestyle='-', label='Average Price', color='green')

        plt.title('Average Price Trends Over Time')
        plt.xlabel('Month')
        plt.ylabel('Average Price')
        plt.xticks(rotation=45)
        plt.grid()
        plt.tight_layout()
        plt.legend()
        plt.show()

    def analyze_price_with_pandas(self):
        """Analyze price column using Pandas."""
        start_time = time.time()
        desc_pandas = self.df['price'].describe()
        pandas_time = time.time() - start_time

        return desc_pandas, pandas_time

    def analyze_price_with_numpy(self):
        """Analyze price column using NumPy."""
        start_time = time.time()
        data_numpy = self.df['price'].to_numpy()

        desc_numpy = {
            "count": np.count_nonzero(~np.isnan(data_numpy)),
            "mean": np.nanmean(data_numpy),
            "std": np.nanstd(data_numpy),
            "min": np.nanmin(data_numpy),
            "25%": np.nanpercentile(data_numpy, 25),
            "50%": np.nanpercentile(data_numpy, 50),
            "75%": np.nanpercentile(data_numpy, 75),
            "max": np.nanmax(data_numpy),
        }

        numpy_time = time.time() - start_time
        return desc_numpy, numpy_time

    def analyze_price_with_list(self):
        """Analyze price column using standard Python lists."""
        start_time = time.time()

        # Convert price column to a list of floats, handling NaN values correctly
        data_list_numeric = []

        for x in self.df['price']:
            if isinstance(x, (int, float)):
                data_list_numeric.append(float(x))
            else:
                data_list_numeric.append(float("nan"))  # Append NaN for non-numeric entries

        # Calculate descriptive statistics
        count = len([x for x in data_list_numeric if not np.isnan(x)])
        mean = sum(data_list_numeric) / count if count > 0 else float("nan")
        std = (sum(
            (x - mean) ** 2 for x in data_list_numeric if not np.isnan(x)) / count) ** 0.5 if count > 0 else float(
            "nan")
        min_val = min(x for x in data_list_numeric if not np.isnan(x))
        percentiles = {
            "25%": sorted(data_list_numeric)[int(len(data_list_numeric) * 0.25)],
            "50%": sorted(data_list_numeric)[int(len(data_list_numeric) * 0.50)],
            "75%": sorted(data_list_numeric)[int(len(data_list_numeric) * 0.75)],
        }
        max_val = max(x for x in data_list_numeric if not np.isnan(x))

        desc_list = {
            "count": count,
            "mean": mean,
            "std": std,
            "min": min_val,
            **percentiles,
            "max": max_val,
        }

        list_time = time.time() - start_time
        return desc_list, list_time

    def compare_price_analysis(self):
        """Compare different methods for analyzing the price column."""
        desc_pandas, pandas_time = self.analyze_price_with_pandas()
        desc_numpy, numpy_time = self.analyze_price_with_numpy()
        desc_list, list_time = self.analyze_price_with_list()

        results = pd.DataFrame({
            "Method": ["Pandas", "NumPy", "List"],
            "Execution Time (sec)": [pandas_time, numpy_time, list_time]
        })

        print("Performance Comparison:")
        print(results)

        print("\nPandas Description:")
        print(desc_pandas)

        print("\nNumPy Description:")
        for key, value in desc_numpy.items():
            print(f"{key}: {value}")

        print("\nList Description:")
        for key, value in desc_list.items():
            print(f"{key}: {value}")

    def visualize_price_trends_3d(self):
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111, projection='3d')

        all_months = self.df['event_time'].dt.to_period("M").unique()

        for month_index, month in enumerate(all_months):
            monthly_prices = self.df[self.df['event_time'].dt.to_period("M") == month]['price']

            print(f"{month}: {monthly_prices.tolist()}")

            if not monthly_prices.empty:
                month_timestamp = month.to_timestamp()


                jittered_x_values = [
                    month_timestamp.toordinal() + np.random.uniform(-0.1, 0.1) for _ in range(len(monthly_prices))
                ]

                # Plot individual prices in 3D
                ax.scatter(jittered_x_values, monthly_prices, zs=0, zdir='y', color='blue', alpha=0.5)

                # Calculate the mean price for this month
                mean_price = monthly_prices.mean()

                ax.scatter(month_timestamp.toordinal(), mean_price, zs=month_index * 0.2, zdir='y', color='red',
                           marker='x', s=100)

        price_trends = self.df.groupby(self.df['event_time'].dt.to_period("M"))['price'].mean().reset_index()
        price_trends['event_time'] = price_trends['event_time'].dt.to_timestamp()

        print("Average price trends:\n", price_trends)


        ax.plot(price_trends['event_time'].apply(lambda x: x.toordinal()),
                price_trends['price'],
                zs=0, zdir='y',
                marker='o', linestyle='-', color='green', label='Average Price')

        ax.set_title('Average Price Trends Over Time (3D)')
        ax.set_xlabel('Month')
        ax.set_ylabel('Average Price / Individual Prices')
        ax.set_zlabel('Mean Price')
        ax.view_init(elev=20, azim=30)
        plt.xticks(rotation=45)
        plt.grid()
        plt.tight_layout()
        plt.legend()
        plt.show()



