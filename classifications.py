import os
from dotenv import load_dotenv
load_dotenv()  # This reads the .env file and loads the variables into os.environ

# You can verify the variable is loaded:
print("OPENBLAS_NUM_THREADS =", os.environ.get("OPENBLAS_NUM_THREADS"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from performance_monitor import monitor_performance, compare_array_operations

class Classifications:
    def parse_salary_range(self, range_str):
        """
        Parses a salary range string like "$59K-$99K" into an average numeric value.
        """
        if not isinstance(range_str, str):
            return None
        parts = range_str.split('-')
        values = []
        for part in parts:
            part = part.strip()
            multiplier = 1
            if 'K' in part.upper():
                multiplier = 1000
            # Remove '$' and 'K'
            num_str = part.replace('$', '').replace('K', '').strip()
            try:
                val = float(num_str) * multiplier
                values.append(val)
            except Exception:
                continue
        return np.mean(values) if values else None

    def __init__(self, df):
        # Map dataset columns to the ones used for plotting.
        self.df = df.copy()
        # Convert "Job Posting Date" to datetime (vectorized)
        if "Job Posting Date" in self.df.columns:
            self.df['job_posting_date'] = pd.to_datetime(self.df['Job Posting Date'])
        else:
            print("Column 'Job Posting Date' not found.")
        # Parse "Salary Range" into an average salary using our helper function.
        if "Salary Range" in self.df.columns:
            self.df['avg_salary'] = self.df['Salary Range'].apply(self.parse_salary_range)
        else:
            print("Column 'Salary Range' not found.")

    @monitor_performance
    def plot_event_time_trends(self):
        """
        Uses vectorized groupby to count job postings per month and plots a line chart.
        """
        compare_array_operations()
        if 'job_posting_date' not in self.df.columns:
            print("Cannot plot event time trends without 'Job Posting Date'.")
            return

        trends = self.df.groupby(self.df['job_posting_date'].dt.to_period('M')).size()
        trends.index = trends.index.to_timestamp()
        plt.figure(figsize=(10,6))
        plt.plot(trends.index, trends.values, marker='o')
        plt.title("Job Posting Trends Over Months (Linear)")
        plt.xlabel("Month")
        plt.ylabel("Number of Job Postings")
        plt.grid(True)
        plt.show()

    @monitor_performance
    def plot_job_posting_weekday_trends(self):
        """
        New hypothesis: Job postings follow a trend by weekday.
        Groups the job_posting_date by weekday and plots the total count.
        """
        compare_array_operations()
        if 'job_posting_date' not in self.df.columns:
            print("Cannot plot weekday trends without 'Job Posting Date'.")
            return

        # Create a weekday column
        self.df['Weekday'] = self.df['job_posting_date'].dt.day_name()
        weekday_data = self.df.groupby('Weekday').size()
        # Ensure weekdays are in order
        order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_data = weekday_data.reindex(order)
        plt.figure(figsize=(10,6))
        plt.plot(weekday_data.index, weekday_data.values, marker='o')
        plt.title("Job Postings by Weekday (Linear)")
        plt.xlabel("Weekday")
        plt.ylabel("Number of Job Postings")
        plt.grid(True)
        plt.show()

    @monitor_performance
    def plot_qualifications_skills_match(self):
        """
        New hypothesis: Specific qualifications are more frequently required.
        Plots the frequency of the top 10 qualifications from 'Qualifications'.
        """
        compare_array_operations()
        if 'Qualifications' not in self.df.columns:
            print("Column 'Qualifications' not found in the dataframe.")
            return

        qual_series = self.df['Qualifications'].dropna().str.split(',')
        qual_exploded = qual_series.explode().str.strip()
        top_quals = qual_exploded.value_counts().head(10)
        
        if top_quals.empty:
            print("No valid qualification data to display.")
            return

        plt.figure(figsize=(10,6))
        plt.plot(top_quals.index, top_quals.values, marker='o')
        plt.title("Top 10 Qualifications Frequency (Linear)")
        plt.xlabel("Qualification")
        plt.ylabel("Frequency")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show()

    @monitor_performance
    def cluster_salary_by_geolocation(self, n_clusters=4):
        """
        Hypothesis: There exist distinct clusters in salary distributions based on geolocation.
        This method computes an average salary from 'Salary Range', then clusters job postings
        using 'latitude', 'longitude', and the computed average salary. It scales the features
        to avoid domination by salary values, then plots the job posting locations with their
        assigned cluster and marks the cluster centroids (in original scale).
        """
        compare_array_operations()

        # If 'avg_salary' doesn't exist but 'Salary Range' is present, compute it.
        if 'avg_salary' not in self.df.columns and 'Salary Range' in self.df.columns:
            self.df['avg_salary'] = self.df['Salary Range'].apply(self.parse_salary_range)

        required_columns = ['latitude', 'longitude', 'avg_salary']
        missing_cols = [col for col in required_columns if col not in self.df.columns]
        if missing_cols:
            print(f"Required column(s) {missing_cols} not found in the dataframe.")
            return

        data = self.df.dropna(subset=required_columns).copy()
        if data.empty:
            print("No valid data available for clustering.")
            return

        # Prepare the features for clustering.
        features = data[['latitude', 'longitude', 'avg_salary']].values

        # Scale the features to avoid domination by avg_salary.
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        try:
            from sklearn.cluster import KMeans
        except ImportError:
            print("scikit-learn is required for clustering. Please install it and try again.")
            return

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features_scaled)
        data['Cluster'] = clusters

        # Get cluster centers in scaled space and inverse-transform to original scale.
        centers_scaled = kmeans.cluster_centers_
        centers_orig = scaler.inverse_transform(centers_scaled)
        centers_lat = centers_orig[:, 0]
        centers_lon = centers_orig[:, 1]

        plt.figure(figsize=(10,6))
        # Plot the job posting locations.
        scatter = plt.scatter(data['longitude'], data['latitude'], c=data['Cluster'],
                              cmap='viridis', marker='o', s=50, alpha=0.7)
        plt.title("Salary by Geolocation Clusters")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.colorbar(scatter, label="Cluster")
        plt.grid(True)

        # Mark and annotate cluster centroids.
        plt.scatter(centers_lon, centers_lat, c='red', marker='*', s=200, label='Centroids')
        for idx, (lon, lat) in enumerate(zip(centers_lon, centers_lat)):
            plt.annotate(f'Cluster {idx}', (lon, lat), textcoords="offset points", xytext=(0,10),
                         ha='center', color='red')
        plt.legend()
        plt.show()