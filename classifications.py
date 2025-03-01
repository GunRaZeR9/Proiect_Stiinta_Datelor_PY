import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

class Classifications:
    def __init__(self, df):
        self.df = df
        self.scaler = StandardScaler()
        self.df['event_time'] = pd.to_datetime(self.df['event_time'])
        
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