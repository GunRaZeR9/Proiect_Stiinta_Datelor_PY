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
from performance_monitor import monitor_performance 


class Classifications:
    def __init__(self, df):
        self.df = df
        self.scaler = StandardScaler()
        self.df['event_time'] = pd.to_datetime(self.df['event_time'])
        self.df['price'] = pd.to_numeric(self.df['price'], errors='coerce')

    