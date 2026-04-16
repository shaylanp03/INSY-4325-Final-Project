import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV

class RealEstateEngine:
    def __init__(self):
        self.model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=8, n_jobs=-1)
        
    def advanced_clean(self, df):
        # Professional cleaning: Vectorized outlier removal
        q_low = df["price"].quantile(0.01)
        q_hi  = df["price"].quantile(0.99)
        df = df[(df["price"] < q_hi) & (df["price"] > q_low)]
        
        # Feature Engineering
        df['price_per_sqft'] = df['price'] / df['house_size']
        return df.dropna()

    def train_optimized(self, X, y):
        # Advanced: Predicting log(price) for higher accuracy ($R^2$)
        y_log = np.log1p(y)
        self.model.fit(X, y_log)
        return self.model

    def query_chatbot(self, df, query):
        # A simple 'Text-to-Insight' engine
        query = query.lower()
        if "average price" in query:
            return f"The current market average is ${df['price'].mean():,.2f}."
        if "expensive" in query:
            top_city = df.groupby('city')['price'].mean().idxmax()
            return f"The most expensive area in this dataset is {top_city}."
        return "I'm analyzing the 500k listings. Ask me about average prices or market trends!"
