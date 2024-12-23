from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import pickle as pkl

data_path = os.path.join("src", "data", "raw")
df = pd.read_csv(data_path+r'\Processed_data.csv')

x = df.drop('Price', axis=1)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

print(f"Training Data Shape: {X_train.shape}, {y_train.shape}")
print(f"Testing Data Shape: {X_test.shape}, {y_test.shape}")

data_path2 = os.path.join("src", "data", "models", "RandomForest")
if not os.path.exists(data_path2):
    os.makedirs(data_path2)
else:
    print(f"The directory '{data_path2}' already exists.")

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
pkl.dump(rf_model,open(data_path2+r"\rf_model.pkl","wb"))
print("Random forest model created and saved successfully")