from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

data_path = os.path.join("src", "data", "raw")
df = pd.read_csv(data_path+r'\Processed_data.csv')

x = df.drop('Price', axis=1)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

all_results = {}

def store_results(model_name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)

    all_results[model_name] = {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2": float(r2)
    }

# Linear Model evaluation
data_path_linear = os.path.join("src", "data", "models", "LinearRegression")
for model in os.listdir(data_path_linear):
    if model.endswith('.pkl'):  
        file_path = os.path.join(data_path_linear,model)
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
            y_pred = model.predict(X_test)
            print("Evaluating Linear Model")
            store_results("LinearRegression", y_test, y_pred)

# RandomForest Model evaluation
data_path_RandomForest = os.path.join("src", "data", "models", "RandomForest")
for model in os.listdir(data_path_RandomForest):
    if model.endswith('.pkl'):  
        file_path = os.path.join(data_path_RandomForest,model)
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
            y_pred = model.predict(X_test)
            print("Evaluating RandomForest model")
            store_results("RandomForest", y_test, y_pred)

# xgboost Model evaluation
data_path_xgboost = os.path.join("src", "data", "models", "xgboost")
for model in os.listdir(data_path_xgboost):
    if model.endswith('.pkl'):  
        file_path = os.path.join(data_path_xgboost,model)
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
            y_pred = model.predict(X_test)
            print("Evaluating xgboost model")
            store_results("XGBoost", y_test, y_pred)

# Neural net Model evaluation
class NeuralNetwork(nn.Module):
    def __init__(self, input_size=X_train.shape[1]):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)  # Output layer for regression

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = torch.relu(self.layer4(x))
        return self.output(x)

# Load and evaluate the neural network model
data_path_nnModel = os.path.join("src", "data", "models", "nnModel")
for model_file in os.listdir(data_path_nnModel):
    if model_file.endswith('.pth'):
        file_path = os.path.join(data_path_nnModel, model_file)

        # Load the saved model state
        model = NeuralNetwork()
        model.load_state_dict(torch.load(file_path))
        model.eval()

        # Convert data to torch tensors
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

        # Make predictions
        with torch.no_grad():
            y_pred_tensor = model(X_test_tensor)

        # Convert predictions and ground truth to NumPy arrays
        y_pred = y_pred_tensor.numpy()
        y_test_np = y_test_tensor.numpy()

        # Evaluate model performance using scikit-learn metrics
        print("Evaluating Neural net model")
        store_results("NeuralNetwork", y_test, y_pred)

with open('metrics.json', 'w') as file:
    json.dump(all_results, file, indent=4)