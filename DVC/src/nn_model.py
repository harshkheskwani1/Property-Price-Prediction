import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import pickle as pkl
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

data_path = os.path.join("src", "data", "raw")
df = pd.read_csv(data_path+r'\Processed_data.csv')

x = df.drop('Price', axis=1)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

print(f"Training Data Shape: {X_train.shape}, {y_train.shape}")
print(f"Testing Data Shape: {X_test.shape}, {y_test.shape}")

X_train = torch.tensor(X_train.values, dtype=torch.float32)
X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
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

# Initialize the model
model = NeuralNetwork(input_size=X_train.shape[1])

# Use MSE loss for regression and Adam optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    
    # Forward pass
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

data_path2 = os.path.join("src", "data", "models", "nnModel")
if not os.path.exists(data_path2):
    os.makedirs(data_path2)
else:
    print(f"The directory '{data_path2}' already exists.")

torch.save(model.state_dict(), data_path2+r"\nn_model.pth")
print("Neural nets created and saved successfully")