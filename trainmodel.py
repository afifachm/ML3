import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('Retail_sales.csv')

# Select features and target
X = df[['Marketing Spend (USD)']].values
y = df['Units Sold'].values

# Fit linear regression
model = LinearRegression()
model.fit(X, y)

model_data = {
    'model': model,
    'X': X,
    'y': y
}

# Save the model as a PKL file
with open('linear_regression_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

