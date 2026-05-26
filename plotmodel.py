import pickle
import numpy as np

import matplotlib.pyplot as plt

# Load the model and data
with open('linear_regression_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

# Assume model_data contains the model and the data
# Example: {'model': model, 'X': X, 'y': y}
model = model_data['model']
X = model_data['X']  # shape: (n_samples, 1)
y = model_data['y']  # shape: (n_samples,)

# Predict using the model
y_pred = model.predict(X)

# Plot scatter and regression line
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel('Marketing Spend (USD)')
plt.ylabel('Units Sold')
plt.title('Marketing Spend vs Units Sold')
plt.legend()
plt.savefig('plot.png')
