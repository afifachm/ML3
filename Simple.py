import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

# Read the dataset
df = pd.read_csv('Retail_sales.csv')

# Display basic info
print("First 5 rows:")
print(df.head())
print("\nInfo:")
print(df.info())
print("\nSummary statistics:")
print(df.describe())

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())
