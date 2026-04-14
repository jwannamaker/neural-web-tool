"""Example using PyTorch for predicting house prices."""
import pandas as pd

data_raw = pd.read_csv('./london_houses_transformed.csv')

# SEPARATE FEATURES FROM THE TARGET
x = data_raw.drop('price', axis=1)
y = data_raw['price']
