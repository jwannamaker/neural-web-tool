import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data_raw = pd.read_csv('./london_houses_transformed.csv')

# SEPARATE FEATURES FROM THE TARGET
x = data_raw.drop('price', axis = 1)
y = data_raw['price']

        
