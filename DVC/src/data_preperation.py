import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
import os

df1 = pd.read_csv(r'D:\Collage\Sem-7\Data Science\Project\Makaan_Properties.csv')  
df = df1[['Size','No_of_BHK','Furnished','Property_type','City_name','Price']]

# Convert the Size and Price to numerical from object type
df['Size'] = df['Size'].str.replace('sq ft','',regex=False)
df['Size'] = df['Size'].str.replace(',','',regex=False)
df['Size'] = df['Size'].str.strip()
df['Size'] = pd.to_numeric(df['Size'])
df['Price'] = pd.to_numeric(df['Price'].str.replace(',', ''))

df['Price'] = df['Price']/100000
df['Size'] = df['Size']/1000

dict1 = {
    "Apartment" : 0,
    "Residential Plot" : 1,
    "Independent Floor" : 2,
    "Independent House" : 3,
    "Villa" : 4
}

dict2 = {
    "Mumbai" : 0,
    "Chennai" : 1,
    "Hyderabad" : 2,
    "Bangalore" : 3,
    "Lucknow" : 4,
    "Delhi" : 5,
    "Kolkata" : 6,
    "Ahmedabad" : 7
}

dict3 = {
    "Unfurnished" : 0,
    "Semi-Furnished" : 1,
    "Furnished" : 2,
}

df["Property_type"] = df["Property_type"].map(dict1)
df['City_name'] = df['City_name'].map(dict2)
df['Furnished'] = df['Furnished'].map(dict3)
df['No_of_BHK'] = df["No_of_BHK"].str.split(" ").str[0].astype(int)

data_path = os.path.join("src", "data", "raw")
if not os.path.exists(data_path):
    os.makedirs(data_path)
else:
    print(f"The directory '{data_path}' already exists.")

df.to_csv(data_path+r"\Processed_data.csv",index=False)