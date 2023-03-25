#%%
#import package
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#%%
#open data
df = pd.read_csv('bald_probability.csv')

#%%[markdown]
## EDA(Exploratory Data Analysis)

#%%
print(df.isna().sum())#check for NA values

print(df) # take a look at our dataset

#%%
print(df.describe())

#%%
print(df['job_role'].unique())
print(df['shampoo'].unique())

#%%[markdown]
## Data Preprocessing

#%%
#%%[markdown]
## Modeling

#%%

