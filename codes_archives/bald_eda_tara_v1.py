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
print(df.head())

#%%
print(df.describe())

#%%

print(df.info())

#%%
print(df['job_role'].unique())
print(df['shampoo'].unique())

#%%[markdown]
## Data Preprocessing

#%%

# dropping nulls- Still vast majority of dataset
df.dropna(inplace=True)

#%%

# Changing male and female to 0 and 1, respectively

df['gender'].replace('female', 0, inplace=True)
df['gender'].replace('male', 1, inplace=True)

df['gender'].unique()

df.head()

#%%[markdown]

## Plots

#%%

# Histogram

df['bald_prob'].hist()
plt.title('Baldness Probability Histogram')
plt.xlabel("Baldness Probability")
plt.show()

#%%

# KDE Plot

sns.kdeplot(df['bald_prob'], shade=True)
plt.title('Baldness Probability KDE')
plt.xlabel("Baldness Probability")
plt.show()

#%%

sns.violinplot(x="gender", y="bald_prob", data=df, palette="Pastel1")
plt.title('Baldness Probability and Gender')
plt.xlabel("Baldness Probability")
plt.ylabel("Gender")
plt.show()

sns.violinplot(x="education", y="bald_prob", data=df, palette="Accent")
plt.title('Baldness Probability and Education Level')
plt.xlabel("Baldness Probability")
plt.ylabel("Education Level")
plt.show()

sns.violinplot(x="job_role", y="bald_prob", data=df, palette="PuBuGn")
plt.title('Baldness Probability and Job Role')
plt.xlabel("Baldness Probability")
plt.ylabel("Job Role")
plt.show()

sns.violinplot(x="is_married", y="bald_prob", data=df, palette="Greens")
plt.title('Baldness Probability and Marital Status')
plt.xlabel("Baldness Probability")
plt.ylabel("Marital Status")
plt.show()

sns.violinplot(x="is_smoker", y="bald_prob", data=df, palette="Purples")
plt.title('Baldness Probability and Smoking')
plt.xlabel("Baldness Probability")
plt.ylabel("Smoking")
plt.show()

sns.violinplot(x="is_hereditary", y="bald_prob", data=df, palette="icefire")
plt.title('Baldness Probability and Hereditary')
plt.xlabel("Baldness Probability")
plt.ylabel("Hereditary")
plt.show()

sns.violinplot(x="shampoo", y="bald_prob", data=df, palette="spring")
plt.title('Baldness Probabilityand Shampoo Type')
plt.xlabel("Baldness Probability")
plt.ylabel("Shampoo Type")
plt.show()

sns.violinplot(x="stress", y="bald_prob", data=df, palette="coolwarm")
plt.title('Baldness Probability and Stress Level')
plt.xlabel("Baldness Probability")
plt.ylabel("Stress Level")
plt.show()


#%%[markdown]
## Modeling

#%%

