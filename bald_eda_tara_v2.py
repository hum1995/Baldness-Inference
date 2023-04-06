#%%
#import packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from bioinfokit.analys import stat


#%%
#open data
df = pd.read_csv('bald_probability.csv')

#%%[markdown]
## EDA (Exploratory Data Analysis) / Pre-processing

#%%
print(df.isna().sum()) # check for NA values
print(df.head()) # take a look at our dataset
print(df.info()) # columns, nulls, and data types
print(df.shape) 


#%% [markdown]
# Summary Statistics

#%%
print(df.describe())

#%%[markdown]
# Data Cleaning

#%%

# Dropping nulls

df.dropna(inplace=True)

#%%
# Changing column names

df.rename(columns = {'job_role':'job', 'is_married':'marital','is_smoker':'smoker', 'is_hereditary':'hereditary'}, inplace = True)
#%%
print(df.columns)
#%%
# Changing male and female to 0 and 1, respectively
df['gender'].replace('female', 0, inplace=True)
df['gender'].replace('male', 1, inplace=True)

print(df['gender'].unique())

print(df.head())

#%%

#Changing float columns (gender, martial, hereditary, smoker, stress, and age) to integers

df['gender'] = df['gender'].astype('int')
df['marital'] = df['marital'].astype('int')
df['hereditary'] = df['hereditary'].astype('int')
df['smoker'] = df['smoker'].astype('int')
df['stress'] = df['stress'].astype('int')
df['age'] = df['age'].astype('int')



#%%
summary_stats = df['stress'].describe()
print(summary_stats)
category_counts = df['stress'].value_counts()
print(category_counts)
category_counts = df['stress'].value_counts()

#%%
# Categorize "stress" column
category_counts.plot(kind='bar')
plt.xlabel('Stress Levels')
plt.ylabel('Frequency')
plt.title('Frequency of Stress Levels')
plt.xticks(rotation=270)
plt.show()
# %%

# Categorizing stress to three categories: Low, Medium, High
df['stress'] = df['stress'].replace([1, 2, 3], 'Low')
df['stress'] = df['stress'].replace([4, 5, 6, 7], 'Medium')
df['stress'] = df['stress'].replace([8, 9, 10], 'High')

# %%
df['stress'].unique()
# %%
df.columns

#%%[markdown]

## Plots

#%%[markdown]

# Histogram

#%%
df['bald_prob'].hist()
plt.title('Baldness Probability Histogram')
plt.xlabel("Baldness Probability")
plt.show()

#%%[markdown]

# KDE Plot

#%%
sns.kdeplot(df['bald_prob'], shade=True)
plt.title('Baldness Probability KDE')
plt.xlabel("Baldness Probability")
plt.show()

#%% [markdown]

# Skew and Kurtosis

print('The skew is:', round(df['bald_prob'].skew(), 4))
print('The kurtosis is:', round(df['bald_prob'].kurt(), 4))
print('Both values are close to zero, indicating a relatively normal distribution, which is seen with the histogram plot.')


#%%


#%%[markdown]

# Violin Plots: Categorical Variables vs Baldness Probability

#%%


sns.violinplot(x="gender", y="bald_prob", data=df, palette="Pastel1")
plt.title('Baldness Probability and Gender')
plt.xlabel("Gender")
plt.ylabel("Baldness Probability")
plt.show()

sns.violinplot(x="education", y="bald_prob", data=df, palette="Accent")
plt.title('Baldness Probability and Education Level')
plt.xlabel("Education Level")
plt.ylabel("Baldness Probability")
plt.show()

sns.violinplot(x="job", y="bald_prob", data=df, palette="PuBuGn")
plt.title('Baldness Probability and Job Role')
plt.xlabel("Job Role")
plt.ylabel("Baldness Probability")
plt.show()

sns.violinplot(x="marital", y="bald_prob", data=df, palette="Greens")
plt.title('Baldness Probability and Marital Status')
plt.xlabel("Marital Status")
plt.ylabel("Baldness Probability")
plt.show()

sns.violinplot(x="smoker", y="bald_prob", data=df, palette="Purples")
plt.title('Baldness Probability and Smoking')
plt.xlabel("Smoking")
plt.ylabel("Baldness Probability")
plt.show()

sns.violinplot(x="hereditary", y="bald_prob", data=df, palette="icefire")
plt.title('Baldness Probability and Hereditary')
plt.xlabel("Hereditary")
plt.ylabel("Baldness Probability")
plt.show()

sns.violinplot(x="shampoo", y="bald_prob", data=df, palette="spring")
plt.title('Baldness Probabilityand Shampoo Type')
plt.xlabel("Shampoo Type")
plt.ylabel("Baldness Probability")
plt.show()

sns.violinplot(x="stress", y="bald_prob", data=df, palette="coolwarm")
plt.title('Baldness Probability and Stress Level')
plt.xlabel("Stress Level")
plt.ylabel("Baldness Probability")
plt.show()


#%%[markdown]

# Scatterplots with Line of Best Fit: Continuous Variables vs Baldness Probability

#%%

sns.regplot(x= df['weight'], y= df['bald_prob'])
plt.title('Weight and Baldness Probability')
plt.xlabel('Weight')
plt.ylabel('Baldness Probability')
plt.show()

sns.regplot(x= df['height'], y= df['bald_prob'])
plt.title('Height and Baldness Probability')
plt.xlabel('Height')
plt.ylabel('Baldness Probability')
plt.show()

sns.regplot(x= df['salary'], y= df['bald_prob'])
plt.title('Salary and Baldness Probability')
plt.xlabel('Salary')
plt.ylabel('Baldness Probability')
plt.show()


#%%[markdown]
## Significance Testing

#%%



#%%[markdown]
## Modeling
