#%%
# Importing packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

#%%
# Reading the CSV file
df = pd.read_csv("C:/Users/hp/Desktop/Data Mining/Project/bald_probability.csv")

#%%
# Changing the column names
df.rename(columns = {'job_role':'job', 'is_married':'marital','is_smoker':'smoker', 'is_hereditary':'hereditary'}, inplace = True)
print(df.columns)

#%%
# Changing male and female to 0 and 1, respectively
df['gender'].replace('female', 0, inplace=True)
df['gender'].replace('male', 1, inplace=True)

print(df['gender'].unique())
print(df.head())

# %%
# QQ plot for 'age' column
sm.qqplot(df['age'], line='s')
plt.title("QQ Plot for 'age'")
plt.show()


#%%
# QQ plot for 'salary' column
sm.qqplot(df['salary'], line='s')
plt.title("QQ Plot for 'salary'")
plt.show()


#%%
# QQ plot for 'weight' column
sm.qqplot(df['weight'], line='s')
plt.title("QQ Plot for 'weight'")
plt.show()

#%%
# Generating QQ plot for 'height'
sm.qqplot(df['height'], line='s')
plt.title("QQ Plot for 'height'")
plt.show()

#%% 
# Generating QQ plot for 'bald_prob'
sm.qqplot(df['bald_prob'], line='s')
plt.title("QQ Plot for 'bald_prob'")
plt.show()

# %%
# Performing ANOVA
model = ols('salary ~ education', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

# Printing the ANOVA table
print("ANOVA Table:\n", anova_table)


# %%
# Performing ANOVA
provinces = df['province'].unique()
grouped_data = []
for province in provinces:
    # Filtering the data for current province and removing rows with missing age values
    data = df[(df['province'] == province) & (~df['age'].isna())]['age']
    if len(data) > 0:  # Checking if data has at least one non-missing value
        grouped_data.append(data)
f_stat, p_value = stats.f_oneway(*grouped_data)

print("One-way ANOVA Results:")
print("F-statistic: ", f_stat)
print("p-value: ", p_value)


# %%
# Performing ANOVA on 'age' based on 'gender' groups
grouped_data = df.groupby('gender')['age']
f_statistic, p_value = stats.f_oneway(*[grouped_data.get_group(x) for x in grouped_data.groups])
print("One-way ANOVA Results for 'age' based on 'gender' groups:")
print(f"F-statistic: {f_statistic:.4f}")
print(f"p-value: {p_value:.4f}")


# %%
# Performing ANOVA on 'salary' based on 'job' groups
grouped_data = df.groupby('job')['salary']
f_statistic, p_value = stats.f_oneway(*[grouped_data.get_group(x) for x in grouped_data.groups])
print("One-way ANOVA Results for 'salary' based on 'job' groups:")
print(f"F-statistic: {f_statistic:.4f}")
print(f"p-value: {p_value:.4f}")


# %%
#Performing ANOVA on 'weight' based on 'marital' groups
grouped_data = df.groupby('marital')['weight']
f_statistic, p_value = stats.f_oneway(*[grouped_data.get_group(x) for x in grouped_data.groups])
print("One-way ANOVA Results for 'weight' based on 'marital' groups:")
print(f"F-statistic: {f_statistic:.4f}")
print(f"p-value: {p_value:.4f}")


# %%
# Scatter plot of stress vs. gender
sns.scatterplot(x='gender', y='stress', data=df)
plt.xlabel('Gender')
plt.ylabel('Stress')
plt.title('Stress vs. Gender')
plt.show()


# %%
# Box plot of educational level vs. gender
sns.boxplot(x='gender', y='education', data=df)
plt.xlabel('Gender')
plt.ylabel('Educational Level')
plt.title('Educational Level vs. Gender')
plt.show()

# %%
# Filtering the dataframe to remove any missing values for stress, gender, and education
df_filtered = df.dropna(subset=['stress', 'gender', 'education'])

# Bar plot of stress vs. gender with hue for educational level
sns.barplot(x='gender', y='stress', hue='education', data=df_filtered)
plt.xlabel('Gender')
plt.ylabel('Stress')
plt.title('Stress vs. Gender with Barplot and Hue for Educational Level')
plt.show()

# %%
# Filtering the dataframe to remove any missing values for gender and education
df_filtered = df.dropna(subset=['gender', 'education'])

# Count plot of educational level vs. gender
sns.countplot(x='gender', hue='education', data=df_filtered)
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Educational Level vs. Gender with Countplot')
plt.show()

# %%
# Continuous variables for correlation
continuous_vars = ['age', 'salary', 'weight', 'height', 'stress', 'bald_prob']

# Calculating the correlation matrix
corr_matrix = df[continuous_vars].corr()

# Creating a heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Correlation Heatmap for Continuous Variables')
plt.show()


# %%
# Calculating the Z-score for 'age'
df['age_zscore'] = (df['age'] - df['age'].mean()) / df['age'].std()

# Defining a threshold for identifying outliers
zscore_threshold = 2  # or any other value you prefer

# Identifying outliers based on Z-score
outliers = df[df['age_zscore'] > zscore_threshold]

# Printing the identified outliers
print(outliers)
