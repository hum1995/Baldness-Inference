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
from statsmodels.tools.tools import add_constant

#%%
#open data
df = pd.read_csv('bald_probability.csv')


#%%

# Dropping nulls

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

#%%
# Changing column names

df.rename(columns = {'job_role':'job', 'is_married':'marital','is_smoker':'smoker', 'is_hereditary':'hereditary'}, inplace = True)
#%%
print(df.columns)
#%%
# Changing male and female to 0 and 1, respectively
df['gender'].replace('female', 0, inplace=True)
df['gender'].replace('male', 1, inplace=True)

# %%

# Categorizing stress to three categories: Low, Medium, High
df['stress'] = df['stress'].replace([1, 2, 3], 'Low')
df['stress'] = df['stress'].replace([4, 5, 6, 7], 'Medium')
df['stress'] = df['stress'].replace([8, 9, 10], 'High')

# %%
df['stress'].unique()
# %%
print(df.columns)


#%%[markdown]

# KDE Plot

#%%
sns.kdeplot(df['bald_prob'], shade=True)
plt.title('Baldness Probability KDE')
plt.xlabel("Baldness Probability")
plt.show()


#%%
#correlation
correlation_table = df.corr()
print("Correlation Table:")
print(correlation_table)

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_table, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()

#%%[markdown]
## Modeling
print(df)
# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

# Encoding categorical variables
cat_vars = ['job', 'province', 'shampoo', 'education', 'stress']
enc = OneHotEncoder(drop='first', sparse=False)
encoded_cat_vars = pd.DataFrame(enc.fit_transform(df[cat_vars]), columns=enc.get_feature_names_out(cat_vars))

# Concatenate the original dataframe with the encoded categorical variables
df_encoded = pd.concat([df.drop(cat_vars, axis=1), encoded_cat_vars], axis=1)

# %%
# Fit linear regression model
X = df_encoded.drop('bald_prob', axis=1)
y = df_encoded['bald_prob']

lr_model = LinearRegression()
lr_model.fit(X, y)

y_pred = lr_model.predict(X)
#%%
# linear regression model statistics
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)


print(f'Mean Squared Error: {mse:.4f}')
print(f'Root Mean Squared Error: {rmse:.4f}')
print(f'R-squared: {r2:.4f}')

feature_names = X.columns.tolist()

coefficients = lr_model.coef_
intercept = lr_model.intercept_

print("Detailed Regression Model:\n")
print(f"Intercept: {intercept:.4f}\n")
print("Feature Coefficients:")

for feature, coef in zip(feature_names, coefficients):
    print(f"{feature}: {coef:.4f}")

#%%
# fit linear regression model with all features
X_with_constant = add_constant(X)
model_with_province = sm.OLS(y, X_with_constant).fit()
# Drop the "province" feature
province_columns = [col for col in X.columns if col.startswith('province')]
X_no_province = X.drop(province_columns, axis=1)

# Fit the model without the "province" feature
X_no_province_with_constant = add_constant(X_no_province)
model_without_province = sm.OLS(y, X_no_province_with_constant).fit()

# print summary for both models
print("Model with 'province':")
print(model_with_province.summary())

print("\nModel without 'province':")
print(model_without_province.summary())

# %%[markdown]
# after removing "province", the model's adj.R^2 didn't change. 

#%%
# Drop the "marital" feature
X_no_marital = X_no_province.drop('marital', axis=1)

# Fit the model without the "marital" feature
X_no_marital_with_constant = add_constant(X_no_marital)
model_without_marital = sm.OLS(y, X_no_marital_with_constant).fit()

print("\nModel without 'marital':")
print(model_without_marital.summary())

# %%[markdown]
# After removing "marital", the model's adj.R^2 didn't change. 

#%%
# Drop the "weight" feature
X_no_weight = X_no_marital.drop('weight', axis=1)

# Fit the model without the "weight" feature
X_no_weight_with_constant = add_constant(X_no_weight)
model_without_weight = sm.OLS(y, X_no_weight_with_constant).fit()

print("\nModel without 'weight':")
print(model_without_weight.summary())

# %%[markdown]
# After removing "weight", the model's adj.R^2 didn't change. 

#%%
# Drop the "height" feature
X_no_height = X_no_weight.drop('height', axis=1)

# Fit the model without the "height" feature
X_no_height_with_constant = add_constant(X_no_height)
model_without_height = sm.OLS(y, X_no_height_with_constant).fit()

print("\nModel without 'height':")
print(model_without_height.summary())

# %%
# After removing "height", the model's adj.R^2 didn't change. 
#%%
# Drop the "education" feature
education_columns = [col for col in X_no_height.columns if col.startswith('education')]
X_no_education = X_no_height.drop(education_columns, axis=1)

# Fit the model without the "education" feature
X_no_education_with_constant = add_constant(X_no_education)
model_without_education = sm.OLS(y, X_no_education_with_constant).fit()

print("\nModel without 'education':")
print(model_without_education.summary())

# %%
# Drop the "shampoo" feature
shampoo_columns = [col for col in X_no_education.columns if col.startswith('shampoo')]
X_no_shampoo = X_no_education.drop(shampoo_columns, axis=1)

# Fit the model without the "shampoo" feature
X_no_shampoo_with_constant = add_constant(X_no_shampoo)
model_without_shampoo = sm.OLS(y, X_no_shampoo_with_constant).fit()

print("\nModel without 'shampoo':")
print(model_without_shampoo.summary())

# %%
# final OLS model
# Calculate the predictions
y_pred = model_without_shampoo.predict(X_no_shampoo_with_constant)

# Calculate the MSE and RMSE
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)

coefficients = model_without_shampoo.params
print("Summary Statistics:\n")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}\n")
print("Coefficients:")

for feature, coef in coefficients.items():
    print(f"{feature}: {coef:.4f}")

# %%[markdown]
# Random Forest Model

#%%
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
# Assuming 'df_encoded' is your encoded dataframe with all the features
X_all_features = df_encoded.drop('bald_prob', axis=1)
y_all_features = df_encoded['bald_prob']
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_all_features, y_all_features)
# Calculate the predictions
y_rf_pred = rf_model.predict(X_all_features)

# Calculate the MSE and RMSE
mse_rf = mean_squared_error(y_all_features, y_rf_pred)
rmse_rf = np.sqrt(mse_rf)
print("Random Forest Model Summary Statistics:\n")
print(f"MSE: {mse_rf:.4f}")
print(f"RMSE: {rmse_rf:.4f}\n")

# %%
feature_importances = rf_model.feature_importances_
importance_df = pd.DataFrame({'Feature': X_all_features.columns, 'Importance': feature_importances})
importance_df_sorted = importance_df.sort_values(by='Importance', ascending=False)

print("Feature Importance Ranking:\n")
print(importance_df_sorted)
plt.figure(figsize=(12, 8))
plt.bar(importance_df_sorted['Feature'], importance_df_sorted['Importance'])
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance Ranking')
plt.show()

# %%