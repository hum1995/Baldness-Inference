#%%
#import packages
import copy as cp
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
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
# Random Forest Model by default

#%%
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

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

#%%
# Tune hyperparameters using GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_all_features, y_all_features, test_size=0.2, random_state=42)

# Define the hyperparameter search space
param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Initialize the GridSearchCV object
grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=param_grid, cv=5, n_jobs=-1)

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print(f"Best parameters found: {best_params}")

#%%
# no run
# Since some optimal hyperparameters' values hit the boundary, continue to try
# Define an extended hyperparameter search space
second_param_grid = {
    'n_estimators': [100, 150, 200, 250, 300],
    'max_depth': [None, 3, 5, 7, 9, 10],
    'min_samples_split': [1, 2, 3, 4, 5],
    'min_samples_leaf': [1, 2, 3],
    'bootstrap': [True]
}


# Initialize the GridSearchCV object with the extended search space
extended_grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=extended_param_grid, cv=5, n_jobs=-1)

# Fit the GridSearchCV object to the training data
extended_grid_search.fit(X_train, y_train)

# Get the best parameters from the extended search
best_params_extended = extended_grid_search.best_params_
print(f"Best parameters found in the extended search: {best_params_extended}")

#%%
# Retrain the Random Forest model with the optimal hyperparameters
optimal_rf_model = RandomForestRegressor(bootstrap=True, max_depth=10, min_samples_leaf=2, min_samples_split=2, n_estimators=200, random_state=42)
optimal_rf_model.fit(X_all_features, y_all_features)

# Calculate the feature importances
feature_importances_optimal = optimal_rf_model.feature_importances_
importance_df_optimal = pd.DataFrame({'Feature': X_all_features.columns, 'Importance': feature_importances_optimal})
importance_df_sorted_optimal = importance_df_optimal.sort_values(by='Importance', ascending=False)

# Plot the feature importance ranking
print("Feature Importance Ranking (Optimal Hyperparameters):\n")
print(importance_df_sorted_optimal)
plt.figure(figsize=(12, 8))
plt.bar(importance_df_sorted_optimal['Feature'], importance_df_sorted_optimal['Importance'])
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance Ranking (Optimal Hyperparameters)')
plt.show()

#%%
# Test overfitting
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_all_features, y_all_features, test_size=0.2, random_state=42)

# Train the model with the optimal hyperparameters
optimal_rf_model.fit(X_train, y_train)

# Make predictions on the training and test sets
y_train_pred = optimal_rf_model.predict(X_train)
y_test_pred = optimal_rf_model.predict(X_test)

# Calculate the MSE and R2 scores for the training and test sets
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

# Print the results
print("Training set: Mean squared error:", mse_train)
print("Training set: R2 score:", r2_train)
print("Test set: Mean squared error:", mse_test)
print("Test set: R2 score:", r2_test)

#%%
# XGBoost by default
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Split the dataset into training and testing sets
X = df_encoded.drop(columns=['bald_prob'])
y = df_encoded['bald_prob']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the XGBoost regression model
xg_reg = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1, max_depth=5, alpha=10, n_estimators=100)
xg_reg.fit(X_train, y_train)

# Predict "bald_prob" for the test set
y_pred = xg_reg.predict(X_test)

# Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean squared error: {mse}")
print(f"R2 score: {r2}")

# Extract the feature importance ranking
feature_importances = xg_reg.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
importance_df = importance_df.sort_values('Importance', ascending=False)
print(importance_df)

#%%
# Plot the feature importance ranking
fig, ax = plt.subplots(figsize=(12, 8))
importance_df.plot(kind='bar', x='Feature', y='Importance', legend=None, ax=ax)
plt.xticks(rotation=90)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importance Ranking")
plt.tight_layout()
plt.show()

#%%
# Hyperparameter tuning using randomized search
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, make_scorer

# Split the dataset into training and testing sets
X = df_encoded.drop(columns=['bald_prob'])
y = df_encoded['bald_prob']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create the XGBoost regressor
xg_reg = xgb.XGBRegressor(objective='reg:squarederror')

# Set up the hyperparameter search
param_dist = {
    'colsample_bytree': [0.3, 0.5, 0.7, 0.9],
    'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'alpha': [0, 1, 5, 10, 20],
    'n_estimators': [50, 100, 150, 200, 300],
}

scoring = {
    'MSE': make_scorer(mean_squared_error, greater_is_better=False),
    'R2': make_scorer(r2_score)
}

# Perform Randomized Search with cross-validation
random_search = RandomizedSearchCV(
    xg_reg, 
    param_distributions=param_dist, 
    n_iter=50, 
    scoring=scoring, 
    refit='R2', 
    cv=5,
    verbose=1,
    n_jobs=-1,
    random_state=0
)

random_search.fit(X_train, y_train)

# Print the best parameters
print("Best parameters found:")
print(random_search.best_params_)

# Train the XGBoost regression model with the best parameters
best_xg_reg = random_search.best_estimator_
best_xg_reg.fit(X_train, y_train)

# Predict "bald_prob" for the test set
y_pred = best_xg_reg.predict(X_test)

# Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean squared error: {mse}")
print(f"R2 score: {r2}")

# Extract the feature importance ranking
feature_importances = best_xg_reg.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
importance_df = importance_df.sort_values('Importance', ascending=False)
print(importance_df)

#%%
# After the optimal hyperparameter
# Train the XGBoost regression model with the best parameters
best_xg_reg = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=50,
    max_depth=4,
    learning_rate=0.2,
    colsample_bytree=0.7,
    alpha=1
)
best_xg_reg.fit(X_train, y_train)

# Extract the feature importance ranking
feature_importances = best_xg_reg.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
importance_df = importance_df.sort_values('Importance', ascending=False)

# Plot the feature importance ranking
fig, ax = plt.subplots(figsize=(12, 8))
importance_df.plot(kind='bar', x='Feature', y='Importance', legend=None, ax=ax)
plt.xticks(rotation=90)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importance Ranking")
plt.tight_layout()
plt.show()

#%%
# Detect overfitting
# Predict "bald_prob" for the training set
y_train_pred = best_xg_reg.predict(X_train)

# Evaluate the model performance on the training set
mse_train = mean_squared_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)

# Print the performance metrics for the training set
print(f"Training set: Mean squared error: {mse_train}")
print(f"Training set: R2 score: {r2_train}")

# Predict "bald_prob" for the test set
y_test_pred = best_xg_reg.predict(X_test)

# Evaluate the model performance on the test set
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

# Print the performance metrics for the test set
print(f"Test set: Mean squared error: {mse_test}")
print(f"Test set: R2 score: {r2_test}")

#%%[markdown]
# # Logistic Regression
# * Create a new column `bald_likelihod` indicating whether an individual is likely to develop baldness
# * The way we create this column is that `bald_likelihood = 1 if bald_prob > 0.59` 
# * Otherwise `bald_prob = 0`
# * Then, we will build a logistic regression to predict `bald_likelihood`

matplotlib.rcParams.update({'font.size': 16})

lklihood = lambda x: 1 if x > 0.59 else 0
df_encoded2 = cp.deepcopy(df_encoded)
df_encoded2['bald_likelihood'] = list(map(lklihood, df_encoded['bald_prob']))
print("The column baldness_likelihood has been created as desired")

#%%[markdown]
# * We plot a Histogram to see if the Baldness Likelihood is balanced in the dataset.

plt.figure(figsize=(9, 8))
plt.hist(df_encoded2['bald_likelihood'], bins=[0,0.9,1,2], align='left')
plt.xticks(range(2))
plt.xlabel("Baldness Likelihood")
plt.ylabel("Count")
plt.show()

print(f"Proportion of Baldness Likelihood 1: {len(df_encoded2[df_encoded2['bald_likelihood']==1])/len(df_encoded2):.2f}%")
print(f"Proportion of Baldness Likelihood 0: {len(df_encoded2[df_encoded2['bald_likelihood']==0])/len(df_encoded2):.2f}%")

#%%[markdown]
# * Yes, the data set is balanced. We have about the same number of people with Baldness Likelihood 1 and Baldness Likelihood 0


# %%
X = df_encoded2.drop(labels=['bald_prob', 'bald_likelihood'], axis=1)
y = df_encoded2['bald_likelihood']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#%%[markdown]
# ## Features Selection and Model Building
# * For feature selection, we will use the Recursive Feature Elimination `(RFE)` algorithm to select only important features.
# * We start with 10 features. 
# * If all the features have `P-value < 0.05`, we will try to increase the number of features.
# * If some features have `P-values > 0.05`, we remove them and the model is completed.

# use RFE to select features for the model
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

logreg = LogisticRegression()
rfe = RFE(logreg, n_features_to_select=10)
rfe = rfe.fit(X_train, y_train)

X_trans = X_train[rfe.get_feature_names_out()]
X_test_trans = X_test[rfe.get_feature_names_out()]


model= sm.Logit(y_train,X_trans)

result=model.fit()

print(result.summary2())
print("We drop the variable weight")
# %%
X_trans = X_trans.drop(labels=["weight"], axis = 1)
X_test_trans = X_test_trans.drop(labels=["weight"], axis = 1)
model= sm.Logit(y_train, X_trans)

result=model.fit()

print(result.summary2())

# %%[markdown]
from sklearn.metrics import (confusion_matrix, 
                           accuracy_score)
# Model Evaluation
# 1. Confusion Matrix

model = logreg.fit(X_trans, y_train)
y_pred = logreg.fit(X_trans, y_train).predict(X_test_trans)
cmatrix = confusion_matrix(y_pred, y_test) 
print(cmatrix)

#%%[markdown]

# 2. Scores:

# 1. $\text{Accuracy} = \frac{TP}{\text{Total numbre of predictions}}$ <br>
# The accuracy measures how many positive predictions are actually correct out of all predictions made. 

# 2. $\text{Precision} = \frac{TP}{FP + TP}$
# The precision is how many positive predictions are correct out of all positive predicitions made.
# A value close to `1` means not many False positive. A value close to `0` means a lot of False positive predictions.

# * $\text{Recall} = \frac{TP}{FN + TP}$
# The recall is out of all incorrect negative and correct positive predictions made, how many are correct positive predicitions?
# A value close to `1` means not many incorrect negative predictions. A value close to `0` means a lot of false negative.

from sklearn.metrics import accuracy_score, precision_score, recall_score
print(f"The accuracy score = {accuracy_score(y_test, y_pred):.2f}")
print(f"The precision score = {precision_score(y_test, y_pred):.2f}")
print(f"The recall score = {recall_score(y_test, y_pred):.2f}")

#%%[markdown]
# 2. The ROC Curve

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test_trans))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test_trans)[:,1])
plt.figure(figsize=(9, 8))
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression of Baldness Likelihood ROC Curve')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
#%%[markdown]
# ## Conclusion 
# The model does not perform well. It is indistinguishable from a model that assigns baldness likelihood 0 or 1 randomly.

# %%[markdown]
# # K-Means Clustering
# We observed that our logistic regression model performed poorly in predicting the `baldness likelihood`. 
# To understand why a simple logistic regression classification was not succesful, and to gain further insight into the relationship
# between the variables, we perform a `K-Means Clustering.`

# %%
from sklearn.cluster import KMeans
y = df_encoded2['bald_prob']

color_map = {0:'teal', 1:'blue', 2:'magenta'}

def make_clusters(col, nclusters):

    xdata = np.array(X[col])

    data = list(zip(xdata, y))
    kmeans = KMeans(n_clusters=nclusters)
    kmeans.fit(data)

    return kmeans.labels_

# determine the number of clusters with the Elbow method
def get_inertia_plots(col):
    inertias = []
    data = list(zip(X[col], y))
    for i in range(1,11):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

    plt.plot(range(1,11), inertias, marker='o')
    #plt.title('Elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()

# Run this cell only once. We don't need to run these codes every time since we know the results.

# get_inertia_plots('age')
# get_inertia_plots('salary')
print("Based on the Elbow methods, we choose 3 clusters for age and salary.")

#%%
# Get clusters for age and salary
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,6))
ax1.scatter(X['age'],y, c=[color_map[j] for j in make_clusters('age', 3)] )
ax2.scatter(X['salary'],y, c=[color_map[j] for j in make_clusters('salary', 3)])
ax1.set_xlabel("Age")
ax1.set_ylabel("Baldness Probabolity")
ax1.set_title("Bladness probability\n with age")
ax2.set_xlabel("Salary")
ax2.set_ylabel("Baldness Probabolity")
ax2.set_title("Bladness probability\n with salary")
plt.show()

#%%
# get clusters for weight and height
# get_inertia_plots('weight')
# get_inertia_plots('height')
print("Based on the Elbow methods, we choose 3 clusters for weight and height.")
# %%
col1 = 'weight'
col2 = 'height'
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,6))
ax1.scatter(X[col1],y, c=[color_map[j] for j in make_clusters(col1, 3)])
ax2.scatter(X[col2],y, c=[color_map[j] for j in make_clusters(col2, 3)])
ax1.set_xlabel(f"{col1}")
ax1.set_ylabel("Baldness Probabolity")
ax1.set_title(f"Bladness probability\n with {col1}")
ax2.set_xlabel(f"{col2}")
ax2.set_ylabel("Baldness Probabolity")
ax2.set_title(f"Bladness probability\n with {col2}")
plt.show()

# %%
df_encoded2.head()
# %%
