# I didn't put much details here since I'm already familiar with sklearn
# and since sklearn's documentation is well written so it's easy to refer to it

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import learning_curve
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, max_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

df = pd.DataFrame()

# ----------------
# data preparation

## train test split

# Define X and y
X = df[["feature one", "feature two"]]
X = df.drop(columns=["useless feature one", "useless feature two"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

## impute

imputer = SimpleImputer(strategy="mean")
X_train = imputer.fit_transform(X_train)

imputer.statistics_ # The mean is stored in the transformer's memory

## binning

df['SalePriceBinary'] = pd.cut(x = df['SalePrice'],
                       bins=[df['SalePrice'].min()-1,
                             df['SalePrice'].mean(),
                             df['SalePrice'].max()+1],
                       labels=['cheap', 'expensive'])

## scale

scaler = StandardScaler()
scaler = RobustScaler()
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

## balance

# SMOTE

## encode target

target_encoder = LabelEncoder()
y_train_encoded = target_encoder.fit_transform(y_train)

## encode features

encoder = OneHotEncoder(drop='if_binary', sparse = False)
X_train_encoded = encoder.fit_transform(X_train_scaled)


# correlation heatmap

corr = df.corr()
sns.heatmap(corr,
        xticklabels=corr.columns,
        yticklabels=corr.columns,
        cmap= "YlGnBu")

# show strongest correlations
# example
corr_df = corr.unstack().reset_index() # Unstack correlation matrix
corr_df.columns = ['feature_1','feature_2', 'correlation'] # rename columns
corr_df.sort_values(by="correlation",ascending=False, inplace=True) # sort by correlation
corr_df = corr_df[corr_df['feature_1'] != corr_df['feature_2']] # Remove self correlation
corr_df.head()

# feature permutation
# example
log_model = LogisticRegression().fit(X, y) # Fit model
permutation_score = permutation_importance(log_model, X, y, n_repeats=10) # Perform Permutation
importance_df = pd.DataFrame(np.vstack((X.columns,
                                        permutation_score.importances_mean)).T) # Unstack results
importance_df.columns=['feature','score decrease']
importance_df.sort_values(by="score decrease", ascending = False) # Order by importance
importance_df.head()
importance_df.tail()
X = X.drop(columns=['Street', "Pave"]) # Drops weak features

# ---------
# baseline

# evaluate baseline
baseline_model = DummyRegressor(strategy="mean")
baseline_model.fit(X_train, y_train)
baseline_model.score(X_test, y_test)

# compare baseline to our model
model = LinearRegression()
model.fit(X_train, y_train)
model.score(X_test, y_test)

# ---------
# models

# -------------------------------
# without K-fold cross validation

# Import the model
from sklearn.linear_model import LinearRegression

# Instanciate the model
model = LinearRegression()

# Train the model on the data
model.fit(X_train, y_train)

# View the model's slope (a)
model.coef_
# View the model's intercept (b)
model.intercept_

# Evaluate the model's performance (R squared by default)
model.score(X_test,y_test)

#  Predict on new data
model.predict([[1000]])

# -----------------------------
# with K-fold cross validation

# Instantiate model
model = LinearRegression()

# 5-Fold Cross validate model
cv_results = cross_validate(model, X, y, cv=5) # usually cv=5 or cv=10

# Scores
cv_results['test_score']

# Mean of scores
cv_results['test_score'].mean()

# ---------------
# learning curves

train_sizes = [25,50,75,100,250,500,750,1000,1150]
# Get train scores (R2), train sizes, and validation scores using `learning_curve`
train_sizes, train_scores, test_scores = learning_curve(
    estimator=LinearRegression(), X=X, y=y, train_sizes=train_sizes, cv=5)
# Take the mean of cross-validated train scores and validation scores
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
plt.plot(train_sizes, train_scores_mean, label = 'Training score')
plt.plot(train_sizes, test_scores_mean, label = 'Test score')
plt.ylabel('r2 score', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title('Learning curves', fontsize = 18, y = 1.03)
plt.legend()
plt.show()

# ------------------
# evaluation

## evaluation without cross validation (linear metrics)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
mae = mean_absolute_error(y_test, y_pred)
rsquared = r2_score(y_test, y_pred)
me = max_error(y_test, y_pred)

## evaluation with cross validation

cv_results = cross_validate(model, X_train, y_train, cv=5,
                            scoring=['max_error',
                                     'r2',
                                     'neg_mean_absolute_error',
                                     'neg_mean_squared_error']
                           )

cv_results['test_r2'].mean()

## evaluation without cross validation (categorical metrics)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# precision_recall_curve, roc_curve, roc_auc_score : voir cours du 4 mai - Performance metrics

# ------------
# model tuning

# grid search

# Hyperparameter Grid
grid = {'alpha': [0.01, 0.1, 1],
        'l1_ratio': [0.2, 0.5, 0.8]}
# Instanciate Grid Search
search = GridSearchCV(model, grid,
                           scoring = 'r2',
                           cv = 5,
                           n_jobs=-1)
# Fit data to Grid Search
search.fit(X_train,y_train)
# Best score
search.best_score_
# Best Params
search.best_params_
# Best estimator
search.best_estimator_

# random search

# Hyperparameter Grid
grid = {'l1_ratio': stats.uniform(0, 1),
        'alpha': stats.loguniform(0.001, 1)}
# Instanciate Grid Search
search = RandomizedSearchCV(model, grid,
                            scoring='r2',
                            n_iter=100,  # number of draws
                            cv=5,
                            n_jobs=-1)
# Fit data to Grid Search
search.fit(X_train, y_train)
# Best estimator
search.best_estimator_

# how to use stats:
dist = stats.norm(10, 2) # if you have a best guess (say: 10)
dist = stats.randint(1, 100) # if you have no idea
dist = stats.uniform(1, 100) # same
dist = stats.loguniform(0.01, 1) # Coarse grain search
