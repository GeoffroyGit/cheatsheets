# I didn't put much details here since I'm already familiar with sklearn
# and since sklearn's documentation is well written so it's easy to refer to it

import numpy as np
import pandas as pd

from tempfile import mkdtemp
from shutil import rmtree

import pickle

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion, make_union
from sklearn.compose import make_column_selector
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

df = pd.DataFrame()

# train test split

X = df[["feature one", "feature two"]]
X = df.drop(columns=["useless feature one", "useless feature two"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# -----------------

# simple pipe to show how to use cache memory

# Create a temp folder
cachedir = mkdtemp()

# Instantiate the pipeline with cache parameter
pipe = make_pipeline([
    SimpleImputer(),
    StandardScaler()
    ],
    memory=cachedir)

# Clear the cache directory after the cross-validation
rmtree(cachedir)

# -----------------

# we can put custom functions in the pipe

# for example, a transformer that compresses data to 2 digits
custom_func = FunctionTransformer(lambda array: np.round(array, decimals=2))

# for example, a custom transformer that multiplies two columns
custom_func_2 = FunctionTransformer(lambda df: pd.DataFrame(df["bmi"] / df["age"]))

# the above examples only work for state-less transformations
# for state-full transformations, see class on Workflow on Kitt


# Impute then Scale for numerical variables

num_pipe = Pipeline([
    ('imputer', SimpleImputer()),
    ('scaler', StandardScaler()),
    ('custom_func', custom_func)
])

num_pipe = make_pipeline([
    SimpleImputer(),
    StandardScaler(),
    custom_func
])

# Encode categorical variables
cat_pipe = OneHotEncoder(handle_unknown='ignore')

# select features automatically based on dtype

num_cols = make_column_selector(dtype_include=['float64'])
cat_cols = make_column_selector(dtype_include=['object','bool'])

# Paralellize pipe for numerical variables and encoder for categorical variables

pipe = ColumnTransformer([
    ('num_tr', num_pipe, num_cols),
    ('cat_tr', cat_pipe, cat_cols)],
    remainder='passthrough')

pipe = make_column_transformer([
    (num_pipe, num_cols),
    (cat_pipe, cat_cols)],
    remainder='passthrough')

# feature union

pipe = FeatureUnion([
    ('preprocess', pipe),
    ('custom_func_2', custom_func_2)
])

pipe = make_union([
    pipe,
    custom_func_2
])

# add a model / an estimator to the pipe

model = Ridge() # for example Ridge

pipe = make_pipeline(pipe, model)

# Train pipeline
pipe.fit(X_train, y_train)

# Score model
pipe.score(X_test, y_test)
# with cross validation
cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2").mean()

# Make predictions
y_pred = pipe.predict(X_test)

# show feature names inside the pipe

pipe.get_feature_names_out()

# show hyper-paremeters inside the pipe

pipe.get_params()

# grid search

grid = {
        # for example
        'columntransformer__pipeline__simpleimputer__strategy': ['mean', 'median'],
        'ridge__alpha': [0.1, 0.5, 1, 5, 10]
        }

search = GridSearchCV(
    pipe,
    param_grid=grid,
    cv=5,
    scoring="r2")

search.fit(X_train, y_train)

search.best_params_

pipe_tuned = search.best_estimator_

# show the steps inside the pipe

# Access component of pipeline with name_steps
pipe_tuned.named_steps.keys()
# Check intermediate steps
# for example
pipe_tuned.named_steps["columntransformer"].fit_transform(X_train).shape

# save

# Export pipeline as pickle file
with open("pipeline.pkl", "wb") as file:
    pickle.dump(pipe_tuned, file)

# Load pipeline from pickle file
pipe = pickle.load(open("pipeline.pkl","rb"))
