from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import graphviz
from sklearn.tree import export_graphviz

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from xgboost import XGBRegressor, XGBClassifier

from sklearn.ensemble import VotingClassifier, StackingClassifier


# example of preprocessing (no need to scale when we use trees)
preproc_pipe = make_pipeline([
    SimpleImputer()
])

# simple tree for regression or classification

model = DecisionTreeRegressor()

model = DecisionTreeClassifier(max_depth=2)

pipe = make_pipeline(preproc_pipe, model)

# display tree

# Export model graph
export_graphviz(model, out_file="iris_tree.dot",
                feature_names=df.drop(columns=['target']).columns,
                class_names=['0','1','2'],
                rounded=True, filled=True)
# Import model graph
with open("iris_tree.dot") as f:
    dot_graph = f.read()
    display(graphviz.Source(dot_graph))

# predict

# predict the class
model.predict(...)
# Predict probability
model.predict_proba(...)

# random forest (bootstrap and bagging)

model = RandomForestRegressor(n_estimators=100)

model = RandomForestClassifier(max_depth=5)

# bagging KNNs

weak_learner = KNeighborsClassifier(n_neighbors=3)
model = BaggingClassifier(weak_learner, n_estimators=40)

weak_learner = KNeighborsRegressor(n_neighbors=3)
model = BaggingRegressor(weak_learner, n_estimators=50, oob_score=True)

# boosting

# adaptative boosting (AdaBoost)

model = AdaBoostRegressor(
    DecisionTreeRegressor(max_depth=3),
    n_estimators=50)

model = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=3),
    n_estimators=50)

# gradient boosting and extreme gradient boosting (XGBoost) are only implemented for trees

model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3
)

model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3
)

model = XGBRegressor(
    max_depth=10,
    n_estimators=100,
    learning_rate=0.1)

model = XGBClassifier(
    max_depth=10,
    n_estimators=100,
    learning_rate=0.1)

# stacking

model_forest = RandomForestClassifier()
model_logreg = LogisticRegression()
model_knn = KNeighborsClassifier(n_neighbors=10)

model = VotingClassifier(
    estimators = [("rf", model_forest),("lr", model_logreg)],
    voting = 'soft', # to use predict_proba of each classifier before voting
    weights = [1,1] # to equally weight forest and logreg in the vote
)

model = StackingClassifier(
    estimators = [("rf", model_forest),
                  ("knn", model_knn)],
    final_estimator = LogisticRegression() # here, voting is learned
)

# don't forget to use pipes
pipe = make_pipeline(preproc_pipe, model)
