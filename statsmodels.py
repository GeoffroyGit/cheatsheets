import pandas as pd
import seaborn as sns

import statsmodels.api as sm
import statsmodels.formula.api as smf


df = pd.DataFrame()


# first, let's have a look at the data
sns.regplot(data=df.sample(100), y="review_score", x="wait_time")

# preprocessing
# at least, remove missing values

# use statsmodels for linear regression:
smf.ols(...)
# use statsmodels for logistic regaression:
smf.logit(...)

# instantiate a linear model
model = smf.ols(formula = 'review_score ~ wait_time + delay_vs_expected', data=df)

#instantiate a logistic model
model = smf.logit(formula='survived ~ fare + C(sex) + age', data=df)

# fit the model
model = model.fit()

# evaluate
model.params
model.rsquared
model.summary()

# check residuals
predicted = model.predict(df['horsepower'])
actual = df['weight']
residuals = predicted - actual

sns.histplot(x=residuals, bins=20)
sns.histplot(data=df, x="residuals", bins=20) # if we stored the residuals in df

# check that residuals are of equal variance
# plot the residuals and check visually
sns.scatterplot(x=predicted, y=residuals)

# RMSE (Root Mean Square Error)

RMSE = (residuals ** 2).mean() ** 0.5


# -----------------------------------------------------------------------------
# I can also use sklearn instead of statsmodels if I want to have a ML approach


# --------
# VIF
# (this is just a quick copy-paste from a notebook (may not work as is))

from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

vif_df = pd.DataFrame()
vif_df["features"] = df.columns
vif_df["VIF"] = [vif(df.values, i) for i in range(2, df.shape[1])]
round(vif_df.sort_values(by="VIF", ascending = False),2)
vif_df
