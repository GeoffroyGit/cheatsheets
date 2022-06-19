# ---------------------------------------------------
# I can simply use sklearn instead of statsmodels :-)
# ---------------------------------------------------

# use statsmodels for linear regression:
smf.ols(...)
# use statsmodels for logistic regaression:
smf.logit(...)

import statsmodels.formula.api as smf

# instantiate a linear model
model = smf.ols(formula = 'weight ~ horsepower', data=df)

# fit the model
model = model.fit()

# evaluate

model.params

model.rsquared

model.summary()

predicted = model.predict(df['horsepower'])
actual = df['weight']
residuals = predicted - actual

# check that residuals are of equal variance
# plot the residuals and check visually
sns.scatterplot(x=predicted, y=residuals)

#instantiate a logistic model
model = smf.logit(formula='survived ~ fare + C(sex) + age', data=titanic_df)

# evaluate
# same as for linear models above
