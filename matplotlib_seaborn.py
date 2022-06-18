import matplotlib.pyplot as plt
import seaborn as sns

# ----------
# matplotlib
# ----------

plt.figure(figsize=(10,5))

plt.title()
plt.xlabel()
plt.ylabel()
plt.xticks()
plt.yticks()
plt.xlim()
plt.ylim()
plt.grid()

plt.plot()

# legend
# First you need to add labels to each plot:
plt.plot(years_x, total_y, label="Total")
plt.plot(years_x, coal_y, label="Coal")
plt.plot(years_x, gas_y, label="Natural Gas")
# Then call:
plt.legend(loc="best")

# styles
print(sorted(plt.style.available))
plt.style.use('seaborn')

# using axes

plt.plot(years_x, total_y)
# Access the ax first
ax = plt.gca()
# then change its properties
ax.set_title('CO2 emissions from electricity production - US')
ax.set_ylabel('MtCO2/yr')
plt.show()

# subplots using axes

fig, axs = plt.subplots(1, 2, figsize=(10,3)) # axs is a (1,2) nd-array
# First subplot
axs[0].plot(years_x, coal_y, label="coal")
axs[0].plot(years_x, gas_y, label = "gas")
axs[0].set_title('coal vs. gas')
axs[0].legend()
# Second subplot
axs[1].plot(years_x, total_y, c='black')
axs[1].set_title('all energies')
# Global figure methods
plt.suptitle('US electricity CO2 emissions')
plt.show()

# using axes and pandas

import pandas as pd
ax = df.plot() # plot all columns against the index

# subplots using axes and pandas

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))
df1.plot(ax=ax1)
df2.plot(ax=ax2)

# other types of plots with plt

plt.plot()
plt.scatter()
plt.bar()
plt.hist()


# ----------
# seaborn
# ----------

# I didn't put much details here since I'm already familiar with seaborn

# subplots without using axes

plt.figure(figsize=(14, 4))
plt.subplot(1, 2, 1)
sns.histplot(...)
plt.subplot(1, 2, 2)
sns.histplot(...)

# other types of plots with sns

sns.histplot()
sns.countplot()
sns.catplot()
sns.regplot()
sns.pairplot()

sns.countplot(data=df, x="time", hue="smoker")
sns.catplot(data=df, x='day', y='total_bill' kind="box")

# subplots by feature

# Create a grid
g = sns.FacetGrid(df, col="time", row="smoker", hue="smoker")
# Plot a graph in each grid element
g.map(sns.histplot, "total_bill");

# cheat sheet:
# https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Python_Seaborn_Cheat_Sheet.pdf

# cheat sheet Le Wagon

# LINE PLOTS
plt.plot(x=df.col1, y=df.col2, c='red', ls='--', lw='0.5')
sns.lineplot(data=df, x='col1', y='col2', hue='col3', size='col4')
# DISTRIBUTIONS
plt.hist()
sns.histplot()
sns.kdeplot()
sns.jointplot()
# SCATTER PLOTS
plt.scatter()
sns.scatterplot()
sns.regplot()
# COUNT PLOTS
sns.countplot()
# CAT PLOTS
plt.bar() # eq. plt.plot(kind=‘bar’)
sns.barplot() # eq. catplot(kind=“bar”)
sns.violinplot() # eq. catplot(kind=“violin”)
sns.boxplot() # eq. catplot(kind=“box”)
# FACET GRID
g = sns.FacetGrid(data=df, col='col1')
g.map(plt.hist, 'col2')
# DATAFRAME-LEVEL MULTI CORRELATIONS
sns.heatmap(df.corr())
sns.pairplot(hue='')
## 2D HISTOGRAMS
plt.hist2d()
plt.colorbar()
sns.jointplot(x,y, kind='kde', data=df)
## 2D PROJECTION
plt.contour(X,Y,Z) # iso lines
plt.contourf(X,Y,Z=f(X,Y)) # area colors


# ----------
# plotly
# ----------

import plotly.express as px
# example:
px.scatter()
