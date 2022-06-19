# cheat sheet
# https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf

# I left this almost empty since I'm familiar enough with Pandas

import numpy as np
import pandas as pd

df = pd.DataFrame()

df.shape
df.head()
df.dtypes
df.info()
df.describe()

# count nulls
df.isnull().sum()

# quick way to use string functions
df['Region'].str.contains('Loire')

df["Region"].isin(["Bretagne", "Vendee"])

df["Region"].str.strip()

df["Region"].unique()

mask = df["Region"].isin(["Bretagne", "Vendee"])
sub_df = df[mask] # apply mask
sub_df = df[~mask] # apply opposite of mask

df.set_index('Country', inplace=True)

df.sort_index(ascending=False) # sort on index
df.sort_values(by='Population', ascending=False) # sort on values
df.sort_values(by='Population', na_position='first')

df.groupby('Region').sum()

# SQL with Pandas

df = pd.read_sql()

df.set_index() # permet de definir les index
df.index # permet d’accéder aux index

# apply, map, applymap...
df.applymap(lambda x: round(x))
df.groupby('col_name').agg({'toto':'mean'}) # apply the function "mean" to the column "toto"

# correlation matrix
df.corr()
round(df.corr(),2)
sns.heatmap(round(df.corr(),2), cmap = "coolwarm", annot = True, annot_kws = {"size":12})

df.dropna()

df.drop(columns=['toto', 'titi', 'tutu'])

df.duplicated().sum() # number of duplicated rows
df.drop_duplicates()

df.isnull().sum().sort_values(ascending=False)/len(df) # NaN percentage for each column

df["toto"].replace(np.nan, "NoData", inplace=True) # Replace NaN by "NoData"

df.dropna(subset=["toto"]) # Drop rows where toto value is missing

df["toto"].replace(np.nan, df["toto"].mean(), inplace=True) # Replace missing Pesos values with mean

# Greater than 30% of values missing: Potentially drop feature or row
# Less than 30% of values missing: impute a value that makes sense e.g. Median, mean, mode...

# but instead of all this, we can use sklearn Imputers (see sklearn.py)
