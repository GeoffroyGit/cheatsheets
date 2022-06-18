# cheat sheet
# https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf

# I left this almost empty since I'm familiar enough with Pandas

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
