import pandas as pd

from sklearn.decomposition import PCA


df = pd.DataFrame()

X = df[["feature one", "feature two"]]
X = df.drop(columns=["useless feature one", "useless feature two"])

# PCA

pca = PCA() # all components

pca.fit(X)

W = pca.components_.T

X_proj = pca.transform(X)

pca.explained_variance_ratio_

n = 3
pca = PCA(n_components=n) # n principal components

X_proj = pca.fit_transform(X)

X_reconstructed = pca.inverse_transform(X_proj)

# we can use PCA in a preprocessing pipeline
