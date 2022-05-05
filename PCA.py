import pandas as pd
import pylab as pl
from sklearn import datasets
from sklearn.decomposition import PCA
# load dataset
wine = datasets.load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
# normalize data
from sklearn import preprocessing
data_scaled = pd.DataFrame(preprocessing.scale(df),columns = df.columns) 
print(data_scaled)
# PCA
K=2 
pca = PCA(n_components=K)
pca.fit_transform(data_scaled)
# Dump components relations with features:
print(pd.DataFrame(pca.components_,columns=data_scaled.columns,index = ['PC-'+ str(i) for i in range(1,K+1)] ))