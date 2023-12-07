# tenho um csv com 8 classes de atacks, eu quero agrupar o csv com as features usando k-means
# e depois usar o k-means para classificar os ataques

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
import time
import os
import sys
import warnings
from sklearn.utils import shuffle
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn import preprocessing

# ignore warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Carregar o CSV
df = pd.read_csv('/home/rodrigo/Desktop/output_bottom.csv')
df.drop(["IT_B_Label",	"IT_M_Label",	"NST_B_Label",	"NST_M_Label",'sAddress', 'rAddress', 'sMACs', 'rMACs', 'sIPs', 'rIPs', 'protocol', 'startDate', 'endDate', 'start', 'end'], axis=1, inplace=True)
df = df.dropna()
print(df)











# K-Means
# Inicializar o K-Means
kmeans = KMeans(n_clusters=8, init='k-means++', random_state=42)

# Treinar o K-Means
kmeans.fit(df)

# Pegar os centróides
centroids = kmeans.cluster_centers_


# Identificando as características dominantes por cluster
n_clusters = 8  # Número de clusters
n_features = len(centroids[0])  # Número de features

for i in range(n_clusters):
    print(f"Cluster {i + 1}:")
    centroid = centroids[i]
    # Calculando o índice da feature com maior valor absoluto no centróide
    dominant_feature = np.argmax(np.abs(centroid))
    print(f"Feature dominante: {dominant_feature}")
    print(df.columns[dominant_feature])

# Obter as coordenadas dos centróides e rótulos dos clusters
centroids = kmeans.cluster_centers_
labels = kmeans.labels_
print(labels)

# Plotar os pontos de dados com cores representando os clusters
plt.figure(figsize=(8, 6))

# Plotar pontos de cada cluster com cores diferentes
for i in range(len(centroids)):
    plt.scatter(df[labels == i][:, 0], df[labels == i][:, 1], label=f'Cluster {i+1}')

# Plotar os centróides
plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', s=200, color='black', label='Centroids')
plt.legend()
plt.title('Gráfico de Dispersão com Clusters')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()