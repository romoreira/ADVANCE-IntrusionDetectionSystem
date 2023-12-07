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


def define_number_clusters():
    # elbow method to know the number of clusters
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++')
        kmeans.fit(df)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 11), wcss)
    plt.show()

df_list = []

# Carregar o CSV
df = pd.read_csv('./dataset/output_bottom.csv')
df.drop(["IT_B_Label",	"IT_M_Label",	"NST_B_Label",	"NST_M_Label",'sAddress', 'rAddress', 'sMACs', 'rMACs', 'sIPs', 'rIPs', 'protocol', 'startDate', 'endDate', 'start', 'end'], axis=1, inplace=True)
df = df.dropna()
df_list.append(df)

df = pd.read_csv('./dataset/output_left.csv')
df.drop(["IT_B_Label",	"IT_M_Label",	"NST_B_Label",	"NST_M_Label",'sAddress', 'rAddress', 'sMACs', 'rMACs', 'sIPs', 'rIPs', 'protocol', 'startDate', 'endDate', 'start', 'end'], axis=1, inplace=True)
df = df.dropna()
df_list.append(df)

df = pd.read_csv('./dataset/output_right.csv')
df.drop(["IT_B_Label",	"IT_M_Label",	"NST_B_Label",	"NST_M_Label",'sAddress', 'rAddress', 'sMACs', 'rMACs', 'sIPs', 'rIPs', 'protocol', 'startDate', 'endDate', 'start', 'end'], axis=1, inplace=True)
df = df.dropna()
df_list.append(df)

n_clusters = 8  # Número de clusters
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
features_importance = []
def comparing_datasets(df_list):
    # Para cada dataset, execute o K-means e identifique as features mais importantes
    for dataset in df_list:
        kmeans.fit(dataset)
        centroids = kmeans.cluster_centers_
        features = []
        for i in range(n_clusters):
            centroid = centroids[i]
            dominant_feature = np.argmax(np.abs(centroid))
            features.append(dataset.columns[dominant_feature])
        features_importance.append(features)

comparing_datasets(df_list)
print(features_importance)



# Contando a frequência das features mais importantes em cada dataset
counts = [{feature: features_importance[i].count(feature) for feature in set(features_importance[i])} for i in range(len(features_importance))]

# Lista de todas as features importantes
all_features = list(set([feature for sublist in features_importance for feature in sublist]))

# Criando o gráfico de barras ajustado
bar_width = 0.2  # Largura das barras
gap_between_bars = 0.01  # Espaço entre as barras de diferentes datasets
index = np.arange(len(all_features))

plt.figure(figsize=(10, 6))
ds_names = ['Bottom', 'Left', 'Right']
for i, count in enumerate(counts):
    bar_position = index + (bar_width + gap_between_bars) * i
    plt.bar(bar_position, [count.get(feature, 0) for feature in all_features], bar_width, label=ds_names[i])
    for j, freq in enumerate([count.get(feature, 0) for feature in all_features]):
        plt.text(bar_position[j], freq + 0.1, str(freq), ha='center', va='bottom', fontsize=8)

plt.xlabel('Features')
plt.ylabel('Frequency')
#plt.title('Frequency of the most important features in the dataset')
plt.xticks(index + ((bar_width + gap_between_bars) * (len(counts) - 1)) / 2, all_features, rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('./results/frequency_features.pdf')

exit()

#---------------------------------------------


df = pd.read_csv('./dataset/output_bottom.csv')
df.drop(["IT_B_Label",	"IT_M_Label",	"NST_B_Label",	"NST_M_Label",'sAddress', 'rAddress', 'sMACs', 'rMACs', 'sIPs', 'rIPs', 'protocol', 'startDate', 'endDate', 'start', 'end'], axis=1, inplace=True)
df = df.dropna()

# K-Means
# Inicializar o K-Means
kmeans = KMeans(n_clusters=8, init='k-means++', random_state=42)

# Treinar o K-Means
kmeans.fit(df)

# Pegar os centróides
centroids = kmeans.cluster_centers_

# Pegar as features


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
print("Labels: "+str(labels))



# Transforme os dados em apenas duas dimensões usando PCA ou t-SNE para visualização
# PCA
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df)

# t-SNE (t-distributed Stochastic Neighbor Embedding)
tsne = TSNE(n_components=2, random_state=42)
df_tsne = tsne.fit_transform(df)

# Plot do resultado do K-means com as dimensões reduzidas
def plot_clusters(data, labels, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=50, alpha=0.5)
    plt.title(title)
    plt.xlabel('Componente 1')
    plt.ylabel('Componente 2')
    plt.colorbar()
    plt.show()

# Plot com PCA
plot_clusters(df_pca, labels, 'K-means Clustering (PCA)')

# Plot com t-SNE
plot_clusters(df_tsne, labels, 'K-means Clustering (t-SNE)')



# Calcular os valores de silhueta para cada amostra
silhouette_vals = silhouette_samples(df, labels)

# Calcular o valor médio da silhueta para o conjunto de dados
silhouette_avg = silhouette_score(df, labels)

# Plotar o gráfico de silhueta
plt.figure(figsize=(8, 6))

y_lower = 10
for i in np.unique(labels):
    cluster_silhouette_vals = silhouette_vals[labels == i]
    cluster_silhouette_vals.sort()
    cluster_size = cluster_silhouette_vals.shape[0]
    y_upper = y_lower + cluster_size

    plt.fill_betweenx(np.arange(y_lower, y_upper),
                      0, cluster_silhouette_vals,
                      alpha=0.7, label=f'Cluster {i}')

    plt.text(-0.05, y_lower + 0.5 * cluster_size, str(i))
    y_lower = y_upper + 10

plt.title('Gráfico de Silhueta por Cluster')
plt.xlabel('Valores de Silhueta')
plt.ylabel('Clusters')
plt.axvline(x=silhouette_avg, color='red', linestyle='--', linewidth=1, label='Média de Silhueta')
plt.yticks([])
plt.legend()
plt.show()

exit()