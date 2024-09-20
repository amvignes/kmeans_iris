# Importer les bibliothèques nécessaires
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Charger le dataset Iris
iris = load_iris()
X = iris.data  # Les caractéristiques (features)

# Appliquer K-means avec 3 clusters (car il y a 3 espèces dans le dataset Iris)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Obtenir les labels prédits par K-means
labels = kmeans.labels_

# Visualiser les résultats en utilisant deux dimensions (longueur et largeur des pétales)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X[:, 2], y=X[:, 3], hue=labels, palette='Set1', s=100)

# Ajouter les centres des clusters au graphique
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 2], centroids[:, 3], s=300, c='yellow', marker='X', label='Centroids')

# Personnaliser le graphique
plt.title("K-means clustering sur le dataset Iris (Pétales)")
plt.xlabel("Longueur des pétales (cm)")
plt.ylabel("Largeur des pétales (cm)")
plt.legend()
plt.show()
