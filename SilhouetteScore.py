from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from yellowbrick.cluster import SilhouetteVisualizer

def showSilhouetteScore(data, lower_, upper_):
    fig, ax = plt.subplots(2, 2, figsize=(15,8))
    for k in range(lower_, upper_):
            kmeans = KMeans(n_clusters=k, random_state=42)
            q, mod = divmod(k, 2)
            visualizer = SilhouetteVisualizer(kmeans, colors='yellowbrick', ax=ax[q-1][mod])
            visualizer.fit(data)
            kmeans.fit(data)
    visualizer.show()

