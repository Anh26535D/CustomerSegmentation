from sklearn.cluster import KMeans
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def showSilhouetteScore(data, lower_, upper_):
    for n_clusters in range(lower_, upper_):
        model = KMeans(n_clusters, random_state=42)
        visualizer = SilhouetteVisualizer(model, colors='yellowbrick')
        visualizer.fit(data)
        visualizer.show()
        print("For n_clusters = {}, Kmeans silhouette score is {}".format(n_clusters, visualizer.silhouette_score_))

def lineSilhouetteScore(data, lower_, upper_):
    kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(data) for k in range(lower_, upper_)]
    silhouette_scores = [silhouette_score(data, model.labels_) for model in kmeans_per_k]
    plt.figure(figsize=(8, 3))
    plt.plot(range(lower_, upper_), silhouette_scores, "bo-")
    plt.xlabel("$k$", fontsize=14)
    plt.ylabel("Silhouette score", fontsize=14)
    plt.show()