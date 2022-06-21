from sklearn import metrics
from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def showDaviesBouldinScore(data, lower_, upper_):
    kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(data) for k in range(lower_, upper_)]
    CH_scores = [davies_bouldin_score(data, model.labels_) for model in kmeans_per_k]
    plt.figure(figsize=(8, 3))
    plt.plot(range(lower_, upper_), CH_scores, "bo-")
    plt.xlabel("$k$", fontsize=14)
    plt.ylabel("Davies Bouldin score", fontsize=14)
    plt.show()