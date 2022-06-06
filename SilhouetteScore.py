from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

def showSilhouetteScore(data, lower_, upper_):
    for k in range(lower_, upper_):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        label = kmeans.predict(data)
        s = silhouette_score(data, label)
        print(s)



