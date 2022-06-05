from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def showElbow(data):
    sse = {}
    for k in range(1,11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        sse[k] = kmeans.inertia_
    sse_keys = list(sse.keys())
    sse_values = list(sse.values())
    plt.plot(sse_keys, sse_values)
    plt.show()