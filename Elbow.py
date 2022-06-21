from matplotlib import markers
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def showElbow(data, lower_, upper_):
    sse = {}
    for k in range(lower_,upper_):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        sse[k] = kmeans.inertia_
    sse_keys = list(sse.keys())
    sse_values = list(sse.values())
    plt.plot(sse_keys, sse_values, marker='o', markerfacecolor='red')
    plt.show()
    return sse