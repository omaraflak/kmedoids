import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics.pairwise import pairwise_distances
from kmedoids import KMedoids

def plot_graphs(data, clusters):
    colors = {0:'b*', 1:'g^',2:'ro',3:'c*', 4:'m^', 5:'yo', 6:'ko', 7:'w*'}
    index = 0
    for key in clusters.keys():
        temp_data = clusters[key]
        x = [data[i][0] for i in temp_data]
        y = [data[i][1] for i in temp_data]
        plt.plot(x, y, colors[index])
        index += 1
    plt.title('Cluster formations')
    plt.show()

def main():
    # generate random points
    X, _ = make_blobs(n_samples=100, centers=3)

    # compute distance matrix
    dist = pairwise_distances(X, metric='euclidean')

    # k-medoids algorithm
    km = KMedoids(distance_matrix=dist, n_clusters=3)
    km.run(max_iterations=10, tolerance=0.00001)

    print(km.clusters)
    plot_graphs(X, km.clusters)

if __name__ == '__main__':
    main()
