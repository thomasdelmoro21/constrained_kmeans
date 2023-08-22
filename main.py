'''
@authors
Lorenzo Baiardi & Thomas Del Moro
'''
import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt
from model import KMeans, MIQKMeans


def plot_clusters(dataset, clusters, k):
    '''
    dataset.insert(2, "cluster", clusters)
    u_labels = np.unique(clusters)
    print(u_labels)
    for i in u_labels:
        plt.scatter(dataset.iloc[2 == i, 0], dataset.iloc[2 == i, 1], label=i)
    plt.legend()
    plt.show()
    '''

def main():
    data = pd.read_csv("./xclara.csv")
    print(data.shape[0], data.shape[1])
    print(data)
    data = data.dropna(how='any')
    v1 = data.iloc[:, 0:1].values
    v2 = data.iloc[:, 1:].values
    plt.scatter(v1, v2)
    plt.show()

    K = 3

    m = MIQKMeans("constrained_kmeans", data, K)
    clusters = m.solve()
    print(clusters)

    data.plot.scatter(0, 1, c=clusters, colormap='gist_rainbow')
    plt.show()

if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
