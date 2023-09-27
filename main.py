'''
@authors
Lorenzo Baiardi & Thomas Del Moro
'''
import pandas as pd
from matplotlib import pyplot as plt

from kmeans import KMeans, header_kmeans_result
from miqkmeans import MIQKMeans, header_miqkmeans_result


def plot_dataset(data):
    print(data.shape[0], data.shape[1])
    print(data)
    x1 = data.iloc[:, 0:1].values
    x2 = data.iloc[:, 1:].values
    plt.scatter(x1, x2)
    plt.show()


def kmeans(data, k):
    header_kmeans_result()
    m = KMeans("constrained_kmeans", data, k)
    clusters, objectives = m.solve()
    data.plot.scatter(0, 1, c=clusters, colormap='gist_rainbow')
    plt.show()


def miq_kmeans(data, k):
    header_miqkmeans_result()
    m = MIQKMeans("constrained_kmeans", data, k)
    clusters, centroids = m.solve()
    data.plot.scatter(0, 1, c=clusters, colormap='gist_rainbow')
    plt.show()


def main():
    data = pd.read_csv("./xclara.csv")
    data = data.dropna(how='any')
    plot_dataset(data)
    k = 3
    kmeans(data, k)
    miq_kmeans(data, k)


if __name__ == '__main__':
    main()
