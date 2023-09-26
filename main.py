'''
@authors
Lorenzo Baiardi & Thomas Del Moro
'''
import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt
from model import KMeans, MIQKMeans, header_result


def main():
    data = pd.read_csv("./xclara.csv")
    print(data.shape[0], data.shape[1])
    print(data)
    data = data.dropna(how='any')
    proc_data = data.iloc[:, :]
    # v1 = data.iloc[:200, 0:1].values
    # v2 = data.iloc[:200, 1:].values
    # plt.scatter(v1, v2)
    # plt.show()

    K = 3
    '''
    m = KMeans("constrained_kmeans", proc_data, K)
    clusters = m.solve()

    proc_data.plot.scatter(0, 1, c=clusters, colormap='gist_rainbow')
    plt.show()
    '''

    header_result()
    m = MIQKMeans("constrained_kmeans", proc_data, K)
    clusters, centroids = m.solve()
    print(centroids)
    print(clusters)

    proc_data.plot.scatter(0, 1, c=clusters, colormap='gist_rainbow')
    plt.show()


if __name__ == '__main__':
    main()
