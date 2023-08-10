'''
@authors
Lorenzo Baiardi & Thomas Del Moro
'''
import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt
from model import KMeans


def main():
    data = pd.read_csv("./xclara.csv")
    print(data.shape[0], data.shape[1])
    print(data)
    data = data.dropna(how='any')
    v1 = data.iloc[:, 0:1].values
    v2 = data.iloc[:, 1:].values
    plt.scatter(v1, v2)
    plt.show()

    m = KMeans("constrained_kmeans", data, k=3)
    clusters, objective_values = m.solve()
    print(clusters)


if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
