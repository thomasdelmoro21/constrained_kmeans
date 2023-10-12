'''
@authors
Lorenzo Baiardi & Thomas Del Moro
'''
import numpy as np
import pandas as pd
from sklearn.datasets import *
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from plotter import plot_all
from kmeans import KMeans, header_kmeans_result
from miqkmeans import MIQKMeans, header_miqkmeans_result

DATASET = 1


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
    return m.solve()


def miq_kmeans(data, k):
    header_miqkmeans_result()
    m = MIQKMeans("constrained_kmeans", data, k)
    clusters = m.solve()
    return clusters


def get_dataset(dataset_id):
    all_dataset = []
    data = None
    k = None

    if dataset_id == 1:
        data, _ = make_blobs(n_samples=10000, n_features=4, centers=3, cluster_std=10, random_state=110)
        n_features = data.shape[1]
        col_names = []
        for i in range(n_features):
            col_names.append('V{}'.format(i + 1))
        data = pd.DataFrame(data, columns=col_names)
        k = 3
        print(data.head())

    if dataset_id == 2:
        data, _ = make_sparse_uncorrelated(n_samples=100, n_features=10, random_state=110)
        n_features = data.shape[1]
        col_names = []
        for i in range(n_features):
            col_names.append('V{}'.format(i + 1))
        data = pd.DataFrame(data, columns=col_names)
        k = 5
        print(data.head())

    if dataset_id == 10:  # XCLARA
        data = pd.read_csv("./xclara.csv")
        k = 3
        plot_dataset(data)
        all_dataset.append({"data": data, "k": k})

    # HEART DISEASE PATIENTS
    if dataset_id == 11:
        data = pd.read_csv("./heart_disease_patients.csv")
        k = 2
        plot_dataset(data)
        all_dataset.append({"data": data, "k": k})

    # KIVA COUNTRY PROFILE VARIABLES
    # BETA
    if dataset_id == 12:
        data = pd.read_csv("./kiva_country_profile_variables.csv")
        print(data)
        data = data.iloc[:, 2:]
        replacement = {'~0': 0,
                       '-~0.0': 0,
                       '~0.0': 0,
                       '...': -99,
                       '': -99}
        data = data.replace(replacement)
        k = 10
        plot_dataset(data)
        all_dataset.append({"data": data, "k": k})

    # COUNTRY PROFILE VARIABLES
    # BETA
    if dataset_id == 13:
        data = pd.read_csv("./country_profile_variables.csv")
        k = 3
        plot_dataset(data)
        all_dataset.append({"data": data, "k": k})

    return data, k


def main():
    data = None
    k = None

    data, k = get_dataset(DATASET)
    if data is None:
        raise Exception("NO DATA AVAILABLE")
    if k is None:
        raise Exception("NUMBER OF CLUSTERS NOT AVAILABLE")

    kmeans_clusters = kmeans(data, k)
    miq_clusters = miq_kmeans(data, k)

    pca = PCA()
    data_pc = pca.fit_transform(data)
    plt.scatter(data_pc[:, 0], data_pc[:, 1], c=kmeans_clusters)
    plt.scatter(data_pc[:, 0], data_pc[:, 1], c=miq_clusters)

    plt.show()


if __name__ == '__main__':
    main()
