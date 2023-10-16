"""
@authors
Lorenzo Baiardi & Thomas Del Moro
"""
import numpy as np
import pandas as pd
from sklearn.datasets import *
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from plotter import plot_all
from kmeans import KMeans, header_kmeans_result
from miqkmeans import MIQKMeans, header_miqkmeans_result

DATASET = 3


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
    return m.solve()


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

    if dataset_id == 3:
        data, _ = make_moons(n_samples=1000, noise=.05)
        n_features = data.shape[1]
        col_names = []
        for i in range(n_features):
            col_names.append('V{}'.format(i + 1))
        data = pd.DataFrame(data, columns=col_names)
        k = 2
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


def test_synthetic_datasets():
    test_sizes = np.linspace(1000, 100000, 2, dtype=int)
    test_sizes = [10000]
    test_features = np.linspace(2, 20, 4, dtype=int)
    test_features = [3]
    test_centers = np.linspace(2, 20, 5, dtype=int)

    for size in test_sizes:
        kmeans_losses = []
        kmeans_runtimes = []
        miq_losses = []
        miq_runtimes = []
        for n_features in test_features:
            data, _ = make_blobs(n_samples=size, n_features=n_features, centers=3, cluster_std=10, random_state=777)
            col_names = []
            for i in range(n_features):
                col_names.append('V{}'.format(i + 1))
            data = pd.DataFrame(data, columns=col_names)
    
            kmeans_clusters, kmeans_loss, kmeans_runtime = kmeans(data, 3)
            kmeans_losses.append(kmeans_loss)
            kmeans_runtimes.append(kmeans_runtime)
            miq_clusters, miq_loss, miq_runtime = miq_kmeans(data, 3)
            miq_losses.append(miq_loss)
            miq_runtimes.append(miq_runtime)

            print(f"\n**TEST: {n_features} features")
            print(f"KMEANS LOSS: {kmeans_loss}")
            print(f"KMEANS RUNTIME: {kmeans_runtime}")
            print(f"MIQKMEANS LOSS: {miq_loss}")
            print(f"MIQKMEANS RUNTIME: {miq_runtime}")

            plt.figure(1)
            pca = PCA()
            data_pc = pca.fit_transform(data)
            plt.scatter(data_pc[:, 0], data_pc[:, 1], c=kmeans_clusters)
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.figure(2)
            plt.scatter(data_pc[:, 0], data_pc[:, 1], c=miq_clusters)
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.show()

        x = test_features
        plt.plot(x, kmeans_losses)
        plt.plot(x, miq_losses)
        plt.xlabel("Number of features")
        plt.ylabel("Loss value")
        plt.legend(["Kmeans", "MIQKmeans"])
        plt.title("Loss Values")

        plt.figure()
        plt.plot(x, kmeans_runtimes)
        plt.plot(x, miq_runtimes)
        plt.xlabel("Number of features")
        plt.ylabel("Runtime(s)")
        plt.legend(["Kmeans", "MIQKmeans"])
        plt.title("Runtimes")

        plt.show()

    '''
    # test al variare dei dati di input: modificare valori in alto e scambiare ordine dei cicli per ottenere tutti i plot
    for size in test_sizes:
        for n_features in test_features:
            kmeans_losses = []
            kmeans_runtimes = []
            miq_losses = []
            miq_runtimes = []
            for n_centers in test_centers:
                data, _ = make_blobs(n_samples=size, n_features=n_features, centers=n_centers, cluster_std=10, random_state=110)
                col_names = []
                for i in range(n_features):
                    col_names.append('V{}'.format(i + 1))
                data = pd.DataFrame(data, columns=col_names)

                kmeans_clusters, kmeans_loss, kmeans_runtime = kmeans(data, n_centers)
                kmeans_losses.append(kmeans_loss)
                kmeans_runtimes.append(kmeans_runtime)
                miq_clusters, miq_loss, miq_runtime = miq_kmeans(data, n_centers)
                miq_losses.append(miq_loss)
                miq_runtimes.append(miq_runtime)

                plt.figure()
                plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=miq_clusters)
                plt.show()

            x = test_centers
            #x = test_features
            #x = test_sizes
            plt.figure()
            plt.plot(x, kmeans_losses)
            plt.plot(x, miq_losses)
            plt.xlabel("Number of clusters")
            #plt.xlabel("Number of features")
            #plt.xlabel("Number of elements")
            plt.ylabel("Loss value")
            plt.legend(["Kmeans", "MIQKmeans"])
            plt.title("Loss Values")

            plt.figure()
            plt.plot(x, kmeans_runtimes)
            plt.plot(x, miq_runtimes)
            plt.xlabel("Number of clusters")
            # plt.xlabel("Number of features")
            # plt.xlabel("Number of elements")
            plt.ylabel("Runtime(s)")
            plt.legend(["Kmeans", "MIQKmeans"])
            plt.title("Runtimes")

            plt.show()
    '''

def main():
    data = None
    k = None

    #data, k = get_dataset(DATASET)
    if data is None:
        raise Exception("NO DATA AVAILABLE")
    if k is None:
        raise Exception("NUMBER OF CLUSTERS NOT AVAILABLE")

    """ 
    kmeans_clusters, kmeans_loss, kmeans_runtime = kmeans(data, k)
    miq_clusters, miq_loss, miq_runtime = miq_kmeans(data, k)

    pca = PCA()
    data_pc = pca.fit_transform(data)
    plt.figure(1)
    plt.scatter(data_pc[:, 0], data_pc[:, 1], c=kmeans_clusters)
    plt.figure(2)
    plt.scatter(data_pc[:, 0], data_pc[:, 1], c=miq_clusters)
    plt.show()

    plot_all()
    """

    test_synthetic_datasets()


if __name__ == '__main__':
    main()
