'''
@authors
Lorenzo Baiardi & Thomas Del Moro
'''
import numpy as np
import pandas as pd
from sklearn import datasets
from matplotlib import pyplot as plt
from plotter import plot_all
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
    clusters = m.solve()
    data.plot.scatter(0, 1, c=clusters, colormap='gist_rainbow')
    plt.show()


def miq_kmeans(data, k):
    header_miqkmeans_result()
    m = MIQKMeans("constrained_kmeans", data, k)
    clusters, objective = m.solve()
    data.plot.scatter(0, 1, c=clusters, colormap='gist_rainbow')
    plt.show()
    return objective


def get_datasets():
    all_dataset = []

    # XCLARA
    data = pd.read_csv("./xclara.csv")
    k = 3
    plot_dataset(data)
    all_dataset.append({"data": data, "k": k})

    # HEART DISEASE PATIENTS
    data = pd.read_csv("./heart_disease_patients.csv")
    k = 2
    plot_dataset(data)
    all_dataset.append({"data": data, "k": k})

    # KIVA COUNTRY PROFILE VARIABLES
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
    data = pd.read_csv("./country_profile_variables.csv")
    k = 3
    plot_dataset(data)
    all_dataset.append({"data": data, "k": k})

    return all_dataset


def main():
    data = None
    k = None

    for data in get_datasets():
        kmeans(data.get("data"), data.get("k"))
        miq_kmeans(data.get("data"), data.get("k"))
        plot_all()


if __name__ == '__main__':
    main()
