"""
@authors
Lorenzo Baiardi & Thomas Del Moro
"""

import pandas as pd
import sklearn.datasets
from sklearn.datasets import *

from utils import plot_dataset


def get_dataset(dataset_id):
    all_dataset = []
    data = None
    k = None

    if dataset_id == 1:
        data, _ = make_blobs(n_samples=1000, n_features=20, centers=5, cluster_std=10, random_state=110)
        n_features = data.shape[1]
        col_names = []
        for i in range(n_features):
            col_names.append('V{}'.format(i + 1))
        data = pd.DataFrame(data, columns=col_names)
        plot_dataset(data)

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
        data = data.drop('id', axis=1)
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


    # COVERAGE TYPE DATASET
    if dataset_id == 14:
        data = sklearn.datasets.fetch_covtype(as_frame=True).data
        k = 7
        print(data.head())
        # normalized_data = (data - data.mean()) / data.std()

    return data, k
