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
        data, _ = make_blobs(n_samples=1000, n_features=20, centers=4, center_box=(-30, 30), cluster_std=10, random_state=110)
        n_features = data.shape[1]
        col_names = []
        for i in range(n_features):
            col_names.append('V{}'.format(i + 1))
        data = pd.DataFrame(data, columns=col_names)
        plot_dataset(data)

    if dataset_id == 2:
        data, _ = make_blobs(n_samples=1000, n_features=20, centers=5, center_box=(-10, 10), cluster_std=10, random_state=110)
        n_features = data.shape[1]
        col_names = []
        for i in range(n_features):
            col_names.append('V{}'.format(i + 1))
        data = pd.DataFrame(data, columns=col_names)
        plot_dataset(data)

    # HEART DISEASE PATIENTS
    if dataset_id == 3:
        data = pd.read_csv("./heart_disease_patients.csv")
        k = 2
        data = data.drop('id', axis=1)
        plot_dataset(data)
        all_dataset.append({"data": data, "k": k})

    # COVERAGE TYPE DATASET
    if dataset_id == 4:
        data = sklearn.datasets.fetch_covtype(as_frame=True).data
        k = 7
        print(data.head())
        # normalized_data = (data - data.mean()) / data.std()

    return data, k
