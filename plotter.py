'''
@authors
Lorenzo Baiardi & Thomas Del Moro
'''

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def plot_all():
    kmeans_data = pd.read_csv("results/results_KMEANS.csv", sep=';')
    miq_data = pd.read_csv("results/results_MIQKMEANS.csv", sep=';')

    x_kmeans = np.arange(0, kmeans_data.shape[0])
    x_miq = np.arange(0, miq_data.shape[0])

    kmeans_loss = kmeans_data.iloc[:, 1].values.tolist()
    miq_loss = miq_data.iloc[:, 1].values.tolist()
    kmeans_times = kmeans_data.iloc[:, 0].values.tolist()
    miq_times = miq_data.iloc[:, 0].values.tolist()

    plt.figure(1)
    plt.plot(kmeans_times, kmeans_loss)
    plt.plot(miq_times, miq_loss)
    plt.xlabel("Runtime(s)")
    plt.ylabel("Loss value")
    plt.ylim([1500, 2400])
    plt.legend(["Kmeans", "MIQKmeans"])

    method = ['kmeans', 'MIQkmeans']
    final_loss = [kmeans_loss[-1], miq_loss[-1]]
    plt.figure(2)
    #plt.bar(method, final_loss, width=0.3, color=['#ff7f0e', '#2ca02c'])
    plt.bar('kmeans', kmeans_loss[-1], width=0.3, color=['#ff7f0e'])
    plt.bar('MIQkmeans', miq_loss[-1], width=0.3, color=['#2ca02c'])
    plt.ylabel("Loss value")
    plt.show()
