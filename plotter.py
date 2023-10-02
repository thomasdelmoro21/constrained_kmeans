'''
@authors
Lorenzo Baiardi & Thomas Del Moro
'''

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def plot_all():
    kmeans_data = pd.read_csv("./results_KMEANS.csv", sep=';')
    miq_data = pd.read_csv("./results_MIQKMEANS.csv", sep=';')

    x_kmeans = np.arange(0, kmeans_data.shape[0])
    x_miq = np.arange(0, miq_data.shape[0])

    plt.figure(1)
    plt.plot(kmeans_data.iloc[:, 0], kmeans_data.iloc[:, 1])
    plt.plot(miq_data.iloc[10:, 0], miq_data.iloc[10:, 1])
    plt.xlabel("Runtime(s)")
    plt.ylabel("Loss value")
    plt.legend(["Kmeans", "MIQKmeans"])
    plt.title("Grafico 1")

    '''
    x = np.arange(2)
    plt.bar(x_kmeans, times, width=0.25, color='r', label='edit distance')
    plt.bar(x + 0.30, twoGramTimes, width=0.25, color='b', label='2-gram')
    plt.bar(x + 0.60, threeGramTimes, width=0.25, color='g', label='3-gram')
    plt.xlabel('Lunghezza parole')
    plt.ylabel('Tempo (secondi)')
    plt.title('Grafico 1')
    plt.xticks(x + 0.30, ('4 lettere', '5 lettere', '7 lettere', '10 lettere', '11 lettere', '15 lettere'))
    plt.legend()
    plt.show()
    '''

    plt.show()
