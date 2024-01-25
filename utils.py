'''
@authors
Lorenzo Baiardi & Thomas Del Moro
'''

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


def copy_csv(filename, copy_filename):
    df = pd.read_csv(filename, sep=";")
    df.to_csv(copy_filename, sep=";")


def plot_dataset(data):
    plt.figure()
    pca = PCA()
    data_pc = pca.fit_transform(data)
    plt.scatter(data_pc[:, 0], data_pc[:, 1])
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()


def plot_loss_runtime():
    kmeans_data = pd.read_csv("./results/csv/results_KMEANS_size20", sep=';')
    miq_data = pd.read_csv("./results/csv/results_MIQKMEANS_size20", sep=';')

    x_kmeans = np.arange(0, kmeans_data.shape[0])
    x_miq = np.arange(0, miq_data.shape[0])

    kmeans_loss = kmeans_data.iloc[:, 2].values.tolist()
    miq_loss = miq_data.iloc[10:300, 2].values.tolist()
    kmeans_times = kmeans_data.iloc[:, 1].values.tolist()
    miq_times = miq_data.iloc[10:300, 1].values.tolist()

    plt.figure(1)
    plt.plot(np.asarray(kmeans_times, float), kmeans_loss)
    plt.plot(np.asarray(miq_times, float), miq_loss)
    plt.xlabel("Runtime(s)")
    plt.ylabel("Loss value")
    #plt.ylim([1500, 2400])
    plt.legend(["Kmeans", "MIQKmeans"])

    method = ['kmeans', 'MIQkmeans']
    final_loss = [kmeans_loss[-1], miq_loss[-1]]
    plt.figure(2)
    #plt.bar(method, final_loss, width=0.3, color=['#ff7f0e', '#2ca02c'])
    plt.bar('kmeans', kmeans_loss[-1], width=0.3, color=['#ff7f0e'])
    plt.bar('MIQkmeans', miq_loss[-1], width=0.3, color=['#2ca02c'])
    plt.ylabel("Loss value")
    plt.show()

def plot_test_size():
    # plot di tutti i runtime in cui miq supera kmeans
    test_sizes = [20, 40, 60, 80, 100, 120]
    kmeans_runtimes = []
    miq_runtimes = []
    miq_best_runtimes = []
    kmeans_losses = []
    miq_best_losses = []
    for size in test_sizes:
        k_path = "results/csv/results_KMEANS_size{}".format(size)
        m_path = "results/csv/results_MIQKMEANS_size{}".format(size)
        kmeans_data = pd.read_csv(k_path, sep=';')
        miq_data = pd.read_csv(m_path, sep=';')
        k_loss = kmeans_data.iloc[-1, 2]
        m_time = None
        for l in miq_data.iloc[:, 2]:
            if l < k_loss:
                m_loss = l
                m_time = miq_data.iloc[miq_data[miq_data.iloc[:, 2] == l].index[0], 1]
                break
        kmeans_runtimes.append(kmeans_data.iloc[-1, 1])
        kmeans_losses.append(k_loss)
        miq_runtimes.append(m_time)
        m_best_loss = miq_data.iloc[-1, 2]
        miq_best_losses.append(m_best_loss)
        miq_best_runtimes.append(miq_data.iloc[miq_data[miq_data.iloc[:, 2] == m_best_loss].index[0], 1])

    plt.figure()
    plt.plot(test_sizes, kmeans_runtimes, '-o', color='#ff7f0e')
    plt.plot(test_sizes, miq_best_runtimes, '-o', color='#2ca02c')
    plt.plot(test_sizes, miq_runtimes, '-o', color='red')
    plt.xlabel("Number of elements")
    plt.ylabel("Runtime(s)")
    #plt.ylim([0, 300])
    plt.legend(["Kmeans", "MIQKmeans-best", "MIQ loss < Kmeans loss"])
    plt.title("Runtimes")
    plt.savefig("results/log_plots/runtime_size_heart.png")

    plt.figure()
    plt.semilogy(test_sizes, kmeans_runtimes, '-o', color='#ff7f0e')
    plt.semilogy(test_sizes, miq_best_runtimes, '-o', color='#2ca02c')
    plt.semilogy(test_sizes, miq_runtimes, '-o', color='red')
    plt.xlabel("Number of elements")
    plt.ylabel("Runtime(s)")
    #plt.ylim([0, 300])
    plt.legend(["Kmeans", "MIQKmeans-best", "MIQ loss < Kmeans loss"])
    plt.title("Runtimes")
    plt.savefig("results/log_plots/runtime_size_heart_log.png")

    plt.figure()
    x = np.asarray(test_sizes)
    plt.bar(x-1, kmeans_losses, width=2, color='#ff7f0e')
    plt.bar(x+1, miq_best_losses, width=2, color='#2ca02c')
    plt.xlabel("Number of elements")
    plt.ylabel("Loss value")
    plt.legend(["Kmeans", "MIQKmeans"])
    plt.title("Loss Values")
    plt.savefig("results/log_plots/loss_size_heart.png")


def plot_test_features():
    test_features = [2, 4, 6, 8, 10]
    kmeans_runtimes = []
    miq_runtimes = []
    miq_best_runtimes = []
    kmeans_losses = []
    miq_best_losses = []
    for n_features in test_features:
        k_path = "results/csv/results_KMEANS_f{}".format(n_features)
        m_path = "results/csv/results_MIQKMEANS_f{}".format(n_features)
        kmeans_data = pd.read_csv(k_path, sep=';')
        miq_data = pd.read_csv(m_path, sep=';')
        k_loss = kmeans_data.iloc[-1, 2]
        m_time = None
        for l in miq_data.iloc[:, 2]:
            if l < k_loss:
                m_loss = l
                m_time = miq_data.iloc[miq_data[miq_data.iloc[:, 2] == l].index[0], 1]
                break
        kmeans_runtimes.append(kmeans_data.iloc[-1, 1])
        kmeans_losses.append(k_loss)
        miq_runtimes.append(m_time)
        m_best_loss = miq_data.iloc[-1, 2]
        miq_best_losses.append(m_best_loss)
        miq_best_runtimes.append(miq_data.iloc[miq_data[miq_data.iloc[:, 2] == m_best_loss].index[0], 1])

    plt.figure()
    plt.plot(test_features, kmeans_runtimes, '-o', color='#ff7f0e')
    plt.plot(test_features, miq_best_runtimes, '-o', color='#2ca02c')
    plt.plot(test_features, miq_runtimes, '-o', color='red')
    plt.xlabel("Number of features")
    plt.ylabel("Runtime(s)")
    #plt.ylim([0, 150])
    plt.legend(["Kmeans", "MIQKmeans-best", "MIQ loss < Kmeans loss"])
    plt.title("Runtimes")
    plt.savefig("results/log_plots/runtime_features_heart.png")

    plt.figure()
    plt.semilogy(test_features, kmeans_runtimes, '-o', color='#ff7f0e')
    plt.semilogy(test_features, miq_best_runtimes, '-o', color='#2ca02c')
    plt.semilogy(test_features, miq_runtimes, '-o', color='red')
    plt.xlabel("Number of features")
    plt.ylabel("Runtime(s)")
    #plt.ylim([0, 150])
    plt.legend(["Kmeans", "MIQKmeans-best", "MIQ loss < Kmeans loss"])
    plt.title("Runtimes")
    plt.savefig("results/log_plots/runtime_features_heart_log.png")

    plt.figure()
    x = np.asarray(test_features)
    plt.bar(x-0.15, kmeans_losses, width=0.3, color=['#ff7f0e'])
    plt.bar(x+0.15, miq_best_losses, width=0.3, color=['#2ca02c'])
    plt.xlabel("Number of features")
    plt.ylabel("Loss value")
    plt.legend(["Kmeans", "MIQKmeans"])
    plt.title("Loss Values")
    plt.savefig("results/log_plots/loss_features_heart.png")


def plot_test_centers():
    test_centers = [2, 4, 6, 8, 10]
    kmeans_runtimes = []
    miq_runtimes = []
    miq_best_runtimes = []
    kmeans_losses = []
    miq_best_losses = []
    for n_centers in test_centers:
        k_path = "results/csv/results_KMEANS_k{}".format(n_centers)
        m_path = "results/csv/results_MIQKMEANS_k{}".format(n_centers)
        kmeans_data = pd.read_csv(k_path, sep=';')
        miq_data = pd.read_csv(m_path, sep=';')
        k_loss = kmeans_data.iloc[-1, 2]
        m_time = None
        for l in miq_data.iloc[:, 2]:
            if l < k_loss:
                m_loss = l
                m_time = miq_data.iloc[miq_data[miq_data.iloc[:, 2] == l].index[0], 1]
                break
        kmeans_runtimes.append(kmeans_data.iloc[-1, 1])
        kmeans_losses.append(k_loss)
        miq_runtimes.append(m_time)
        m_best_loss = miq_data.iloc[-1, 2]
        miq_best_losses.append(m_best_loss)
        miq_best_runtimes.append(miq_data.iloc[miq_data[miq_data.iloc[:, 2] == m_best_loss].index[0], 1])

    plt.figure()
    plt.plot(test_centers, kmeans_runtimes, '-o', color='#ff7f0e')
    plt.plot(test_centers, miq_best_runtimes, '-o', color='#2ca02c')
    plt.plot(test_centers, miq_runtimes, '-o', color='red')
    plt.xlabel("Number of clusters")
    plt.ylabel("Runtime(s)")
    # plt.ylim([0, 10])
    plt.legend(["Kmeans", "MIQKmeans-best", "MIQ loss < Kmeans loss"])
    plt.title("Runtimes")
    plt.savefig("results/log_plots/runtime_centers_heart.png")

    plt.figure()
    plt.semilogy(test_centers, kmeans_runtimes, '-o', color='#ff7f0e')
    plt.semilogy(test_centers, miq_best_runtimes, '-o', color='#2ca02c')
    plt.semilogy(test_centers, miq_runtimes, '-o', color='red')
    plt.xlabel("Number of clusters")
    plt.ylabel("Runtime(s)")
    # plt.ylim([0, 10])
    plt.legend(["Kmeans", "MIQKmeans-best", "MIQ loss < Kmeans loss"])
    plt.title("Runtimes")
    plt.savefig("results/log_plots/runtime_centers_heart_log.png")

    plt.figure()
    x = np.asarray(test_centers)
    plt.bar(x - 0.15, kmeans_losses, width=0.3, color=['#ff7f0e'])
    plt.bar(x + 0.15, miq_best_losses, width=0.3, color=['#2ca02c'])
    plt.xlabel("Number of clusters")
    plt.ylabel("Loss value")
    plt.legend(["Kmeans", "MIQKmeans"])
    plt.title("Loss Values")
    plt.savefig("results/log_plots/loss_centers_heart.png")


def plot_multiple_inits(vars, kmeans_losses, kmeans_runtimes):
    miq_runtimes = []
    miq_best_runtimes = []
    miq_best_losses = []
    for n_vars in vars:
        m_path = "results/csv/results_MIQKMEANS_mult{}".format(n_vars)
        miq_data = pd.read_csv(m_path, sep=';')
        k_loss = kmeans_losses[vars.index(n_vars)]
        m_time = None
        for l in miq_data.iloc[:, 2]:
            if l < k_loss:
                m_loss = l
                m_time = miq_data.iloc[miq_data[miq_data.iloc[:, 2] == l].index[0], 1]
                break
        miq_runtimes.append(m_time)
        m_best_loss = miq_data.iloc[-1, 2]
        miq_best_losses.append(m_best_loss)
        miq_best_runtimes.append(miq_data.iloc[miq_data[miq_data.iloc[:, 2] == m_best_loss].index[0], 1])

    plt.figure()
    plt.plot(vars, kmeans_runtimes, '-o', color='#ff7f0e')
    plt.plot(vars, miq_best_runtimes, '-o', color='#2ca02c')
    plt.plot(vars, miq_runtimes, '-o', color='red')
    plt.xlabel("Number of binary variables")
    plt.ylabel("Runtime(s)")
    #plt.ylim([0, 300])
    plt.legend(["Kmeans", "MIQKmeans-best", "MIQ loss < Kmeans loss"])
    plt.title("Runtimes")
    plt.savefig("results/log_plots/runtime_multiple_inits_heart.png")

    plt.figure()
    plt.semilogy(vars, kmeans_runtimes, '-o', color='#ff7f0e')
    plt.semilogy(vars, miq_best_runtimes, '-o', color='#2ca02c')
    plt.semilogy(vars, miq_runtimes, '-o', color='red')
    plt.xlabel("Number of binary variables")
    plt.ylabel("Runtime(s)")
    #plt.ylim([0, 300])
    plt.legend(["Kmeans", "MIQKmeans-best", "MIQ loss < Kmeans loss"])
    plt.title("Runtimes")
    plt.savefig("results/log_plots/runtime_multiple_inits_heart_log.png")

    plt.figure()
    x = np.asarray(vars)
    plt.bar(x-4, kmeans_losses, width=8, color=['#ff7f0e'])
    plt.bar(x+4, miq_best_losses, width=8, color=['#2ca02c'])
    plt.xlabel("Number of binary variables")
    plt.ylabel("Loss value")
    plt.legend(["Kmeans", "MIQKmeans"])
    plt.title("Loss Values")
    plt.savefig("results/log_plots/loss_multiple_inits_heart.png")