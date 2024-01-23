"""
@authors
Lorenzo Baiardi & Thomas Del Moro
"""

from sklearn.datasets import *

from kmeans import KMeans, header_kmeans_result
from miqkmeans import MIQKMeans, header_miqkmeans_result
from handle_data import get_dataset
from utils import *

DATASET = 1
TEST = 1    # 1: test size, 2: test features, 3: test centers


def kmeans(data, k):
    header_kmeans_result()
    m = KMeans("constrained_kmeans", data, k)
    return m.solve()


def miq_kmeans(data, k, timelimit):
    header_miqkmeans_result()
    m = MIQKMeans("constrained_kmeans", data, k, timelimit)
    return m.solve()


def test_sizes():
    data, k = get_dataset(DATASET)

    sizes = [10, 20, 30, 40, 50, 60]

    kmeans_losses = []
    kmeans_runtimes = []
    miq_losses = []
    miq_runtimes = []
    for size in sizes:
        cur_data = data.sample(frac=1)
        cur_data = cur_data.iloc[:size, :4]
        k = 3
        kmeans_clusters, kmeans_loss, kmeans_runtime = kmeans(cur_data, k)
        miq_clusters, miq_loss, miq_runtime = miq_kmeans(cur_data, k, cur_data.shape[0] * 10)

        print(f"\n**TEST: SIZE {size}")
        print(f"KMEANS LOSS: {kmeans_loss}")
        print(f"KMEANS RUNTIME: {kmeans_runtime}")
        print(f"MIQKMEANS LOSS: {miq_loss}")
        print(f"MIQKMEANS RUNTIME: {miq_runtime}")

        kmeans_losses.append(kmeans_loss)
        kmeans_runtimes.append(kmeans_runtime)
        miq_losses.append(miq_loss)
        miq_runtimes.append(miq_runtime)

        copy_csv('results/results_MIQKMEANS.csv', 'results/csv/results_MIQKMEANS_size{}'.format(size))
        copy_csv('results/results_KMEANS.csv', 'results/csv/results_KMEANS_size{}'.format(size))

        plt.figure()
        pca = PCA()
        data_pc = pca.fit_transform(cur_data)
        plt.scatter(data_pc[:, 0], data_pc[:, 1], c=kmeans_clusters)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.figure()
        plt.scatter(data_pc[:, 0], data_pc[:, 1], c=miq_clusters)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.show()

    x = sizes
    plt.plot(x, kmeans_losses)
    plt.plot(x, miq_losses)
    plt.xlabel("Number of elements")
    plt.ylabel("Loss value")
    plt.legend(["Kmeans", "MIQKmeans"])
    plt.title("Loss Values")

    plt.figure()
    plt.plot(x, kmeans_runtimes)
    plt.plot(x, miq_runtimes)
    plt.xlabel("Number of elements")
    plt.ylabel("Runtime(s)")
    plt.legend(["Kmeans", "MIQKmeans"])
    plt.title("Runtimes")

    plt.show()


def test_features():
    data, k = get_dataset(DATASET)

    features = [2, 4, 6, 8, 10]

    kmeans_losses = []
    kmeans_runtimes = []
    miq_losses = []
    miq_runtimes = []
    for n_features in features:
        cur_data = data.sample(frac=1)
        cur_data = cur_data.iloc[:30, :n_features]
        k = 3
        kmeans_clusters, kmeans_loss, kmeans_runtime = kmeans(cur_data, k)
        miq_clusters, miq_loss, miq_runtime = miq_kmeans(cur_data, k, 300 + 50 * n_features)

        print(f"\n**TEST: FEATURES {n_features}")
        print(f"KMEANS LOSS: {kmeans_loss}")
        print(f"KMEANS RUNTIME: {kmeans_runtime}")
        print(f"MIQKMEANS LOSS: {miq_loss}")
        print(f"MIQKMEANS RUNTIME: {miq_runtime}")

        kmeans_losses.append(kmeans_loss)
        kmeans_runtimes.append(kmeans_runtime)
        miq_losses.append(miq_loss)
        miq_runtimes.append(miq_runtime)

        copy_csv('results/results_MIQKMEANS.csv', 'results/csv/results_MIQKMEANS_f{}'.format(n_features))
        copy_csv('results/results_KMEANS.csv', 'results/csv/results_KMEANS_f{}'.format(n_features))

        plt.figure()
        pca = PCA()
        data_pc = pca.fit_transform(cur_data)
        plt.scatter(data_pc[:, 0], data_pc[:, 1], c=kmeans_clusters)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.figure()
        plt.scatter(data_pc[:, 0], data_pc[:, 1], c=miq_clusters)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.show()

    x = features
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


def test_centers():
    data, k = get_dataset(DATASET)

    centers = [2, 4, 6, 8, 10]

    kmeans_losses = []
    kmeans_runtimes = []
    miq_losses = []
    miq_runtimes = []
    for n_centers in centers:
        cur_data = data.sample(frac=1)
        cur_data = cur_data.iloc[:30, :4]
        k = n_centers
        kmeans_clusters, kmeans_loss, kmeans_runtime = kmeans(cur_data, k)
        miq_clusters, miq_loss, miq_runtime = miq_kmeans(cur_data, k, n_centers*100)

        print(f"\n**TEST: REAL DATA {k} centers")
        print(f"KMEANS LOSS: {kmeans_loss}")
        print(f"KMEANS RUNTIME: {kmeans_runtime}")
        print(f"MIQKMEANS LOSS: {miq_loss}")
        print(f"MIQKMEANS RUNTIME: {miq_runtime}")

        kmeans_losses.append(kmeans_loss)
        kmeans_runtimes.append(kmeans_runtime)
        miq_losses.append(miq_loss)
        miq_runtimes.append(miq_runtime)

        copy_csv('results/results_MIQKMEANS.csv', 'results/csv/results_MIQKMEANS_k{}'.format(k))
        copy_csv('results/results_KMEANS.csv', 'results/csv/results_KMEANS_k{}'.format(k))

        plt.figure()
        pca = PCA()
        data_pc = pca.fit_transform(cur_data)
        plt.scatter(data_pc[:, 0], data_pc[:, 1], c=kmeans_clusters)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.figure()
        plt.scatter(data_pc[:, 0], data_pc[:, 1], c=miq_clusters)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.show()

    x = centers
    plt.plot(x, kmeans_losses)
    plt.plot(x, miq_losses)
    plt.xlabel("Number of clusters")
    plt.ylabel("Loss value")
    plt.legend(["Kmeans", "MIQKmeans"])
    plt.title("Loss Values")

    plt.figure()
    plt.plot(x, kmeans_runtimes)
    plt.plot(x, miq_runtimes)
    plt.xlabel("Number of clusters")
    plt.ylabel("Runtime(s)")
    plt.legend(["Kmeans", "MIQKmeans"])
    plt.title("Runtimes")

    plt.show()


def main():

    if TEST == 1:
        test_sizes()
        plot_test_size()
    elif TEST == 2:
        test_features()
        plot_test_features()
    elif TEST == 3:
        test_centers()
        plot_test_centers()
    else:
        pass


if __name__ == '__main__':
    main()
