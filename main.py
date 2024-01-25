"""
@authors
Lorenzo Baiardi & Thomas Del Moro
"""
import math

from sklearn.datasets import *

from kmeans import KMeans, header_kmeans_result
from miqkmeans import MIQKMeans, header_miqkmeans_result
from handle_data import get_dataset
from utils import *

DATASET = 1     # 1: Synthetic 1, 2: Synthetic 2, 3: Heart Disease, 4: Coverage Type
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

    sizes = [20, 40, 60, 80, 100, 120]

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


def test_features():
    data, k = get_dataset(DATASET)

    features = [2, 4, 6, 8, 10]

    kmeans_losses = []
    kmeans_runtimes = []
    miq_losses = []
    miq_runtimes = []
    for n_features in features:
        cur_data = data.sample(frac=1)
        cur_data = cur_data.iloc[:40, :n_features]
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


def test_centers():
    data, k = get_dataset(DATASET)

    centers = [2, 4, 6, 8, 10]

    kmeans_losses = []
    kmeans_runtimes = []
    miq_losses = []
    miq_runtimes = []
    for n_centers in centers:
        cur_data = data.sample(frac=1)
        cur_data = cur_data.iloc[:40, :4]
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


def test_multiple_inits():
    data, k = get_dataset(DATASET)
    centers = [3]
    sizes = [40, 60, 80, 100]
    kmeans_losses = []
    kmeans_runtimes = []
    miq_losses = []
    miq_runtimes = []
    vars = []
    for n_centers in centers:
        for size in sizes:
            n_vars = size * n_centers
            vars.append(n_vars)
            cur_data = data.sample(frac=1)
            cur_data = cur_data.iloc[:size, :4]
            k = n_centers
            kmeans_min_loss = math.inf
            kmeans_min_runtime = math.inf
            for i in range(10):
                kmeans_clusters, kmeans_loss, kmeans_runtime = kmeans(cur_data, k)
                if kmeans_loss < kmeans_min_loss:
                    kmeans_min_loss = kmeans_loss
                    kmeans_min_runtime = kmeans_runtime
            miq_clusters, miq_loss, miq_runtime = miq_kmeans(cur_data, k, 1000)

            print(f"\n**TEST: SIZE {size}")
            print(f"KMEANS LOSS: {kmeans_min_loss}")
            print(f"KMEANS RUNTIME: {kmeans_min_runtime}")
            print(f"MIQKMEANS LOSS: {miq_loss}")
            print(f"MIQKMEANS RUNTIME: {miq_runtime}")

            kmeans_losses.append(kmeans_min_loss)
            kmeans_runtimes.append(kmeans_min_runtime)
            miq_losses.append(miq_loss)
            miq_runtimes.append(miq_runtime)

            copy_csv('results/results_MIQKMEANS.csv', 'results/csv/results_MIQKMEANS_mult{}'.format(n_vars))

    plot_multiple_inits(vars, kmeans_losses, kmeans_runtimes)


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
    elif TEST == 4:
        test_multiple_inits()


if __name__ == '__main__':
    main()
