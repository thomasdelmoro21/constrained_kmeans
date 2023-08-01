'''
@authors
Lorenzo Baiardi & Thomas Del Moro
'''
import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt

from gurobipy import Model, GRB, LinExpr


def l2_dist(x, y):
    return np.linalg.norm(x - y)


def inizialize_centers(data, K):
    ids = list(range(len(data)))
    random.shuffle(ids)
    return [data[id] for id in ids[:K]]


def create_model(name, data, K):
    model = Model(name)
    N = data.shape[0]  # number of instances in dataset
    delta = dict()
    C = N/(2*K)  #minimum number of instances contained in each cluster
    for i in range(N):
        for k in range(K):
            delta[(i, k)] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY)

    for i in range(N):
        expr = LinExpr([1] * K, [delta[(i, k)] for k in range(K)])
        model.addConstr(expr, GRB.EQUAL, 1.0, 'c%d'%i)

    for k in range(K):
        expr = LinExpr([1] * N, [delta[(i, k)] for i in range(N)])
        model.addConstr(expr, GRB.GREATER_EQUAL, C, 's%d'%i)

    centroids = inizialize_centers(data, K)

    distances = np.zeros((N, K))
    for i in range(N):
        for k in range(K):
            distances[i][k] = l2_dist(data[i], centroids[k])

    model.setObjective(0.5 * (delta * distances), GRB.MINIMIZE)

    return model


def main():
    data = pd.read_csv("./xclara.csv")
    print(data.shape[0], data.shape[1])
    print(data)
    data = data.dropna(how='any')
    v1 = data.iloc[:, 0:1].values
    v2 = data.iloc[:, 1:].values

    plt.scatter(v1, v2)
    plt.show()

    model = create_model("constrained_kmeans", data, K=3)


if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
