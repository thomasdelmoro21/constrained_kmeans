'''
@authors
Lorenzo Baiardi & Thomas Del Moro
'''
import math

import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt

from gurobipy import Model, GRB, LinExpr


def l2_dist(x, y):
    return np.linalg.norm(x - y)


class KMeans:
    def __init__(self, name, data, K):
        self.centroids = None
        self.data = data
        self.K = K
        self.N = data.shape[0]
        self.timeout = 5*60
        self.delta = dict()
        self.model = self.create_model(name)

    def create_model(self, name):
        model = Model(name)
        C = self.N / (2 * self.K)  # minimum number of instances contained in each cluster
        for i in range(self.N):
            for k in range(self.K):
                self.delta[(i, k)] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY)

        for i in range(self.N):
            expr = LinExpr([1] * self.K, [self.delta[(i, k)] for k in range(self.K)])
            model.addConstr(expr, GRB.EQUAL, 1.0, 'c%d' % i)

        for k in range(self.K):
            expr = LinExpr([1] * self.N, [self.delta[(i, k)] for i in range(self.N)])
            model.addConstr(expr, GRB.GREATER_EQUAL, C, 's%d' % i)

        self.model = model
        self.centroids = self.inizialize_centers()
        return model

    def inizialize_centers(self):
        ids = list(range(len(self.data)))
        random.shuffle(ids)
        return [self.data.iloc[id, :] for id in ids[:self.K]]

    def getDistances(self):
        distances = [[-1] * self.K for i in range(self.N)]
        for i in range(self.N):
            for k in range(self.K):
                distances[i][k] = l2_dist(self.data.iloc[i, :], self.centroids[k])
        return distances

    def setObjective(self):
        obj_dist = []
        obj_delta = []
        distances = self.getDistances()
        for i in range(self.N):
            for j in range(self.K):
                obj_dist.append(distances[i][j])
                obj_delta.append(self.delta[(i, j)])
        self.model.setObjective(LinExpr(obj_dist, obj_delta), GRB.MINIMIZE)
        self.model.update()

    def update(self):
        # assignment step
        self.setObjective()
        self.model.Params.TimeLimit = self.timeout
        self.model.optimize()

        # update centroids
        delta = np.zeros((self.N, self.K))
        for i in range(self.N):
            for k in range(self.K):
                delta[i][k] = 1 if self.delta[(i, k)].x > 0.5 else 0
        # update in closed form
        for k in range(self.K):
            sdx = 0
            sd = 0
            for i in range(self.N):
                sdx += delta[i][k] * self.data.iloc[i, :]
                sd *= delta[i][k]
            self.centroids[k] = sdx / sd
        return self.centroids, self.model.getObjective()

    def solve(self):
        epsilon = 1e-4 * self.data.shape[1] * self.K
        shift = math.inf
        objective_values = []
        while shift > epsilon:
            shift = 0
            old_centroids = self.centroids
            new_centroids, obj_val = self.update()
            objective_values.append(obj_val)
            for i in range(self.K):
                shift += l2_dist(old_centroids[i], new_centroids[i])
                print(shift)

        clusters = [-1 for i in range(self.N)]
        for i in range(self.N):
            for k in range(self.K):
                if self.delta[(i, k)].x > 0.5:
                    clusters[i] = k
        return clusters, objective_values
