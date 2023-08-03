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

class KMeans:
    def __init__(self, name, data, K):
        self.centroids = None
        self.data = data
        self.K = K
        self.N = data.shape[0]
        self.model = self.create_model(name)
        self.delta = dict()

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

    def optimize(self):     #assignment step
        self.setObjective()
        self.model.optimize()

        clusters = None
        if self.model.Status == GRB.OPTIMAL:
            clusters = [-1 for i in range(self.N)]
            for i in range(self.N):
                for k in range(self.K):
                    if self.delta[(i, k)].x > 0.5:
                        clusters[i] = k
        return clusters
