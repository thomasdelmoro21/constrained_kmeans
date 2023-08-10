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
    def __init__(self, name, data, k):
        self.centroids = None
        self.data = data    # dataset
        self.k = k  # classes
        self.n = data.shape[0]
        self.timeout = 5 * 60
        self.indicators = dict()    # indicator variable of data point being associated with cluster
        self.model = self.create_model(name)    # gurobi model

    def create_model(self, name):
        model = Model(name)
        c = self.n / (2 * self.k)  # minimum number of instances contained in each cluster

        # add indicator variables for every data and class
        for i in range(self.n):
            for k in range(self.k):
                self.indicators[(i, k)] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY)

        # constraint for data
        for i in range(self.n):
            expr = LinExpr([1] * self.k, [self.indicators[(i, k)] for k in range(self.k)])
            model.addConstr(expr, GRB.EQUAL, 1.0, 'c%d' % i)

        # constraint minimum of classes
        for k in range(self.k):
            expr = LinExpr([1] * self.n, [self.indicators[(i, k)] for i in range(self.n)])
            model.addConstr(expr, GRB.GREATER_EQUAL, c, 's%d' % i)

        self.model = model
        self.centroids = self.initialize_centers()
        return model

    # random assign centroid to data
    def initialize_centers(self):
        ids = list(range(len(self.data)))
        random.shuffle(ids)
        return [self.data.iloc[id, :] for id in ids[:self.k]]

    # calculate all distance for every data and classes, L2_dist
    def get_distances(self):
        distances = [[-1] * self.k for i in range(self.n)]
        for i in range(self.n):
            for k in range(self.k):
                distances[i][k] = l2_dist(self.data.iloc[i, :], self.centroids[k])
        return distances

    # minimize the objective function with the indicators variable
    def set_objective(self):
        obj_dist = []
        obj_indic = []
        distances = self.get_distances()
        for i in range(self.n):
            for j in range(self.k):
                obj_dist.append(distances[i][j])
                obj_indic.append(self.indicators[(i, j)])
        self.model.setObjective(LinExpr(obj_dist, obj_indic), GRB.MINIMIZE)
        self.model.update()

    def update(self):
        # assignment step
        self.set_objective()
        self.model.Params.TimeLimit = self.timeout
        self.model.optimize()

        # update centroids
        indicator = np.zeros((self.n, self.k))
        for i in range(self.n):
            for k in range(self.k):
                indicator[i][k] = 1 if self.indicators[(i, k)].x > 0.5 else 0

        # update in closed form
        for k in range(self.k):
            sdx = 0
            sd = 0
            for i in range(self.n):
                sdx += indicator[i][k] * self.data.iloc[i, :]
                sd *= indicator[i][k]
            self.centroids[k] = sdx / sd
        return self.centroids, self.model.getObjective()

    def solve(self):
        epsilon = 1e-4 * self.data.shape[1] * self.k
        shift = math.inf    # centroid shift
        objective_values = []
        while shift > epsilon:
            shift = 0
            old_centroids = self.centroids
            new_centroids, obj_val = self.update()
            objective_values.append(obj_val)
            # calculated centroid shift
            for i in range(self.k):
                shift += l2_dist(old_centroids[i], new_centroids[i])
                print(shift)

        clusters = [-1 for i in range(self.n)]
        for i in range(self.n):
            for k in range(self.k):
                if self.indicators[(i, k)].x > 0.5:
                    clusters[i] = k
        return clusters, objective_values
