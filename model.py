'''
@authors
Lorenzo Baiardi & Thomas Del Moro
'''
import math

import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt

from gurobipy import Model, GRB, LinExpr, QuadExpr, norm


def l2_dist(x, y):
    return np.linalg.norm(x - y)


class KMeans:
    def __init__(self, name, data, k):
        self.centroids = None
        self.data = data  # dataset
        self.k = k  # classes
        self.n = data.shape[0]
        self.timeout = 300
        self.indicators = dict()  # indicator variable of data point being associated with cluster
        self.model = self.create_model(name)  # gurobi model

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
            for k in range(self.k):
                obj_dist.append(distances[i][k])
                obj_indic.append(self.indicators[(i, k)])
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
        centroids = [[] for j in range(self.k)]
        for k in range(self.k):
            sdx = 0
            sd = 0
            for i in range(self.n):
                sdx += indicator[i][k] * self.data.iloc[i, :]
                sd += indicator[i][k]
            centroids[k] = sdx / sd
        self.centroids = centroids
        return centroids

    def solve(self):
        epsilon = 1e-4 * self.data.shape[1] * self.k
        shift = math.inf  # centroid shift
        objective_values = []
        while shift > epsilon:
            shift = 0
            old_centroids = self.centroids
            new_centroids = self.update()
            if self.model.Status == GRB.OPTIMAL:
                # calculated centroid shift
                for i in range(self.k):
                    print('old ', old_centroids[i])
                    print('new ', new_centroids[i])
                    shift += l2_dist(old_centroids[i], new_centroids[i])
                    print('shift ', shift)

        clusters = [-1 for i in range(self.n)]
        for i in range(self.n):
            for k in range(self.k):
                if self.indicators[(i, k)].x > 0.5:
                    clusters[i] = k
        return clusters


class MIQKMeans:
    def __init__(self, name, data, k):
        self.data = data  # dataset
        self.k = k  # classes
        self.n = data.shape[0]
        self.N = data.shape[1]
        self.bigM = 1e100
        self.timeout = 3000
        self.centroids = dict()
        self.indicators = dict()  # indicator variable of data point being associated with cluster
        self.vars = dict()
        self.vars_norms = dict()
        self.model = self.create_model(name)  # gurobi model

    def create_model(self, name):
        model = Model(name)
        c = self.n / (2 * self.k)  # minimum number of instances contained in each cluster

        # add indicator variables for every data and class
        for i in range(self.n):
            for k in range(self.k):
                self.indicators[(i, k)] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY)

        # constraint element belonging to a unique cluster
        for i in range(self.n):
            expr = LinExpr([1] * self.k, [self.indicators[(i, k)] for k in range(self.k)])
            model.addConstr(expr, GRB.EQUAL, 1.0, 'c%d' % i)

        # constraint minimum of clusters
        for k in range(self.k):
            expr = LinExpr([1] * self.n, [self.indicators[(i, k)] for i in range(self.n)])
            model.addConstr(expr, GRB.GREATER_EQUAL, c, 's%d' % i)

        for i in range(self.n):
            for k in range(self.k):
                self.vars[(i, k)] = model.addVars(self.N, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
                self.vars_norms[(i, k)] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
                model.addGenConstrNorm(self.vars_norms[(i, k)], self.vars[(i, k)], 2.0, "normconstr")

        for k in range(self.k):
            self.centroids[k] = model.addVars(self.N, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)

        # constraints on distances
        for k in range(self.k):
            for i in range(self.n):
                for j in range(self.N):
                    model.addConstr(self.vars[(i, k)][j] >= - self.bigM * (1 - self.indicators[(i, k)]) + (
                                self.data.iloc[i, j] - self.centroids[k][j]), '1sconstr')
                    model.addConstr(self.vars[(i, k)][j] <= self.bigM * (1 - self.indicators[(i, k)]) + (
                                self.data.iloc[i, j] - self.centroids[k][j]), '2sconstr')

        self.model = model
        return model

    def set_objective(self):
        obj_coeff = [1] * (self.n * self.k)
        obj_vars = []
        for i in range(self.n):
            for k in range(self.k):
                obj_vars.append(self.vars_norms[(i, k)])
        self.model.setObjective(QuadExpr(LinExpr(obj_coeff, obj_vars)), GRB.MINIMIZE)
        self.model.update()

    def solve(self):
        self.set_objective()
        self.model.Params.TimeLimit = self.timeout
        self.model.Params.NumericFocus = 3
        #self.model.Params.ScaleFlag = 3
        self.model.Params.Presolve = 0
        self.model.Params.NonConvex = 2
        self.model.optimize()

        if self.model.Status == GRB.OPTIMAL:
            clusters = [-1 for i in range(self.n)]
            for i in range(self.n):
                for k in range(self.k):
                    if self.indicators[(i, k)].x > 0.5:
                        clusters[i] = k

        for i in range(self.n):
            for j in range(self.N):
                for k in range(self.k):
                    print(self.vars[(i, k)][j].x)
                    # print(self.centroids[k][j].x)
                    # print(self.indicators[(i, k)].x)

        return clusters, self.centroids
