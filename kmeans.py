"""
@authors
Lorenzo Baiardi & Thomas Del Moro
"""

import numpy as np
import random
from gurobipy import *
from timeit import default_timer as timer


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
            model.addConstr(expr, GRB.GREATER_EQUAL, c, 's%d' % k)

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
        return centroids, self.model.ObjVal

    def solve(self):
        #epsilon = 1e-5 * self.data.shape[0] * self.data.shape[1] * self.k
        epsilon = 0.001
        shift = math.inf  # centroid shift
        cur_objective = None
        start_time = timer()
        runtime = None
        while shift > epsilon:
            shift = 0
            old_centroids = self.centroids
            cur_centroids, cur_objective = self.update()
            runtime = timer() - start_time
            export_kmeans_data(runtime, cur_objective)
            if self.model.Status == GRB.OPTIMAL:
                # calculated centroid shift
                for i in range(self.k):
                    print('old ', old_centroids[i])
                    print('new ', cur_centroids[i])
                    shift += l2_dist(old_centroids[i], cur_centroids[i])
                    print('shift ', shift)

        clusters = [-1 for _ in range(self.n)]
        for i in range(self.n):
            for k in range(self.k):
                if self.indicators[(i, k)].x > 0.5:
                    clusters[i] = k
        return clusters, cur_objective, runtime


def l2_dist(x, y):
    return np.linalg.norm(x - y)


def header_kmeans_result():
    with open("results/results_KMEANS.csv", 'w') as f:
        f.write("TIME;OBJ\n")


def export_kmeans_data(time, cur_obj):
    with open("results/results_KMEANS.csv", 'a') as f:
        f.write(f"{time};{cur_obj}\n")


def data_cb(model, where):
    if where == GRB.Callback.MIP:
        time = model.cbGet(GRB.Callback.RUNTIME)
        cur_obj = model.cbGet(GRB.Callback.MIP_OBJBST)
        cur_bd = model.cbGet(GRB.Callback.MIP_OBJBND)
        export_kmeans_data(time, cur_obj)


