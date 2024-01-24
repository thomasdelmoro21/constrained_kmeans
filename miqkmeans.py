"""
@authors
Lorenzo Baiardi & Thomas Del Moro
"""

from gurobipy import Model, LinExpr, GRB, QuadExpr
from timeit import default_timer as timer


class MIQKMeans:
    def __init__(self, name, data, k, timelimit):
        self.data = data  # dataset
        self.k = k  # classes
        self.n = data.shape[0]  # number of elements
        self.N = data.shape[1]  # number of features
        self.bigM = 500
        self.timeout = timelimit
        self.centroids = dict()  # centroids of clusters
        self.indicators = dict()  # indicator variable of data point being associated with cluster
        self.vars = dict()  # residual variables per component
        self.vars_norms = dict()  # residual variables to be minimize
        self.model = self.create_model(name)  # gurobi model

    def create_model(self, name):
        model = Model(name)
        c = self.n / (2 * self.k)  # minimum number of instances contained in each cluster

        # add indicator variables for every data and class
        for i in range(self.n):
            for k in range(self.k):
                self.indicators[(i, k)] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY)

        # constraint of element belonging to a unique cluster
        for i in range(self.n):
            expr = LinExpr([1] * self.k, [self.indicators[(i, k)] for k in range(self.k)])
            model.addConstr(expr, GRB.EQUAL, 1.0, 'c%d' % i)

        # constraint of minimum of clusters
        for k in range(self.k):
            expr = LinExpr([1] * self.n, [self.indicators[(i, k)] for i in range(self.n)])
            model.addConstr(expr, GRB.GREATER_EQUAL, c, 's%d' % k)

        # add residual variables and their norms
        for i in range(self.n):
            for k in range(self.k):
                self.vars[(i, k)] = model.addVars(self.N, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
                self.vars_norms[(i, k)] = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
                model.addGenConstrNorm(self.vars_norms[(i, k)], self.vars[(i, k)], 2.0, "normconstr%d%d" % (i, k))

        # add centroids variables
        for k in range(self.k):
            self.centroids[k] = model.addVars(self.N, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)

        # constraints on residuals
        for k in range(self.k):
            for i in range(self.n):
                for j in range(self.N):
                    model.addConstr(self.vars[(i, k)][j] >= - self.bigM * (1 - self.indicators[(i, k)]) + (
                            self.data.iloc[i, j] - self.centroids[k][j]), "sl%d%d%d" % (j, i, k))
                    model.addConstr(self.vars[(i, k)][j] <= self.bigM * (1 - self.indicators[(i, k)]) + (
                            self.data.iloc[i, j] - self.centroids[k][j]), "su%d%d%d" % (j, i, k))
        self.model = model
        return model

    def set_objective(self):
        obj_coeffs = [1] * (self.n * self.k)
        obj_vars = []
        for i in range(self.n):
            for k in range(self.k):
                obj_vars.append(self.vars_norms[(i, k)])
        self.model.setObjective(QuadExpr(LinExpr(obj_coeffs, obj_vars)), GRB.MINIMIZE)
        self.model.update()

    def solve(self):
        start_time = timer()
        runtime = None
        self.model.update()
        self.set_objective()
        self.model.Params.TimeLimit = self.timeout
        self.model.Params.MIPGap = 1e-2
        self.model.optimize(data_cb)

        runtime = timer() - start_time
        clusters = [-1 for i in range(self.n)]
        for i in range(self.n):
            for k in range(self.k):
                if self.indicators[(i, k)].x > 0.5:
                    clusters[i] = k

        return clusters, self.model.objVal, runtime


def header_miqkmeans_result():
    with open("results/results_MIQKMEANS.csv", 'w') as f:
        f.write("TIME;OBJ;BOUND\n")


def export_miqkmeans_data(time, cur_obj, cur_bd):
    with open("results/results_MIQKMEANS.csv", 'a') as f:
        f.write(f"{time};{cur_obj};{cur_bd}\n")


def data_cb(model, where):
    if where == GRB.Callback.MIP:
        time = model.cbGet(GRB.Callback.RUNTIME)
        cur_obj = model.cbGet(GRB.Callback.MIP_OBJBST)
        cur_bd = model.cbGet(GRB.Callback.MIP_OBJBND)
        export_miqkmeans_data(time, cur_obj, cur_bd)




