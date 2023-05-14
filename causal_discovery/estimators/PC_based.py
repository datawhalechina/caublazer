# **
# * @author yunfeng
# * @edit yunfeng

import datetime
import gc
import time
import numpy as np
import pandas as pd
from .base import base_discover
from itertools import combinations
from scipy.stats import norm
import math
from typing import List

class PC_class():
    def __init__(self):
        super().__init__()

    def get_neighbors(self, G, x: int, y: int):
        return [i for i in range(len(G)) if G[x][i] == True and i != y]

    def gauss_ci_test(self, suff_stat, x: int, y: int, K: List[int], cut_at: float = 0.9999999):
        C = suff_stat["C"]
        n = suff_stat["n"]
        if len(K) == 0:
            r = C[x, y]
        elif len(K) == 1:
            k = K[0]
            r = (C[x, y] - C[x, k] * C[y, k]) / math.sqrt((1 - C[y, k] ** 2) * (1 - C[x, k] ** 2))
        else:
            m = C[np.ix_([x] + [y] + K, [x] + [y] + K)]
            p = np.linalg.pinv(m)
            r = -p[0, 1] / math.sqrt(abs(p[0, 0] * p[1, 1]))
        r = min(cut_at, max(-cut_at, r))

        z = 0.5 * math.log1p((2 * r) / (1 - r))
        z_standard = z * math.sqrt(n - len(K) - 3)
        p_value = 2 * (1 - norm.cdf(abs(z_standard)))

        return p_value

    def skeleton(self, suff_stat, alpha: float):
        n_nodes = suff_stat["C"].shape[0]
        O = [[[] for _ in range(n_nodes)] for _ in range(n_nodes)]
        G = [[i != j for i in range(n_nodes)] for j in range(n_nodes)]
        pairs = [(i, (n_nodes - j - 1)) for i in range(n_nodes) for j in range(n_nodes - i - 1)]
        done = False
        l = 0
        while done != True and any(G):
            done = True
            for x, y in pairs:
                if G[x][y] == True:
                    neighbors = self.get_neighbors(G, x, y)
                    if len(neighbors) >= l:
                        done = False
                        for K in set(combinations(neighbors, l)):
                            p_value = self.gauss_ci_test(suff_stat, x, y, list(K))
                            if p_value >= alpha:
                                G[x][y] = G[y][x] = False
                                O[x][y] = O[y][x] = list(K)
                                break
            l += 1
        return np.asarray(G, dtype=int), O

    def extend_cpdag(self, G, O):
        n_nodes = G.shape[0]
        def rule1(g):
            pairs = [(i, j) for i in range(n_nodes) for j in range(n_nodes) if g[i][j] == 1 and g[j][i] == 0]  # 所有 i - j 点对
            for i, j in pairs:
                all_k = [k for k in range(n_nodes) if (g[j][k] == 1 and g[k][j] == 1) and (g[i][k] == 0 and g[k][i] == 0)]
                if len(all_k) > 0:
                    g[j][all_k] = 1
                    g[all_k][j] = 0
            return g

        def rule2(g):
            pairs = [(i, j) for i in range(n_nodes) for j in range(n_nodes) if g[i][j] == 1 and g[j][i] == 1]  # 所有 i - j 点对
            for i, j in pairs:
                all_k = [k for k in range(n_nodes) if (g[i][k] == 1 and g[k][i] == 0) and (g[k][j] == 1 and g[j][k] == 0)]
                if len(all_k) > 0:
                    g[i][j] = 1
                    g[j][1] = 0

            return g

        def rule3(g):
            pairs = [(i, j) for i in range(n_nodes) for j in range(n_nodes) if g[i][j] == 1 and g[j][i] == 1]  # 所有 i - j 点对
            for i, j in pairs:
                all_k = [k for k in range(n_nodes) if (g[i][k] == 1 and g[k][i] == 1) and (g[k][j] == 1 and g[j][k] == 0)]
                if len(all_k) >= 2:
                    for k1, k2 in combinations(all_k, 2):
                        if g[k1][k2] == 0 and g[k2][k1] == 0:
                            g[i][j] = 1
                            g[j][i] = 0
                            break

            return g

        pairs = [(i, j) for i in range(n_nodes) for j in range(n_nodes) if G[i][j] == 1]
        for x, y in sorted(pairs, key=lambda x:(x[1], x[0])):
            all_z = [z for z in range(n_nodes) if G[y][z] == 1 and z != x]
            for z in all_z:
                if G[x][z] == 0 and y not in O[x][z]:
                    G[x][y] = G[z][y] = 1
                    G[y][x] = G[y][z] = 0

        old_G = np.zeros((n_nodes, n_nodes))
        while not np.array_equal(old_G, G):
            old_G = G.copy()
            G = rule1(G)
            G = rule2(G)
            G = rule3(G)
        return np.array(G)

    def pc(self, suff_stat, alpha: float = 0.05, verbose: bool = False):
        G, O = self.skeleton(suff_stat, alpha)
        cpdag = self.extend_cpdag(G, O)
        if verbose:
            print(cpdag)
        return cpdag


class PC(base_discover):
    def __init__(self, dataset_discover):
        super(PC, self).__init__(dataset_discover)

    def _get_feature_columns(self):
        return self._dataset.get_data_columns()

    def _get_adjacent_matrix(self, method = "PC", random_seed = 0):
        if method == "PC":
            np.set_printoptions(precision=4, suppress=True)
            model = PC_class()
            adjacent_matrix = model.pc(
                suff_stat = { "C": self._dataset.data.corr().values, "n": self._dataset.data.shape[0]},
                verbose = True        
            )
            edge_table = self._get_edge_table(method_name=method, is_sparse=False, edge_data_map=adjacent_matrix)
        else:
            edge_table = []
        return edge_table