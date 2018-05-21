import numpy as np
from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections


# 使用LSHForest进行nn搜索的尝试

# 兼容实现qp的一些功能, get_knn


class scLSH(object):
    def __init__(self, x):
        self.n, self.f = x.shape
        # Use NearPy lsh for fast ann
        rbp = RandomBinaryProjections('rbp', 10)

        self.engine = Engine(self.f, lshashes=[rbp])
        for i in np.arange(self.n):
            v = x[i, :]
            self.engine.store_vector(v, i)

    def get_one_knn(self, v, k=3):
        vl = v.shape
        if vl[0] != self.f:
            print(vl)
            raise Exception("Data Not Match")
        N = self.engine.neighbours(v)
        nni = -np.ones(k, dtype='int')
        nnd = np.empty(k)
        nnd[:] = np.nan
        for i in np.arange(k):
            try:
                nni[i] = N[i][1]
                nnd[i] = N[i][2]
            except IndexError:
                break
        return (nni, nnd)

    def get_knn(self, x, k=3):
        self.n, self.f = x.shape
        nni = -np.ones((self.n, k), dtype='int')
        nnd = np.empty((self.n, k))
        nnd[:] = np.nan
        for i in np.arange(self.n):
            i_i, i_d = self.get_one_knn(x[i, :])
            nni[i, :] = i_i
            nnd[i, :] = i_d
        return (nni, nnd)

