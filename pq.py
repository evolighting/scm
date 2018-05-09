import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
# product quantization


class pqNN(object):
    def __init__(self, m=None, k=None):
        self.m = m
        self.k = k

    def fit(self, X):
        """
        X: n * f data
        n: sample number
        f: feature number
        """
        self.n, self.f = X.shape
        if self.m is None:
            self.m = min(np.int(np.ceil(self.f / 10)), 100)
        if self.k is None:
            self.k = max(np.int(np.sqrt(self.n)), 2)
        # split feature space
        chunks_i = [int(i * self.f / (self.m)) for i in range(self.m + 1)]
        self.ci_list = \
            [np.arange(chunks_i[i], chunks_i[i + 1]) for i in range(self.m)]
        self.cluster_centers = [None] * self.m
        self.predict_labels = [None] * self.m
        for i in range(self.m):
            x_i = X[:, self.ci_list[i]]
            x_i = normalize(x_i, axis=1)
            km = KMeans(n_clusters=self.k, n_jobs=-2)
            km.fit(x_i)
            self.predict_labels[i] = km.labels_
            self.cluster_centers[i] = normalize(km.cluster_centers_, axis=1)

    def get_knn(self, qx, k=3):
        # qx should be 1D array
        qx = np.array(qx)
        if len(qx.shape) != 1:
            raise Exception("Invalid input")
        if len(qx) != self.f:
            raise Exception("Invalid input")
        cosm = [None] * self.m
        for mi in range(self.m):
            qx_i = qx[self.ci_list[mi]]
            cc_i = self.cluster_centers[mi]
            pl_i = self.predict_labels[mi]
            d_i = np.zeros(self.k)
            for mj in range(self.k):
                d_i[mj] = np.dot(qx_i, cc_i[mj, :])
            cosm[mi] = d_i

        dist = np.zeros(self.n)
        for ni in range(self.n):
            for nj in range(self.m):
                k_i = self.predict_labels[nj][ni]
                dist[ni] += cosm[nj][k_i]

        di = (-dist).argsort()
        return di[:k]


class pqIndex(object):
    def __init__(self, x, m, k):
        """
        more coooool version?
        x: n * f data
        n: sample number
        f: feature number
        """
        self.n, self.f = x.shape
        if self.m is None:
            self.m = min(np.int(np.ceil(self.f / 10)), 100)
        if self.k is None:
            self.k = max(np.int(np.sqrt(self.n)), 2)
        # split feature space
        chunks_i = [int(i * self.f / (self.m)) for i in range(self.m + 1)]
        self.ci_list = \
            [np.arange(chunks_i[i], chunks_i[i + 1]) for i in range(self.m)]
        self.cluster_centers = [None] * self.m
        self.predict_labels = [None] * self.m
        for i in range(self.m):
            x_i = x[:, self.ci_list[i]]
            x_i = normalize(x_i, axis=1)
            km = KMeans(n_clusters=self.k, n_jobs=-2)
            km.fit(x_i)
            self.predict_labels[i] = km.labels_
            self.cluster_centers[i] = normalize(km.cluster_centers_, axis=1)
