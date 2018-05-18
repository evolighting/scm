import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import euclidean_distances
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
    def __init__(self, x, m=None, k=None):
        """
        more compact version?
        x: n * f data
        n: sample number
        f: feature number
        """
        self.m, self.k = m, k
        self.n, self.f = x.shape
        if self.m is None:
            self.m = min(np.int(np.ceil(self.f / 10)), 100)
        if self.k is None:
            self.k = max(np.int(np.sqrt(self.n)), 2)
        # split feature space
        chunks_i = [int(i * self.f / (self.m)) for i in range(self.m + 1)]
        self.ci_list = \
            [np.arange(chunks_i[i], chunks_i[i + 1]) for i in range(self.m)]
        self.cluster_centers = np.zeros((self.f, self.k))
        self.predict_labels = np.zeros((self.m, self.n), dtype='int')
        for i in range(self.m):
            x_i = x[:, self.ci_list[i]]
            x_i = normalize(x_i, axis=1)
            km = KMeans(n_clusters=self.k, n_jobs=-2)
            km.fit(x_i)
            self.predict_labels[i, :] = km.labels_
            self.cluster_centers[self.ci_list[i], :] = \
                normalize(km.cluster_centers_, axis=1).T

    def get_knn(self, x, k=3):
        h, w = x.shape
        if k > self.n or w > self.f:
            raise Exception("Invalid input")
        cosm = np.zeros((h, self.n))
        dist = np.zeros((self.m, h, self.k))

        sq_norm = sq_norm = np.sum(x**2, axis=1)

        for mi in range(self.m):
            qx_i = x[:, self.ci_list[mi]]
            cc_i = self.cluster_centers[self.ci_list[mi], :].T
            pl_i = self.predict_labels[mi]
            dist[mi, :, :] = qx_i.dot(cc_i.T)

        for ni in range(self.m):
            k_i = self.predict_labels[ni, :]
            cosm += dist[ni, :, :][:, k_i]
        cosm = cosm / np.sqrt(sq_norm[:, np.newaxis] * self.m)

        di = (-cosm).argsort()
        return (di[:, :k], cosm[np.arange(h)[:, np.newaxis], di[:, :k]])

    def get_knn_filtered(self, x, filter, k=3):
        # in case of missing ,should add 0 to input
        h, w = x.shape
        x_e = np.zeros((h, self.f))
        x_e[:, filter] = x
        return self.get_knn(x_e, k=3)


class pqIndex_Euc(object):
    def __init__(self, x, m=None, k=None):
        """
        more compact version, with "p2 distance"
        x: n * f data
        n: sample number
        f: feature number
        """
        self.metrics = "Euc"
        self.m, self.k = m, k
        self.n, self.f = x.shape
        if self.m is None:
            self.m = min(np.int(np.ceil(self.f / 10)), 100)
        if self.k is None:
            self.k = max(np.int(np.sqrt(self.n)), 2)
        # split feature space
        chunks_i = [int(i * self.f / (self.m)) for i in range(self.m + 1)]
        self.ci_list = \
            [np.arange(chunks_i[i], chunks_i[i + 1]) for i in range(self.m)]
        self.cluster_centers = np.zeros((self.f, self.k))
        self.predict_labels = np.zeros((self.m, self.n), dtype='int')
        for i in range(self.m):
            x_i = x[:, self.ci_list[i]]
            km = KMeans(n_clusters=self.k, n_jobs=-2)
            km.fit(x_i)
            self.predict_labels[i, :] = km.labels_
            self.cluster_centers[self.ci_list[i], :] = \
                km.cluster_centers_.T

    def get_knn(self, x, k=3):
        h, w = x.shape
        if k > self.n or w > self.f:
            raise Exception("Invalid input")
        cosm = np.zeros((h, self.n))
        dist = np.zeros((self.m, h, self.k))

        for mi in range(self.m):
            qx_i = x[:, self.ci_list[mi]]
            cc_i = self.cluster_centers[self.ci_list[mi], :].T
            pl_i = self.predict_labels[mi]
            d_v = euclidean_distances(qx_i, cc_i)
            dist[mi, :, :] = d_v**2
        for ni in range(self.m):
            k_i = self.predict_labels[ni, :]
            cosm += dist[ni, :, :][:, k_i]
        cosm = np.sqrt(cosm)

        di = cosm.argsort()
        return (di[:, :k], cosm[np.arange(h)[:, np.newaxis], di[:, :k]])

    def get_knn_filtered(self, x, filter, k=3):
        # incase of missing ,should add 0 to input
        h, w = x.shape
        x_e = np.zeros((h, self.f))
        x_e[:, filter] = x
        return self.get_knn(x_e, k=3)


class pqIndex_supervised(object):
    def __init__(self, x, y, m=None, k=None):
        """
        more compact version, with "p2 distance"
        x: n * f data
        y: labels for x
        n: sample number
        f: feature number
        """
        self.metrics = "Euc"
        self.m, self.k = m, k
        self.n, self.f = x.shape
        if self.m is None:
            self.m = min(np.int(np.ceil(self.f / 10)), 100)
        if self.k is None:
            self.k = max(np.int(np.sqrt(self.n)), 2)
        # split feature space
        chunks_i = [int(i * self.f / (self.m)) for i in range(self.m + 1)]
        self.ci_list = \
            [np.arange(chunks_i[i], chunks_i[i + 1]) for i in range(self.m)]
        self.predict_labels = np.zeros((self.m, self.n), dtype='int')\

        # kmean group by labels
        ck = self.n / self.k

        def split_by_labels(y, label):
            label_i = y == label
            label_num = label_i.sum()
            # k_i 中，每个组最小分组值也许会有影响
            k_i = max(np.int(label_num / ck), 1)
            return(label_i, k_i)

        label_i_list = [split_by_labels(y, i) for i in np.unique(y)]
        self.k = np.sum([i[1] for i in label_i_list])
        self.cluster_centers = np.zeros((self.f, self.k))

        for i in range(self.m):
            x_i = x[:, self.ci_list[i]]

            cluster_centers_i = np.zeros((len(self.ci_list[i]), self.k))     

            pl = 0
            for j, k in label_i_list:
                km = KMeans(n_clusters=k)
                x_ij = x_i[j, :]
                km.fit(x_ij)
                self.predict_labels[i, j] = km.labels_ + pl
                cluster_centers_i[:, np.arange(pl, pl + k)] = \
                    km.cluster_centers_.T
                pl += k
            self.cluster_centers[self.ci_list[i], :] = cluster_centers_i
        if self.k != pl:
            raise Exception("kmean error")
        print(self.cluster_centers.shape)

    def get_knn(self, x, k=3):
        h, w = x.shape
        if k > self.n or w > self.f:
            raise Exception("Invalid input")
        cosm = np.zeros((h, self.n))
        dist = np.zeros((self.m, h, self.k))

        for mi in range(self.m):
            qx_i = x[:, self.ci_list[mi]]
            cc_i = self.cluster_centers[self.ci_list[mi], :].T 
            pl_i = self.predict_labels[mi]
            d_v = euclidean_distances(qx_i, cc_i)
            dist[mi, :, :] = d_v**2
        for ni in range(self.m):
            k_i = self.predict_labels[ni, :]
            cosm += dist[ni, :, :][:, k_i]
        cosm = np.sqrt(cosm)

        di = cosm.argsort()
        return (di[:, :k], cosm[np.arange(h)[:, np.newaxis], di[:, :k]])
