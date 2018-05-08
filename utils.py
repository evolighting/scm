import numpy as np
from scipy import stats
from sklearn import linear_model


class lm(object):
    def __init__(self, x, y):
        from sklearn import linear_model
        regr = linear_model.LinearRegression()
        regr.fit(x, y)
        self.residuals = y - regr.predict(x)

    def get_residuals(self):
        return self.residuals


class lm1(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.slope, self.intercept, self.r_value, \
            self.p_value, self.std_err = stats.linregress(x, y)

    def get_residuals(self):
        return self.y - (self.x * self.slope + self.intercept)


def linear_model(counts, n_features):
    h, w = counts.shape
    bm = counts == 0
    dropouts = bm.sum(axis=1) / w * 100
    # 去除spike-in和无drop out现象
    # 暂时不管spikes
    dropouts_filter = (dropouts != 0) & (dropouts != 100)
    dropouts_filter = np.asarray(dropouts_filter)[:, 0]
    counts = counts[dropouts_filter, :]
    dropouts = dropouts[dropouts_filter]
    GiCsum = counts.sum(axis=1)

    fit = lm(dropouts, GiCsum)
    residuals = fit.get_residuals()

    r_sort_ind = np.argsort(-residuals)[:n_features]

    s_features = np.arange(h)[dropouts_filter][r_sort_ind]
    s_scores = residuals[r_sort_ind]
    return (s_features, s_scores)


def random(counts, n_features=500):
    h, w = counts.shape
    s_features = np.random.shuffle(np.arange(h))[:n_features]
    s_scores = np.repeat(1, n_features)
    return (s_features, s_scores)


def select_genes(counts, n_features=500, model=linear_model):
    (s_features, s_scores) = model(counts, n_features)
    # selected_data = counts[s_features, :]
    return (s_features[:, 0], s_scores)


##
# 基因参考集合
