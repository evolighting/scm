import numpy as np
from scipy import stats
from sklearn import linear_model
import modshogun
from sklearn.model_selection import train_test_split


def cor_ndarr0(xm, ym, m):
    # 更加“友好”的版本

    # 为了表现的一致性，np.copy是必要的？
    ym = np.copy(ym)
    xm = np.copy(xm)
    xh, xw = xm.shape
    yh, yw = ym.shape
    pwr = np.zeros((xh, yh))
    if xw != yw:
        raise Exception("Data not Match")
    for i in range(xh):
        x_i = xm[i, :]
        for j in range(yh):
            y_i = ym[j, :]
            cor = m(x_i, y_i)[0]
            if np.isnan(cor):
                pwr[i, j] = 0
                raise Exception("get nan!")
            else:
                pwr[i, j] = cor
    return pwr


def cor_ndarr(xm, ym, m):
    # 成对计算xm和ym中向量的相关性质.必须有相同的第二个维度
    # m为接受两个向量返回一个"距离"的函数

    # 为了表现的一致性，np.copy是必要的？
    ym = np.copy(ym)
    xm = np.copy(xm)

    def alx(x):
        def aly(y):
            cor = m(x, y)[0]
            if np.isnan(cor):
                print("get nan!")
                return 0
            else:
                return cor

        return np.apply_along_axis(aly, 1, ym)
    return np.apply_along_axis(alx, 1, xm)


class lm(object):
    def __init__(self, x, y):
        from sklearn import linear_model
        regr = linear_model.LinearRegression()
        regr.fit(x, y)
        self.residuals = y - regr.predict(x)

    def get_residuals(self):
        return np.copy(self.residuals).flatten()


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
    # 线性模型其实是都要取对数的（针对表达量小的情况）
    fit = lm(np.log10(dropouts), GiCsum)
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
    return (s_features.flatten(), s_scores.flatten())


class KMeans(object):
    # test shogun kmean for cosine
    def __init__(self, k, distance):
        self.k = k
        self.distance = distance

    def fit(self, x):
        x = np.array(x).T
        features_train = modshogun.RealFeatures(x)
        distance = self.distance(features_train, features_train)
        self.kmeans = modshogun.KMeans(self.k, distance)
        self.kmeans.train()
        self.cluster_centers_ = self.kmeans.get_cluster_centers().T
        kcc = modshogun.RealFeatures(self.cluster_centers_.T)
        discc = self.distance(kcc, features_train).get_distance_matrix()
        self.labels_ = np.copy(discc.argsort(axis=0)[0, :]).T
        return self

    def fit_predict(self, x):
        features_train = modshogun.RealFeatures(x)
        kcc = RealFeatures(self.cluster_centers_)
        discc = self.distance(kcc, features_train).get_distance_matrix()
        return np.copy(discc.argsort(axis=0)[0, :]).T


class cosKMeans(KMeans):
    def __init__(self, k):
        self.k = k
        self.distance = modshogun.CosineDistance


def get_CosineDistance(xm, ym):
    # CosineDistance by shogun
    xm = np.array(xm).T
    ym = np.array(ym).T
    fxm = modshogun.RealFeatures(xm)
    fym = modshogun.RealFeatures(ym)
    return modshogun.CosineDistance(fxm, fym).get_distance_matrix()


def check_nrom(x, axis):
    # get l2 norm info
    sq_norm = np.sum(x**2, axis=axis)
    print(sq_norm)
    print(sq_norm.shape)


def tts_data(sco):
    # train_test_split
    xi = np.arange(sco.cell_num)
    y = sco.labels[0]
    i_train, i_test, y_train, y_test = train_test_split(xi, y, test_size=0.33)
    bs_train = scm.basicScObject(
        sco.expression_matrix[:, i_train], sco.gene_list,
        sco.cell_list[i_train], sco.labels[:, i_train])
    bs_test = scm.basicScObject(
        sco.expression_matrix[:, i_test], sco.gene_list,
        sco.cell_list[i_test], sco.labels[:, i_test])
    return (bs_train, bs_test)
