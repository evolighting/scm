import numpy as np
from . import utils
from .scr import indexGeneMapper, scBase, basicScObject, scmObject
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr, pearsonr
from sklearn.cluster import KMeans


def cor_ndarr(xm, ym, m):
    # 成对计算xm和ym中向量的相关性质.必须有相同的第二个维度
    # m为接受两个向量返回一个"距离"的函数
    def alx(x):
        def aly(y):
            return m(x, y)[0]
        return np.apply_along_axis(aly, 1, ym)
    return np.apply_along_axis(alx, 1, xm)

class indexCluster(scBase):
    def __init__(self, scmObject):
        self.expression_matrix = scmObject.expression_matrix
        # 如果要对表达量进行处理，应该在之前进行，
        # 这里接受的应该是处理之后的对象，
        # 具体的，认为已经使用过log(x+1)对表达矩阵进行处理
        s_genes_i, self.s_scores = scmObject.select_genes()
        self.s_genes = scmObject.gene_list[s_genes_i]
        x = scmObject.get_selected_expression(s_genes_i)
        h, w = x.shape

        # 计算中位数组
        # 其实这里大概可以计算布尔分组情况，也就是一定不表达，一定表达等
        y = scmObject.labels[0]
        self.index_type_num = len(np.unique(y))
        self.index_expression = np.zeros((h, self.index_type_num))
        self.index_type = np.zeros(self.index_type_num)

        for i, t in enumerate(np.unique(y)):
            cell_type_mask = y == t
            x_i = x[:, cell_type_mask]
#
            med = np.array(np.median(x_i, axis=1))[:, 0]
            self.index_expression[:, i] = med
            self.index_type[i] = t
        # 需要增加全零和没有方差的情况的检查
        # 防止之后的相关性计算出现nan
        # 传递gene和index对应关系
        self.g_i_m = indexGeneMapper(self.s_genes, np.arange(h))


class scmapCluster(object):
    def __init__(self, p_scObject, index, threshold):
        s_genes = index.s_genes
        cf = np.intersect1d(np.unique(p_scObject.gene_list), s_genes)
        cf_pi = p_scObject.gene_to_index(cf)
        # 需要注意的是，这里需要验证方差，否则之后的计算会有nan
        p_expression = p_scObject.expression_matrix[cf_pi, :]
        cf_ii = index.gene_to_index(cf)
        i_expression = index.index_expression[cf_ii]

        # 计算相似度，大概可以扩展以及改造？拓展更多的相似度计算方式？
        # 以及，我是不是忘了要不要normlize一下？
        self.similarity_used = ['cosine', 'spearmanr', 'pearsonr']
        self.similarity_p = np.zeros(
            (3, index.index_type_num, p_scObject.cell_num))
        self.similarity_p[0] = \
            cosine_similarity(i_expression, p_expression)
        self.similarity_p[1] = \
            cor_ndarr(i_expression, p_expression, spearmanr)
        self.similarity_p[1] = \
            cor_ndarr(i_expression, p_expression, pearsonr)
        # 分别在在每个相似标准下得到对应细胞类型，以及打分？

    def get_p_labels(self):
        def max_labels(c_s_l):
            # 如果之前的计算已经排除的方差为零的情况，这里应该不会出现nan
            return np.argmax(c_s_l, axis=0)
        max_l = np.apply_along_axis(max_labels, 0, self.similarity_p)
        # max_l => (3, npc), 数值对应index中的细胞类型index

        def set_labels(s_v):
            l = len(np.unique(s_v))
            if l == 1:
                return s_v[0]
            elif l == 2:
                e = s_v[:, np.newaxis] == s_v[np.newaxis, :]
                e.sum(axis=1)
                return s_vp[e.argmax()]
            else:
                return -1

        def get_p_score(p_l):
            return self.similarity_p[:, p_labels, :]

        p_labels = np.apply_along_axis(set_labels, 0, max_l)
        p_scores = get_p_score(p_labels)

        return (p_labels, p_scores)


class indexCell(scBase):
    def __init__(self, scmObject, M=None, k=None):
        # 其实看起来这个就是拆开的Mini Batch K-Means，不过手动拆分而已
        s_genes_i, self.s_scores = scmObject.select_genes()
        self.s_genes = scmObject.index_to_gene(s_genes_i)
        self.cell_num = scmObject.cell_num
        if M is None:
            M = np.int(np.ceil(max(len(self.s_genes) / 10, 100)))
        if k in None:
            k = max(np.int(np.sqrt(self.cell_num)), 2)
        self.M = M
        self.k = k

        # x 之后你可以对x进行一些预处理
        s_expression = scmObject.get_selected_expression(s_genes_i)
        s_expression = normalize(s_expression, axis=0)
        # 比如x = norm(x), 需要注意一下方向

        self.g_i_m = indexGeneMapper(self.s_genes, ind)

        f_num = len(self.s_genes)
        # M 对genes分块
        chunks_i = [int(i * f_num / (M)) for i in range(M + 1)]
        self.predict_cluster = np.zeros((M, self.cell_num), dtype='int')
        # 需要注意的是，每个chunks不一定相等,以下不应该使用三维矩阵
        self.chunks_genes = []
        self.cluster_centers = np.zeros((f_num, self.k))
        for i in range(M):
            x_i = s_expression[chunks_i[i]:chunks_i[1+i], :].T
            try:
                # 前后距离度量要统一，都是使用cos
                kmeans = KMeans(n_clusters=k).fit(x_i)
                # 一些确认性的工作,比如聚类是否成功
            except Exception as e:
                raise e
            self.chunks_genes.append(
                self.index_to_gene(np.arange(chunks_i[i], chunks_i[1+i])))

            self.predict_cluster[chunks_i[i]:chunks_i[1+i], :] = \
                kmeans.predict_cluster.T

            self.cluster_centers.append(kmeans.cluster_centers_)
        self.chunks_i = np.array(chunks_i, dtype='int')
        self.labels = scmObject.labels


class scmapCell(object):
    def __init__(self, p_scObject, index, threshold):
        s_genes = index.s_genes
        cf = np.intersect1d(np.unique(p_scObject.gene_list), s_genes)
        cf_pi = p_scObject.gene_to_index(cf)
        p_expression = p_scObject.expression_matrix[cf_pi, :]
        p_expression = normalize(p_expression, axis=0)
        cf_ii = index.gene_to_index(cf)
        self.index_cluster = index.index.predict_cluster
        self.index_cell_list = index.cell_list
        self.index_labels = index.labels
        # 对index中的每个chunks分别选取相同genes
        # 然后搜索最近邻的点

        # 对每个cell需要求范数，归一化,这里是算个平方和
        sq_norm = np.sum(p_expression**2, axis=0)

        dist_to_index = np.zeros(index.M, index.k, p_scObject.cell_num)
        split = 0
        # 计算p对应m中的k个点的距离
        for i in range(index.M):
            hi, ti = index.chunks_i[i], index.chunks_i[i + 1]
            c_mask = np.intersect1d(cf_ii, np.arange(hi, ti))
            if len(c_mask) == 0:
                # 大概应该记录并丢弃
                raise Exception("No commom genes in chunk")
            xi_cluster_centers = index.cluster_centers[c_mask, :].T
            xp_expression = p_expression[split: split + len(c_mask)]
            # pair wise 计算xi，和xp之间的距离
            dist_to_index[i, :, :] = cosine_similarity(
                xi_cluster_centers, xp_expression)
        # 于是你得到了 c * r 的矩阵对应c到r个点的投影距离
        self.p_to_i_d = np.apply_along_axis(self._get_dist, 2, dist_to_index)
        # c * r 的index矩阵，对应r的排序index
        self.sort_di = np.apply_along_axis(np.argsort, 1, self.p_to_i_d)

    def _get_dist(self, dist_m):
        # return r vecter
        index_m = self.index_cluster

        m0, k = dist_m.shape
        m, r = index_m.shape

        if m0 != m:
            raise Exception('Data Not Match for NN')

        def sum_d(x):
            print(dist_m[range(m), x])
            return np.sum(dist_m[range(m), x])
        return np.apply_along_axis(sum_d, 0, index_m)

    def get_knn(self, k=3):
        if k > len(self.index_cell_list):
            raise Exception("Invalid 'k'")
        nnc_i = self.index_cell_list[self.sort_di[:k, :]]
        nnc = self.index_cell_list[nnc_i, 0]
        nnc_d = self.p_to_i_d[self.sort_di[:k, :]]
        return (nnc, nnc_d)

    def get_predict_labels(self, k=3):
        if k > len(self.index_cell_list):
            raise Exception("Invalid 'k'")
        nnc_i = self.index_cell_list[self.sort_di[:k, :]]
        nnc = self.index_cell_list[nnc_i, 0]
        nnc_l = self.index_labels[nnc_i, 0]
        nnc_d = self.p_to_i_d[self.sort_di[:k, :]]

        def assign_labels(x):
            if len(np.unique(x)) == 1:
                return x[0]
            else:
                return 'unassign'

        return np.apply_along_axis(assign_labels, 1, nnc_l)
