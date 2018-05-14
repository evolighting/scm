import numpy as np
from .utils import cor_ndarr
from .scr import indexGeneMapper, scBase
from .pq import pqIndex, pqIndex_Euc
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr, pearsonr
from sklearn.cluster import KMeans


class indexCluster(scBase):
    def __init__(self, scmObject):
        self.expression_matrix = scmObject.expression_matrix
        # 如果要对表达量进行处理，应该在之前进行，
        # 这里接受的应该是处理之后的对象，
        # 具体的，认为已经使用过log(x+1)对表达矩阵进行处理
        s_genes_i, s_scores = scmObject.select_genes()
        s_genes = scmObject.gene_list[s_genes_i]
        x = scmObject.get_selected_expression(s_genes_i)
        h, w = x.shape

        # 计算中位数组
        # 其实这里大概可以计算布尔分组情况，也就是一定不表达，一定表达等
        y = scmObject.labels[0]
        self.index_type_num = len(np.unique(y))
        index_expression = np.zeros((h, self.index_type_num))
        self.index_type = np.zeros(self.index_type_num)

        for i, t in enumerate(np.unique(y)):
            cell_type_mask = y == t
            x_i = x[:, cell_type_mask]
#
            med = np.array(np.median(x_i, axis=1))[:, 0]
            index_expression[:, i] = med
            self.index_type[i] = t
        # 需要增加全零和没有方差的情况的检查
        z_f = index_expression.sum(axis=1) != 0
        self.index_expression = index_expression[z_f, :]
        self.s_genes = s_genes[z_f]
        self.s_scores = s_scores[z_f]
        # 防止之后的相关性计算出现nan
        # 传递gene和index对应关系
        self.g_i_m = indexGeneMapper(self.s_genes, range(np.sum(z_f)))


class scmapCluster(object):
    def __init__(self, p_scObject, index, threshold=0.5):
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
            cosine_similarity(i_expression.T, p_expression.T)
        self.similarity_p[1] = \
            cor_ndarr(i_expression.T, p_expression.T, spearmanr)
        self.similarity_p[2] = \
            cor_ndarr(i_expression.T, p_expression.T, pearsonr)
        # 分别在在每个相似标准下得到对应细胞类型，以及打分？

    def get_p_labels(self):
        def max_labels(c_s_l):
            # 如果之前的计算已经排除的方差为零的情况，这里应该不会出现nan
            return np.argmax(c_s_l, axis=0)
        max_l = np.apply_along_axis(max_labels, 1, self.similarity_p)
        # max_l => (3, npc), 数值对应index中的细胞类型index

        def set_labels(s_v):
            l = len(np.unique(s_v))
            if l == 1:
                return s_v[0]
            elif l == 2:
                e = s_v[:, np.newaxis] == s_v[np.newaxis, :]
                e.sum(axis=1)
                return s_v[e.argmax()]
            else:
                return -1

        def get_p_score(p_l):
            return self.similarity_p[:, p_l, np.arange(len(p_l))].T

        p_labels = np.apply_along_axis(set_labels, 0, max_l)
        p_scores = get_p_score(p_labels)

        return (p_labels, p_scores)


class indexCell(scBase):
    def __init__(self, scmObject, M=None, k=None):
        # Product Quantization for ANN
        self.cell_list = scmObject.cell_list
        s_genes_i, self.s_scores = scmObject.select_genes()
        self.s_genes = scmObject.index_to_gene(s_genes_i)
        self.cell_num = scmObject.cell_num
        self.labels = scmObject.labels
        if M is None:
            M = np.int(min(len(self.s_genes) / 10, 100))
        if k is None:
            k = np.int(max(np.sqrt(self.cell_num), 2))
        self.M = M
        self.k = k

        s_expression = scmObject.get_selected_expression(s_genes_i)

        self.g_i_m = indexGeneMapper(self.s_genes, range(len(self.s_genes)))

        self._pqindex = pqIndex(s_expression.T, m=self.M, k=self.k)

    def _get_knn_to_index(self, p_scObject, k=3):
        cf = np.intersect1d(p_scObject.gene_list, self.s_genes)
        cf_ii = self.gene_to_index(cf)
        cf_pi = p_scObject.gene_to_index(cf)
        p_expression = p_scObject.expression_matrix[cf_pi, :]

        return self._pqindex.get_knn_filtered(p_expression.T, cf_ii, k)

    def get_knn_cells(self, p_scObject, k=3):
        if k > len(self.cell_list):
            raise Exception("Invalid 'k'")
        knn_i, knn_s = self._get_knn_to_index(p_scObject, k=k)
        knn_c = self.cell_list[knn_i]
        return (knn_i, knn_c, knn_s)

    def get_predict_labels(self, p_scObject, k=3):
        if k > len(self.cell_list):
            raise Exception("Invalid 'k'")
        knn_i, knn_c, knn_s = self.get_knn_cells(p_scObject, k=k)
        knn_l = self.labels[0, knn_i]

        def assign_labels(x):
            if len(np.unique(x)) == 1:
                return x[0]
            else:
                return -1

        return np.apply_along_axis(assign_labels, 1, knn_l)


class indexCell_Euc(scBase):
    def __init__(self, scmObject, M=None, k=None):
        # Product Quantization for ANN
        # 替换欧氏距离
        self.cell_list = scmObject.cell_list
        s_genes_i, self.s_scores = scmObject.select_genes()
        self.s_genes = scmObject.index_to_gene(s_genes_i)
        self.cell_num = scmObject.cell_num
        self.labels = scmObject.labels
        if M is None:
            M = np.int(min(len(self.s_genes) / 10, 100))
        if k is None:
            k = np.int(max(np.sqrt(self.cell_num), 2))
        self.M = M
        self.k = k

        s_expression = scmObject.get_selected_expression(s_genes_i)

        self.g_i_m = indexGeneMapper(self.s_genes, range(len(self.s_genes)))

        self._pqindex = pqIndex_Euc(s_expression.T, m=self.M, k=self.k)

    def _get_knn_to_index(self, p_scObject, k=3):
        cf = np.intersect1d(p_scObject.gene_list, self.s_genes)
        cf_ii = self.gene_to_index(cf)
        cf_pi = p_scObject.gene_to_index(cf)
        p_expression = p_scObject.expression_matrix[cf_pi, :]

        return self._pqindex.get_knn_filtered(p_expression.T, cf_ii, k)

    def get_knn_cells(self, p_scObject, k=3):
        if k > len(self.cell_list):
            raise Exception("Invalid 'k'")
        knn_i, knn_s = self._get_knn_to_index(p_scObject, k=k)
        knn_c = self.cell_list[knn_i]
        return (knn_i, knn_c, knn_s)

    def get_predict_labels(self, p_scObject, k=3):
        if k > len(self.cell_list):
            raise Exception("Invalid 'k'")
        knn_i, knn_c, knn_s = self.get_knn_cells(p_scObject, k=k)
        knn_l = self.labels[0, knn_i]

        def assign_labels(x):
            if len(np.unique(x)) == 1:
                return x[0]
            else:
                return -1

        return np.apply_along_axis(assign_labels, 1, knn_l)
