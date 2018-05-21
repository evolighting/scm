import numpy as np
from .utils import cor_ndarr
from .scr import indexGeneMapper, scBase
from .pq import pqIndex, pqIndex_Euc, pqIndex_supervised
from .lshs import scLSH


class indexCell_ml(scBase):
    def __init__(self, scmObject, ml, M=None, k=None):
        # Product Quantization for ANNimport numpy as np
        # 增加ml，对input进行处理
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

        s_expression = np.copy(scmObject.get_selected_expression(s_genes_i))

        self.m_l = ml.fit(s_expression.T, scmObject.labels[0])
        s_expression = self.m_l.transform()

        self.g_i_m = indexGeneMapper(self.s_genes, range(len(self.s_genes)))

        self._pqindex = pqIndex_Euc(s_expression, m=self.M, k=self.k)

    def _get_knn_to_index(self, p_scObject, k=3):
        cf = np.intersect1d(p_scObject.gene_list, self.s_genes)
        cf_ii = self.gene_to_index(cf)
        cf_pi = p_scObject.gene_to_index(cf)
        p_expression = np.zeros((len(self.s_genes), p_scObject.cell_num))
        p_expression[cf_ii, :] = p_scObject.expression_matrix[cf_ii, :]
        p_expression = self.m_l.transform(p_expression.T)

        return self._pqindex.get_knn(p_expression, k)

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


class indexCell_supervised(scBase):
    def __init__(self, scmObject, M=None, k=None):
        # Product Quantization for ANNimport numpy as np
        # 利用label信息的Product Quantization版本
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

        s_expression = np.copy(scmObject.get_selected_expression(s_genes_i))

        self.g_i_m = indexGeneMapper(self.s_genes, range(len(self.s_genes)))

        self._pqindex = pqIndex_supervised(
            s_expression.T, self.labels[0], m=self.M, k=self.k)

    def _get_knn_to_index(self, p_scObject, k=3):
        cf = np.intersect1d(p_scObject.gene_list, self.s_genes)
        cf_ii = self.gene_to_index(cf)
        cf_pi = p_scObject.gene_to_index(cf)
        p_expression = np.zeros((len(self.s_genes), p_scObject.cell_num))
        p_expression[cf_ii, :] = p_scObject.expression_matrix[cf_ii, :]

        return self._pqindex.get_knn(p_expression.T, k)

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


class indexCell_lsh(scBase):
    def __init__(self, scmObject, M=None, k=None):
        # Product Quantization for ANNimport numpy as np
        # 利用label信息的Product Quantization版本
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

        s_expression = np.copy(scmObject.get_selected_expression(s_genes_i))

        self.g_i_m = indexGeneMapper(self.s_genes, range(len(self.s_genes)))

        self._pqindex = scLSH(s_expression.T)

    def _get_knn_to_index(self, p_scObject, k=3):
        cf = np.intersect1d(p_scObject.gene_list, self.s_genes)
        cf_ii = self.gene_to_index(cf)
        cf_pi = p_scObject.gene_to_index(cf)
        p_expression = np.zeros((len(self.s_genes), p_scObject.cell_num))
        p_expression[cf_ii, :] = p_scObject.expression_matrix[cf_ii, :]

        return self._pqindex.get_knn(p_expression.T, k)

    def get_knn_cells(self, p_scObject, k=3):
        if k > len(self.cell_list):
            raise Exception("Invalid 'k'")
        knn_i, knn_s = self._get_knn_to_index(p_scObject, k=k)
        cell_list = np.append(self.cell_list, ['NAN'])
        print(cell_list[-1])
        knn_c = cell_list[knn_i]
        return (knn_i, knn_c, knn_s)

    def get_predict_labels(self, p_scObject, k=3):
        if k > len(self.cell_list):
            raise Exception("Invalid 'k'")
        knn_i, knn_c, knn_s = self.get_knn_cells(p_scObject, k=k)
        labels = np.append(self.labels[0], '-1')
        knn_l = labels[knn_i]

        def assign_labels(x):
            if len(np.unique(x)) == 1:
                return x[0]
            else:
                return -1

        return np.apply_along_axis(assign_labels, 1, knn_l)