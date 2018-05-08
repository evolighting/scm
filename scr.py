import numpy as np
from . import utils
from sklearn.preprocessing import normalize


class indexGeneMapper(object):
    def __init__(self, genes_list, index):
        self.index_genes = dict(zip(index, genes_list))
        self.gene_index = dict(zip(genes_list, index))

    def gene_to_index(self, g_list):
        # 找不到应报错
        try:
            ind_l = [self.gene_index[i] for i in g_list]
        except KeyError:
            raise Exception("target gene not in list")
        return np.array(ind_l)

    def index_to_gene(self, i_list):
        # 找不到应报错
        try:
            ind_g = [self.index_genes[i] for i in i_list]
        except KeyError:
            raise Exception("index out of gene list's range")
        return np.array(ind_g)


class scBase(object):
    def __init__(self, gene_list):
        # gene_list 是一个基因的字符list
        self.gene_list = gene_list
        ind = np.arange(len(gene_list))
        self.g_i_m = indexGeneMapper(self.gene_list, ind)

    def gene_to_index(self, g_list):
        return self.g_i_m.gene_to_index(g_list)

    def index_to_gene(self, i_list):
        return self.g_i_m.index_to_gene(i_list)


class basicScObject(scBase):
    def __init__(self, expression_matrix, gene_list, cell_list, labels):
        # gene和cell应该可以通过一个转换方式，在各个数据集合间转换
        # 具体的，认为这里的gene是gengeid，各个数据集合之间应该是通用的
        self.expression_matrix = expression_matrix
        self.gene_list = gene_list
        self.cell_list = cell_list
        self.labels = labels
        self.gene_num, self.cell_num = expression_matrix.shape
        if (len(cell_list) != self.cell_num):
            raise Exception("Cells Number Not Match")
        if (len(cell_list) != self.cell_num):
            raise Exception("Genes Number Not Match")
        ind = np.arange(self.gene_num)
        self.g_i_m = indexGeneMapper(self.gene_list, ind)


def data_filter(scobject, m):
    f1, f2 = m(scobject)
    f1 = f1.getA1()
    f2 = f2.getA1()
    expression_matrix = scobject.expression_matrix[f1, :]
    expression_matrix = expression_matrix[:, f2]
    scobject = basicScObject(
        expression_matrix,
        scobject.gene_list[f1],
        scobject.cell_list[f2],
        scobject.labels[:, f2]
        )
    return (scobject, f1, f2)


class basic_filter(object):
    def __init__(self, min_cells, min_genges):
        self._min_cells = min_cells
        self._min_genes = min_genges

    def __call__(self, scobject):
        bm = scobject.expression_matrix > 0
        cells_filter = bm.sum(axis=0) >= self._min_genes
        genges_filter = bm.sum(axis=1) >= self._min_cells
        return (genges_filter, cells_filter)


class scmObject(basicScObject):
    def __new__(cls, parentInst):
        parentInst.__class__ = scmObject
        return parentInst

    def __init__(self, scObject):
        self.selected_genes = None

    def set_selected_genes(self, s_genes):
        # 有兴趣可以加一些验证
        self.selected_genes = None

    def get_selected_genes(self):
        if self.selected_genes is None:
            self.selected_genes = self.select_genes()
        return self.selected_genes

    def select_genes(self, n_genes=500, model=utils.linear_model):
        return utils.select_genes(self.expression_matrix, n_genes, model)

    def get_selected_expression(self, selected_genes=None):
        if selected_genes is None:
            if self.selected_genes is None:
                self.selected_genes = self.select_genes()[0]
            selected_genes = self.selected_genes
        return self.expression_matrix[selected_genes, :]