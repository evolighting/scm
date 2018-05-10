import numpy as np
from scipy.io import mmread
import scm
import json
from sklearn.metrics import cohen_kappa_score


paht = '<path>'
jsonf = open(path+'info.json', 'r')
jd = json.load(jsonf)
ex_m = mmread(path+'expr_m.mtx').todense()
ex_m = np.log10(ex_m+1)
genes_list = np.array(jd[1])
cell_list = np.array(jd[2])
cell_type_list = np.array(jd[3])

f1 = cell_type_list != 'alpha.contaminated'
f2 = cell_type_list != 'beta.contaminated'
f3 = cell_type_list != 'gamma.contaminated'
f4 = cell_type_list != 'delta.contaminated'
fc = f1 * f2 * f3 * f4

cell_list = cell_list[fc]
cell_type_list = cell_type_list[fc]
ex_m = ex_m[:, fc]

refd1 = scm.celltypesref(cell_type_list)
snc1 = scm.csnc(refd1)
o_label = snc1(cell_type_list)
labels = np.array(o_label, dtype='int')[np.newaxis, :]

bs = scm.basicScObject(ex_m, genes_list, cell_list, labels)
# test with self
bs2 = scm.basicScObject(
    ex_m[:, 0:999], genes_list, cell_list[0:999], labels[:, 0:999])
del(ex_m)

bf = scm.basic_filter(min_cells=100, min_genges=5)
bs = scm.data_filter(bs, bf)[0]
ss = scm.scmObject(bs)

index_cluster = scm.indexCluster(ss)

scmap_cluster = scm.scmapCluster(bs, index_cluster)
# test cluster
print(cohen_kappa_score(bs.labels[0], scmap_cluster.get_p_labels()[0]))

index_cell = scm.indexCell(ss)
# test self knn
print(index_cell.get_knn_cells(bs2))

pl = index_cell.get_predict_labels(bs2, k=2)
# test self label
print(cohen_kappa_score(bs2.labels[0, :], pl))