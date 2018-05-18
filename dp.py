# -*- coding: utf-8 -*-
import numpy as np


def celltypesref(pl):
    """
    get coding cell dict for cell type
    local version
    """
    ppl = set(pl)
    refnd = dict(zip(ppl, range(len(ppl))))
    return refnd


def osnc(pl, refnd):
    """
    being passed coding funtion
    """
    return [refnd[i] for i in pl]


class csnc(object):
    def __init__(self, refnd):
        self.refnd = refnd
        self.func = osnc

    def __call__(self, pl):
        return self.func(pl, self.refnd)


def log_normalize(expression, scalefactor=10000):
    total_expression_e_cell = expression.sum(axis=0)

    log_norm_expr = np.log1p(
        expression / total_expression_e_cell *
        scalefactor)
    return log_norm_expr

# Eigen::SparseMatrix<double> LogNorm(Eigen::SparseMatrix<double> data, int scale_factor, bool display_progress = true){
#   Progress p(data.outerSize(), display_progress);
#   Eigen::VectorXd colSums = data.transpose() * Eigen::VectorXd::Ones(data.cols());
#   for (int k=0; k < data.outerSize(); ++k){
#     p.increment();
#     for (Eigen::SparseMatrix<double>::InnerIterator it(data, k); it; ++it){
#       data.coeffRef(it.row(), it.col()) = log1p(double(it.value()) / colSums[k] * scale_factor);
#     }
#   }
#   return data;
# }
