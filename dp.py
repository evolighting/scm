# -*- coding: utf-8 -*-


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
