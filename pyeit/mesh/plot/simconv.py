# -*- coding: utf-8 -*-
# pylint: disable=no-member, invalid-name
""" common function for simplices """

from itertools import combinations
import numpy as np


def sim_conv(simplices, N=3):
    """ simplices to any dimension """
    v = [list(combinations(sim, N)) for sim in simplices]
    # change to (num_of_points x N)
    t = np.sort(np.array(v).reshape(-1, N), axis=1)
    # delete duplicated entries
    t_unique = np.unique(t.view([('', t.dtype)]*N)).view(np.uint32)
    return t_unique


def sim2tri(simplices):
    """ convert simplices of high dimension to indices of triangles """
    return sim_conv(simplices, 3)


def sim2edge(simplices):
    """ convert simplices of high dimension to indices of edges """
    return sim_conv(simplices, 2)
