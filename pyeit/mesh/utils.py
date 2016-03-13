# coding: utf-8
# pylint: disable=invalid-name, no-member, no-name-in-module
""" post process for distmesh 2d and 3d """
from __future__ import absolute_import

import numpy as np
from numpy import sqrt


def dist(p):
    """ distances to origin of nodes

    Parameters
    ----------
    p : array_like
        points in 2D

    Returns
    -------
    array_like
        distances of points to origin
    """
    return np.array([sqrt(row[0]**2 + row[1]**2) for row in p])


def edge_project(pts, fd, h0=1.0):
    """
    project points back on the boundary (where fd=0) using numerical gradient

    Parameters
    ----------
    pts : array_like
        points on 2D
    fd : str
        function handler of distances
    h0 : float
        minimal distance

    Returns
    -------
    array_like
        gradients of points on the boundary

    Note
    ----
    you should specify h0 according to your actual mesh size
    """
    deps = sqrt(np.finfo(float).eps)*h0
    d = fd(pts)
    dgradx = (fd(pts + [deps, 0]) - d) / deps
    dgrady = (fd(pts + [0, deps]) - d) / deps
    dgrad2 = dgradx**2 + dgrady**2
    dgrad2[dgrad2 == 0] = 1.
    # calculate gradient vector (minus)
    pgrad = np.vstack([d*dgradx/dgrad2, d*dgrady/dgrad2]).T
    return pgrad


def edge_list(tri):
    """
    edge of delaunay triangles are unique bars, O(n^2)

    besides this duplication test, you may also use fd to test edge bars,
    where the two nodes of any edge both satisfy fd=0 (or with a geps gap)

    Parameters
    ----------
    tri : array_like
        triangles list
    """
    bars = tri[:, [[0, 1], [1, 2], [2, 0]]].reshape((-1, 2))
    bars = np.sort(bars, axis=1)
    bars = bars.view('i, i')
    n = bars.shape[0]

    ix = [True] * n
    for i in range(n-1):
        # skip if already marked as duplicated
        if ix[i] is not True:
            continue
        # mark duplicate entries, at most 2-duplicate bars, if found, break
        for j in range(i+1, n):
            if bars[i] == bars[j]:
                ix[i], ix[j] = False, False
                break

    return bars[np.array(ix)].view('i')
