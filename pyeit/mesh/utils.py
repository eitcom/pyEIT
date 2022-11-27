# coding: utf-8
# pylint: disable=invalid-name, no-member, no-name-in-module
""" post process for distmesh 2d and 3d """
# Copyright (c) Benyuan Liu. All rights reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import division, absolute_import, print_function
from typing import Any
import numpy as np
import scipy.linalg as la


def dist(p: np.ndarray) -> Any:
    """distances to origin of nodes. '3D', 'ND' compatible

    Parameters
    ----------
    p : array_like
        points in 2D, 3D. i.e., in 3D
        [[x, y, z],
         [2, 3, 3],
         ...
         [1, 2, 1]]

    Returns
    -------
    array_like
        distances of points to origin
    """
    if p.ndim == 1:
        d = np.sqrt(np.sum(p**2))
    else:
        d = np.sqrt(np.sum(p**2, axis=1))

    return d


def edge_project(pts, fd, h0=1.0):
    """project points back on edge"""
    g_vec = edge_grad(pts, fd, h0)
    return pts - g_vec


def edge_grad(p, fd, h0=1.0):
    """
    project points back on the boundary (where fd=0) using numerical gradient
    3D, ND compatible

    Parameters
    ----------
    p : array_like
        points on 2D, 3D
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
        numerical gradient:
        f'_x = (f(p+delta_x) - f(p)) / delta_x
        f'_y = (f(p+delta_y) - f(p)) / delta_y
        f'_z = (f(p+delta_z) - f(p)) / delta_z

        you should specify h0 according to your actual mesh size
    """
    # d_eps = np.sqrt(np.finfo(float).eps)*h0
    # d_eps = np.sqrt(np.finfo(float).eps)
    d_eps = 1e-8 * h0

    # get dimensions
    if np.ndim(p) == 1:
        p = p[:, np.newaxis]

    # distance
    d = fd(p)

    # calculate the gradient of each axis
    ndim = p.shape[1]
    pts_xyz = np.repeat(p, ndim, axis=0)
    delta_xyz = np.repeat([np.eye(ndim)], p.shape[0], axis=0).reshape(-1, ndim)
    deps_xyz = d_eps * delta_xyz
    g_xyz = (fd(pts_xyz + deps_xyz) - np.repeat(d, ndim, axis=0)) / d_eps

    # normalize gradient, avoid divide by zero
    g = g_xyz.reshape(-1, ndim)
    g2 = np.sum(g**2, axis=1)

    # move unit
    g_num = g / g2[:, np.newaxis] * d[:, np.newaxis]

    return g_num


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
    bars = bars.view("i, i")
    n = bars.shape[0]

    ix = [True] * n
    for i in range(n - 1):
        # skip if already marked as duplicated
        if ix[i] is not True:
            continue
        # mark duplicate entries, at most 2-duplicate bars, if found, break
        for j in range(i + 1, n):
            if bars[i] == bars[j]:
                ix[i], ix[j] = False, False
                break

    return bars[np.array(ix)].view("i")


def check_ccw(no2xy, el2no):
    """
    check whether the simplices are CCW ordered, triangles only
    """
    xys = no2xy[el2no]
    a = [tri_area(xy) > 0 for xy in xys]
    return np.all(a)


def check_order(no2xy, el2no):
    """
    loop over all elements, calculate the Area of Elements (aoe)
    if AOE > 0, then the order of element is correct
    if AOE < 0, reorder the element

    Parameters
    ----------
    no2xy : NDArray
        Nx2 ndarray, (x,y) locations for points
    el2no : NDArray
        Mx3 ndarray, elements (triangles) connectivity

    Returns
    -------
    NDArray
        ae, area of each element

    Notes
    -----
    tetrahedron should be parsed that the sign of volume is [1, -1, 1, -1]
    """
    el_num, n_vertices = np.shape(el2no)
    # select ae function
    if n_vertices == 3:
        _fn = tri_area
    elif n_vertices == 4:
        _fn = tet_volume
    # calculate ae and re-order tri if necessary
    for ei in range(el_num):
        no = el2no[ei, :]
        xy = no2xy[no, :]
        v = _fn(xy)
        # if CCW, area should be positive, otherwise,
        if v < 0:
            el2no[ei, [1, 2]] = el2no[ei, [2, 1]]

    return el2no


def tri_area(xy):
    """
    return area of a triangle, given its tri-coordinates xy

    Parameters
    ----------
    xy : NDArray
        (x,y) of nodes 1,2,3 given in counterclockwise manner

    Returns
    -------
    float
        area of this element
    """
    s = xy[[2, 0]] - xy[[1, 2]]
    a_tot = 0.50 * la.det(s)
    # (should be positive if tri-points are counter-clockwise)
    return a_tot


def tet_volume(xyz):
    """calculate the volume of tetrahedron"""
    s = xyz[[2, 3, 0]] - xyz[[1, 2, 3]]
    v_tot = (1.0 / 6.0) * la.det(s)
    return v_tot


def to_polar(xy, shift=True, sort=True):
    vec = xy
    if shift:
        pc = np.median(xy, axis=0)
        print(pc)
        vec = vec - pc
    dist = np.sqrt(np.sum(vec**2, axis=1))
    deg = np.rad2deg(np.arctan2(vec[:, 1], vec[:, 0]))
    deg = deg % 360
    if sort:
        ind = np.argsort(deg)
        dist, deg = dist[ind], deg[ind]
    return dist, deg


def to_xy(dist, deg):
    x = dist * np.cos(np.deg2rad(deg))
    y = dist * np.sin(np.deg2rad(deg))
    return x, y


if __name__ == "__main__":
    # test 'edge_project'
    def fd_test(p):
        """unit circle/ball"""
        if len(p.shape) == 1:
            d = np.sqrt(np.sum(p**2)) - 1.0
        else:
            d = np.sqrt(np.sum(p**2, axis=1)) - 1.0

        return d

    p_test = [[1, 2, 3], [2, 2, 2], [1, 3, 3], [1, 1, 1]]
    a = edge_project(p_test, fd_test)

    # test 'edge_list'
