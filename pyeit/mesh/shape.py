# coding: utf-8
# pylint: disable=invalid-name, no-member, too-many-locals
""" implement distance functions for distmesh """
# Copyright (c) Benyuan Liu. All rights reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import division, absolute_import, print_function

import numpy as np

from .utils import dist, edge_project


def circle(pts, pc=None, r=1.0):
    """
    Distance function for the circle centered at pc = [xc, yc]

    Parameters
    ----------
    pts : array_like
        points on 2D
    pc : array_like, optional
        center of points
    r : float, optional
        radius

    Returns
    -------
    array_like
        distance of (points - pc) - r

    Note
    ----
    copied and modified from https://github.com/ckhroulev/py_distmesh2d
    """
    if pc is None:
        pc = [0, 0]
    return dist(pts - pc) - r


def ellipse(pts, pc=None, ab=None):
    """Distance function for the ellipse
    centered at pc = [xc, yc], with a, b = [a, b]
    """
    if pc is None:
        pc = [0, 0]
    if ab is None:
        ab = [1.0, 2.0]
    return dist((pts - pc) / ab) - 1.0


def unit_circle(pts):
    """ unit circle at (0,0) """
    return circle(pts, r=1.0)


def box_circle(pts):
    """ unit circle at (0.5,0.5) with r=0.5 """
    return circle(pts, pc=[0.5, 0.5], r=0.5)


def ball(pts, pc=None, r=1.0):
    """
    generate balls in 3D (default: unit ball)

    See Also
    --------
    circle : generate circles in 2D
    """
    if pc is None:
        pc = [0, 0, 0]
    return circle(pts, pc, r)


def unit_ball(pts):
    """ generate unit ball in 3D """
    return ball(pts)


def rectangle0(pts, p1=None, p2=None):
    """
    Distance function for the rectangle p1=[x1, y1] and p2=[x2, y2]

    Note
    ----
    p1 should be bottom-left, p2 should be top-right
    if p in rect(p1, p2), then (p-p1)_x and (p-p2)_x must have opposite sign

    Parameters
    ----------
    pts : array_like
    p1 : array_like, optional
        bottom left coordinates
    p2 : array_like, optional
        top tight coordinates

    Returns
    -------
    array_like
        distance
    """
    if p1 is None:
        p1 = [0, 0]
    if p2 is None:
        p2 = [1, 1]
    if pts.ndim == 1:
        pts = pts[np.newaxis]
    pd_left = [-min(row) for row in pts - p1]
    pd_right = [max(row) for row in pts - p2]

    return np.maximum(pd_left, pd_right)


def rectangle(pts, p1=None, p2=None):
    """
    smoothed rectangle

    p1: buttom-left corner
    p2: top-right corner
    """
    if p1 is None:
        p1 = [0, 0]
    if p2 is None:
        p2 = [1, 1]
    if pts.ndim == 1:
        pts = pts[np.newaxis]

    d1x = -(pts[:, 0] - p1[0])
    d2x = pts[:, 0] - p2[0]
    d1y = -(pts[:, 1] - p1[1])
    d2y = pts[:, 1] - p2[1]

    # find interior points (d < 0)
    dx = np.maximum(d1x, d2x)
    dy = np.maximum(d1y, d2y)
    d = np.maximum(dx, dy)

    # smoothed corner distance function
    ix_left = d1x > 0
    ix_right = d2x > 0
    iy_below = d1y > 0
    iy_above = d2y > 0

    # 1, 2, 3, 4 Quadrant outside-points smooth distance
    ix1 = np.logical_and(ix_right, iy_above)
    d[ix1] = np.sqrt(d2x[ix1] ** 2 + d2y[ix1] ** 2)
    ix2 = np.logical_and(ix_left, iy_above)
    d[ix2] = np.sqrt(d1x[ix2] ** 2 + d2y[ix2] ** 2)
    ix3 = np.logical_and(ix_left, iy_below)
    d[ix3] = np.sqrt(d1x[ix3] ** 2 + d1y[ix3] ** 2)
    ix4 = np.logical_and(ix_right, iy_below)
    d[ix4] = np.sqrt(d2x[ix4] ** 2 + d1y[ix4] ** 2)

    return d


def fix_points_fd(fd, n_el=16, pc=None):
    """
    return fixed and uniformly distributed points on
    fd with equally distributed angles

    Parameters
    ----------
    fd : distance function
    pc : array_like, optional
        center of points
    n_el : number of electrodes, optional

    Returns
    -------
    array_like
        coordinates of fixed points
    """
    if pc is None:
        pc = [0, 0]

    # initialize points
    r = 10.0
    theta = 2.0 * np.pi * np.arange(n_el) / float(n_el)
    # add offset of theta
    # theta += theta[1] / 2.0
    p_fix = [[-r * np.cos(th), r * np.sin(th)] for th in theta]
    pts = np.array(p_fix) + pc

    # project back on edges
    pts_new = np.inf * np.ones_like(pts)
    c = False
    d_eps = 0.1
    max_iter = 10
    niter = 0
    while not c:
        # project on fd
        pts_new = edge_project(pts, fd)
        # project on rays
        r = dist(pts_new)
        pts_new = [[-ri * np.cos(ti), ri * np.sin(ti)] for ri, ti in zip(r, theta)]
        pts_new = np.array(pts_new)
        # check convergence
        c = np.sum(dist(pts_new - pts)) < d_eps or niter > max_iter
        pts = pts_new
        niter += 1
    return pts_new


def fix_points_circle(pc=None, offset=0, r=1.0, ppl=16):
    """
    return fixed and uniformly distributed points on
    a circle with radius r

    Parameters
    ----------
    pc : array_like, optional
        center of points
    r : float, optional
        radius
    ppl : number of points, optional

    Returns
    -------
    array_like
        coordinates of fixed points
    """
    if pc is None:
        pc = [0, 0]

    delta_theta = 2.0 * np.pi / float(ppl)
    theta = np.arange(ppl) * delta_theta + delta_theta * offset
    p_fix = [[-r * np.cos(th), r * np.sin(th)] for th in theta]
    return np.array(p_fix) + pc


def fix_points_ball(pc=None, r=1.0, z=0.0, n_el=16):
    """
    return fixed and uniformly distributed points on
    a circle with radius r

    Parameters
    ----------
    pc : array_like,
        center of points
    r : float,
        radius
    z : float,
        z level of points
    n_el : number of electrodes, optional

    Returns
    -------
    array_like
        coordinates of fixed points
    """
    if pc is None:
        pc = [0, 0, 0]

    ry = np.sqrt(r ** 2 - z ** 2)
    theta = 2.0 * np.pi * np.arange(n_el) / float(n_el)
    p_fix = [[ry * np.sin(th), ry * np.cos(th), z] for th in theta]
    return np.array(p_fix) + pc


def dist_diff(d1, d2):
    """Distance function for the difference of two sets.

    Parameters
    ----------
    d1 : array_like
    d2 : array_like
        distance of two functions

    Returns
    -------
    array_like
        maximum difference

    Note
    ----
    boundary is denoted by d=0
    copied and modified from https://github.com/ckhroulev/py_distmesh2d
    """
    return np.maximum(d1, -d2)


def dist_intersect(d1, d2):
    """Distance function for the intersection of two sets.

    Parameters
    ----------
    d1 : array_like
    d2 : array_like
        distance of two functions

    Returns
    -------
    array_like

    Note
    ----
    boundary is denoted by d=0
    copied and modified from https://github.com/ckhroulev/py_distmesh2d
    """
    return np.maximum(d1, d2)


def dist_union(d1, d2):
    """Distance function for the union of two sets.

    Parameters
    ----------
    d1 : array_like
    d2 : array_like
        distance of two functions

    Returns
    -------
    array_like

    Note
    ----
    boundary is denoted by d=0
    copied and modified from https://github.com/ckhroulev/py_distmesh2d
    """
    return np.minimum(d1, d2)


def area_uniform(p):
    """uniform mesh distribution

    Parameters
    ----------
    p : array_like
        points coordinates

    Returns
    -------
    array_like
        ones

    """
    return np.ones(p.shape[0])
