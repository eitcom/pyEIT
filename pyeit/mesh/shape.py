# coding: utf-8
# pylint: disable=invalid-name, no-member, too-many-locals
""" implement distance functions for distmesh """
# Copyright (c) Benyuan Liu. All rights reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import division, absolute_import, print_function
from typing import Any, List, Union
import numpy as np

from .utils import dist, edge_project
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


def circle(pts, pc: Union[np.ndarray, List] = [0, 0], r: float = 1.0) -> Any:
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
    if len(pc) != pts.shape[1]:
        pc = [0] * pts.shape[1]
    return dist(pts - pc) - r


def ellipse(
    pts, pc: Union[np.ndarray, List] = [0, 0], ab: Union[np.ndarray, List] = [1.0, 2.0]
):
    """Distance function for the ellipse
    centered at pc = [xc, yc], with a, b = [a, b]
    """
    return dist((pts - pc) / ab) - 1.0


def unit_circle(pts):
    """unit circle at (0, 0)"""
    return circle(pts, r=1.0)


def box_circle(pts):
    """unit circle at (0.5, 0.5) with r=0.5"""
    return circle(pts, pc=[0.5, 0.5], r=0.5)


def ball(pts, pc: Union[np.ndarray, List] = [0, 0, 0], r: float = 1.0):
    """
    generate balls in 3D (default: unit ball)

    See Also
    --------
    circle : generate circles in 2D
    """
    return circle(pts, pc, r)


def unit_ball(pts):
    """generate unit ball in 3D"""
    return ball(pts)


def rectangle0(
    pts, p1: Union[np.ndarray, List] = [0, 0], p2: Union[np.ndarray, List] = [1, 1]
):
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
    if pts.ndim == 1:
        pts = pts[np.newaxis]
    pd_left = [-min(row) for row in pts - p1]
    pd_right = [max(row) for row in pts - p2]

    return np.maximum(pd_left, pd_right)


def rectangle(
    pts, p1: Union[np.ndarray, List] = [0, 0], p2: Union[np.ndarray, List] = [1, 1]
):
    """
    smoothed rectangle

    p1: buttom-left corner
    p2: top-right corner
    """
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


def fix_points_fd(fd, n_el: int = 16, pc: Union[np.ndarray, List] = [0, 0]):
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
    # initialize points
    r0: float = 10.0
    theta: np.ndarray = 2.0 * np.pi * np.arange(n_el) / float(n_el)
    # add offset of theta
    # theta += theta[1] / 2.0
    p_fix = [[-r0 * np.cos(th), r0 * np.sin(th)] for th in theta]
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


def fix_points_circle(
    pc: Union[np.ndarray, List] = [0, 0],
    offset: float = 0,
    r: float = 1.0,
    ppl: int = 16,
):
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
    delta_theta = 2.0 * np.pi / float(ppl)
    theta = np.arange(ppl) * delta_theta + delta_theta * offset
    p_fix = [[-r * np.cos(th), r * np.sin(th)] for th in theta]
    return np.array(p_fix) + pc


def fix_points_ball(
    pc: Union[np.ndarray, List] = [0, 0, 0],
    r: float = 1.0,
    z: float = 0.0,
    n_el: int = 16,
):
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
    ry = np.sqrt(r**2 - z**2)
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


def area_uniform(p: np.ndarray):
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


def lshape(pts):
    """L_shaped mesh (for testing)"""
    return dist_diff(
        rectangle(pts, p1=[-1, -1], p2=[1, 1]), rectangle(pts, p1=[0, 0], p2=[1, 1])
    )


lshape_pfix = np.array(
    [
        [1, 0],
        [1, -1],
        [0, -1],
        [-1, -1],
        [-1, 0],
        [-1, 1],
        [0, 1],
        [0, 0],
    ]
)


def fd_polygon(poly, pts):
    """return signed distance of polygon"""
    pts_ = [Point(p) for p in pts]
    # calculate signed distance
    dist = [poly.exterior.distance(p) for p in pts_]
    sign = np.sign([-int(poly.contains(p)) + 0.5 for p in pts_])

    return sign * dist


def thorax(pts):
    """
    thorax polygon signed distance function

    Thorax contour points coordinates are taken from
    a thorax simulation based on EIDORS
    """
    poly = [
        (0.0487, 0.6543),
        (0.1564, 0.6571),
        (0.2636, 0.6697),
        (0.3714, 0.6755),
        (0.479, 0.6686),
        (0.5814, 0.6353),
        (0.6757, 0.5831),
        (0.7582, 0.5137),
        (0.8298, 0.433),
        (0.8894, 0.3431),
        (0.9347, 0.2452),
        (0.9698, 0.1431),
        (0.9938, 0.0379),
        (1.0028, -0.0696),
        (0.9914, -0.1767),
        (0.9637, -0.281),
        (0.9156, -0.3771),
        (0.8359, -0.449),
        (0.7402, -0.499),
        (0.6432, -0.5463),
        (0.5419, -0.5833),
        (0.4371, -0.6094),
        (0.3308, -0.6279),
        (0.2243, -0.6456),
        (0.1168, -0.6508),
        (0.0096, -0.6387),
        (-0.098, -0.6463),
        (-0.2058, -0.6433),
        (-0.313, -0.6312),
        (-0.4181, -0.6074),
        (-0.5164, -0.5629),
        (-0.6166, -0.5232),
        (-0.7207, -0.4946),
        (-0.813, -0.4398),
        (-0.8869, -0.3614),
        (-0.933, -0.2647),
        (-0.9451, -0.1576),
        (-0.9425, -0.0498),
        (-0.9147, 0.0543),
        (-0.8863, 0.1585),
        (-0.8517, 0.2606),
        (-0.8022, 0.3565),
        (-0.7413, 0.4455),
        (-0.6664, 0.5231),
        (-0.5791, 0.5864),
        (-0.4838, 0.6369),
        (-0.3804, 0.667),
        (-0.2732, 0.6799),
        (-0.1653, 0.6819),
        (-0.0581, 0.6699),
    ]
    poly_obj = Polygon(poly)
    return fd_polygon(poly_obj, pts)


thorax_pfix = np.array(
    [
        (-0.098, -0.6463),
        (-0.4181, -0.6074),
        (-0.7207, -0.4946),
        (-0.933, -0.2647),
        (-0.9147, 0.0543),
        (-0.8022, 0.3565),
        (-0.5791, 0.5864),
        (-0.1653, 0.6819),
        (0.1564, 0.6571),
        (0.5814, 0.6353),
        (0.8298, 0.433),
        (0.9698, 0.1431),
        (0.9914, -0.1767),
        (0.8359, -0.449),
        (0.5419, -0.5833),
        (0.2243, -0.6456),
    ]
)


head_symm_poly = (
    np.array(
        [
            [197, 0],
            [188, 43],
            [174, 83],
            [150, 120],
            [118, 148],
            [81, 168],
            [41, 182],
            [0, 186],
            [-41, 182],
            [-81, 168],
            [-118, 148],
            [-150, 120],
            [-174, 83],
            [-188, 43],
            [-197, 0],
            [-201, -35],
            [-194, -70],
            [-185, -106],
            [-169, -141],
            [-148, -177],
            [-123, -213],
            [-88, -241],
            [-45, -259],
            [0, -263],
            [45, -259],
            [88, -241],
            [123, -213],
            [148, -177],
            [169, -141],
            [185, -106],
            [194, -70],
            [201, -35],
        ]
    )
    / 255.0
)

head_symm_pfix = np.array(head_symm_poly[::-2])


def head_symm(pts):
    """symmetric head polygon"""
    poly_obj = Polygon(head_symm_poly)
    return fd_polygon(poly_obj, pts)
