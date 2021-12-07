# coding: utf-8
# pylint: disable=invalid-name, no-member, too-many-arguments
""" wrapper function of distmesh for EIT """
# Copyright (c) Benyuan Liu. All rights reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import division, absolute_import, print_function

import numpy as np

from .distmesh import build
from .mesh_circle import MeshCircle
from .utils import check_order
from .shape import circle, area_uniform, ball, thorax, L_shaped
from .shape import fix_points_fd, fix_points_ball


def create(n_el=16, fd=None, fh=area_uniform, h0=0.1, p_fix=None, bbox=None):
    """
    Generating 2D/3D meshes using distmesh (pyEIT built-in)

    Parameters
    ----------
    n_el: int
        number of electrodes (point-type electrode)
    fd: function
        distance function (circle in 2D, ball in 3D)
    fh: function
        mesh size quality control function
    p_fix: NDArray
        fixed points
    bbox: NDArray
        bounding box
    h0: float
        initial mesh size, default=0.1

    Returns
    -------
    mesh_obj: dict
        {'element', 'node', 'perm'}
    """

    # test conditions if fd or/and bbox are none

    if bbox is None:
        if fd != ball:
            bbox = np.array([[-1, -1], [1, 1]])
        else:
            bbox = [[-1.2, -1.2, -1.2], [1.2, 1.2, 1.2]]

    bbox = np.array(
        bbox
    )  # list is converted to Numpy array so we can use it then (calling shape method..)
    n_dim = bbox.shape[1]  # bring dimension

    # infer dim
    if fd is None:
        if n_dim == 2:
            fd = circle
        elif n_dim == 3:
            fd = ball

    if n_dim not in [2, 3]:
        raise TypeError("distmesh only supports 2D or 3D")
    if bbox.shape[0] != 2:
        raise TypeError("please specify lower and upper bound of bbox")

    if p_fix is None:
        if n_dim == 2:
            if fd == thorax:
                # thorax shape is generated so far without fixed points (to be updated later)
                p_fix = [
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
                p_fix = np.array(p_fix)
            elif fd == L_shaped:
                p_fix = [
                    [1, 0],
                    [1, -1],
                    [0, -1],
                    [-1, -1],
                    [-1, 0],
                    [-1, 1],
                    [0, 1],
                    [0, 0],
                ]  # values brought from distmesh2D L shaped mesh example
                p_fix = np.array(p_fix)
                h0 = 0.15
            else:
                p_fix = fix_points_fd(fd, n_el=n_el)
        elif n_dim == 3:
            p_fix = fix_points_ball(n_el=n_el)

    # 1. build mesh
    p, t = build(fd, fh, pfix=p_fix, bbox=bbox, h0=h0)
    # 2. check whether t is counter-clock-wise, otherwise reshape it
    t = check_order(p, t)
    # 3. generate electrodes, the same as p_fix (top n_el)
    el_pos = np.arange(n_el)
    # 4. init uniform element permittivity (sigma)
    perm = np.ones(t.shape[0], dtype=np.float)
    # 5. build output structure
    mesh = {"element": t, "node": p, "perm": perm}
    return mesh, el_pos


def set_perm(mesh, anomaly=None, background=None):
    """wrapper for pyEIT interface

    Note
    ----
    update permittivity of mesh, if specified.

    Parameters
    ----------
    mesh: dict
        mesh structure
    anomaly: dict, optional
        anomaly is a dictionary (or arrays of dictionary) contains,
        {'x': val, 'y': val, 'd': val, 'perm': val}
        all permittivity on triangles whose distance to (x,y) are less than (d)
        will be replaced with a new value, 'perm' may be a complex value.
    background: float, optional
        set background permittivity

    Returns
    -------
    mesh_obj: dict
        updated mesh structure, {'element', 'node', 'perm'}
    """
    pts = mesh["element"]
    tri = mesh["node"]
    perm = mesh["perm"].copy()
    tri_centers = np.mean(tri[pts], axis=1)

    # this code is equivalent to:
    # >>> N = np.shape(tri)[0]
    # >>> for i in range(N):
    # >>>     tri_centers[i] = np.mean(pts[tri[i]], axis=0)
    # >>> plt.plot(tri_centers[:,0], tri_centers[:,1], 'kx')
    n = np.size(mesh["perm"])

    # reset background if needed
    if background is not None:
        perm = background * np.ones(n)

    # change dtype to 'complex' for complex-valued permittivity
    if anomaly is not None:
        for attr in anomaly:
            if np.iscomplex(attr["perm"]):
                perm = perm.astype("complex")
                break

    # assign anomaly values (for elements in regions)
    if anomaly is not None:
        for _, attr in enumerate(anomaly):
            d = attr["d"]
            # find elements whose distance to (cx,cy) is smaller than d
            if "z" in attr:
                index = (
                    np.sqrt(
                        (tri_centers[:, 0] - attr["x"]) ** 2
                        + (tri_centers[:, 1] - attr["y"]) ** 2
                        + (tri_centers[:, 2] - attr["z"]) ** 2
                    )
                    < d
                )
            else:
                index = (
                    np.sqrt(
                        (tri_centers[:, 0] - attr["x"]) ** 2
                        + (tri_centers[:, 1] - attr["y"]) ** 2
                    )
                    < d
                )
            # update permittivity within indices
            perm[index] = attr["perm"]

    mesh_new = {"node": tri, "element": pts, "perm": perm}
    return mesh_new


def layer_circle(n_el=16, n_fan=8, n_layer=8):
    """generate mesh on unit-circle"""
    model = MeshCircle(n_fan=n_fan, n_layer=n_layer, n_el=n_el)
    p, e, el_pos = model.create()
    perm = np.ones(e.shape[0])

    mesh = {"element": e, "node": p, "perm": perm}
    return mesh, el_pos
