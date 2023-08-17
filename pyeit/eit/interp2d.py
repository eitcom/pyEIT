# coding: utf-8
# pylint: disable=invalid-name, no-member, too-many-locals, no-name-in-module
""" interpolation on 2D/3D irregular/regular grids """
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import absolute_import, division, print_function, annotations

from typing import Tuple, Union, Any

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
from matplotlib.path import Path

# for debugging
from pyeit.mesh import layer_circle, set_perm
from scipy.sparse import coo_matrix
from scipy.spatial import ConvexHull


class TriangleRasterizer:
    def __init__(self, pts, tri):
        tp = pts[:, np.newaxis][tri].squeeze()
        tri_vec = tp[:, [1, 2, 0]] - tp
        self.tp = tp
        self.atot = np.abs(self._tri_area(tri_vec[:, 0], tri_vec[:, 1]))

    @staticmethod
    def _tri_area(bar0, bar1):
        return bar0[:, 0] * bar1[:, 1] - bar0[:, 1] * bar1[:, 0]

    def _point_in_triangle(self, v):
        tv = self.tp - v
        a0 = self._tri_area(tv[:, 0], tv[:, 1])
        a1 = self._tri_area(tv[:, 1], tv[:, 2])
        a2 = self._tri_area(tv[:, 2], tv[:, 0])
        asum = np.sum(np.abs(np.vstack([a0, a1, a2])), axis=0)
        # add a margin for in-triangle test
        return np.any(asum <= 1.01 * self.atot)

    def points_in_triangles(self, varray):
        return np.array([self._point_in_triangle(v) for v in varray])


def rasterize(
    pts: np.ndarray,
    tri: np.ndarray,
    method: str = "cg",
    n: int = 32,
    ext_ratio: float = 0.0,
    gc: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    rasterize triangles point cloud and returns (xg, yg, mask)
    function for interpolating regular grids

    Parameters
    ----------
    pts: np.ndarray
        nx2 array of points {(x, y)}
    tri: np.ndarray
        nx3 array of points connection {(i0, i1, i2)}
    method: str
        "cg", test a point in a triangle using barycentric coordinates
        "quick": test the distance from a point to centers of elements
        "qhull": using convex hull
    n: int
        the number of meshgrid per dimension, by default 32
    ext_ratio: float
        extend the boundary of meshgrid by ext_ratio*d, by default 0.0
    gc: bool
        grid_correction, offset xgrid and ygrid by half step size , by default
        False

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        x grid, y grid, mask

    Notes
    -----
    mask denotes points outside mesh.
    """
    xg, yg = _build_grid(pts, n=n, ext_ratio=ext_ratio, gc=gc)
    points = np.vstack((xg.flatten(), yg.flatten())).T

    # perform rasterize on meshgrids
    if method == "cg":
        TR = TriangleRasterizer(pts[:, :2], tri)
        mask = ~TR.points_in_triangles(points)
    else:
        pts_edges = _hull_points(pts)
        mask = _build_mask(pts_edges, xg, yg)

    return xg, yg, mask


def _build_grid(
    pts: np.ndarray, n: int = 32, ext_ratio: float = 0.0, gc: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generating mesh grid from triangles point cloud

    Parameters
    ----------
    pts: np.ndarray
        nx2 array of points (x, y)
    n: int
        the number of meshgrid per dimension, by default 32
    ext_ratio: float
        extend the boundary of meshgrid by ext_ratio*d, by default 0.0
    gc: bool
        grid_correction, offset xgrid and ygrid by half step size , by default
        False

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        x grid, y grid
    """
    x, y = pts[:, 0], pts[:, 1]
    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y), max(y)
    x_ext = (x_max - x_min) * ext_ratio
    y_ext = (y_max - y_min) * ext_ratio
    xv, xv_step = np.linspace(
        x_min - x_ext, x_max + x_ext, num=n, endpoint=False, retstep=True
    )
    yv, yv_step = np.linspace(
        y_min - y_ext, y_max + y_ext, num=n, endpoint=False, retstep=True
    )
    # if need grid correction
    if gc:
        xv = xv + xv_step / 2.0
        yv = yv + yv_step / 2.0
    xg, yg = np.meshgrid(xv, yv, sparse=False, indexing="xy")
    return xg, yg


def _build_mask(pts_edges: np.ndarray, xg: np.ndarray, yg: np.ndarray):
    """
    find whether meshgrids is interior of mesh

    Parameters
    ----------
    pts_edges : np.ndarray
        points on the edges of the mesh
    xg : np.ndarray
        x grid
    yg : np.ndarray
        x grid

    Returns
    -------
    np.ndarray
        mask (denotes points outside mesh.)
    """
    # 1. create mask based on meshes
    points = np.vstack((xg.flatten(), yg.flatten())).T

    # 2. extract edge points using el_pos
    path = Path(pts_edges, closed=False)
    mask = path.contains_points(points)
    return ~mask


def _hull_points(pts: np.ndarray) -> np.ndarray:
    """
    return the convex hull points from a point cloud

    Parameters
    ----------
    pts: np.ndarray
        nx2 array of points (x, y) (can be also (x,y,z))

    Returns
    -------
    np.ndarray
        convex hull points (edge points)
    """
    pts_2D = pts[:, :2]  # get only x and y
    cv = ConvexHull(pts_2D)
    hull_nodes = cv.vertices
    return pts_2D[hull_nodes, :]


def _distance2d(x: np.ndarray, y: np.ndarray, center: Union[str, list] = "mean"):
    """
    Calculate radius given center.
    This function can be OPTIMIZED using numba or cython.

    Parameters
    ----------
    x : np.ndarray
        nx1 array of x coordiante
    y : np.ndarray
        nx1 array of y coordiante
    center : Tuple[str, list], optional
        center definition, by default "mean".
        If center is `None`, [0,0] will be used.
        If center is "mean", [np.mean(x), np.mean(y)] will be used.
        If center is list, [center[0], center[1]] will be used.

    Returns
    -------
    np.ndarray
        nx1 array of distance from center to points (x,y)
    """
    if center is None:
        xc, yc = 0, 0
    elif center == "mean":
        xc, yc = np.mean(x), np.mean(y)
    else:
        xc, yc = center[0], center[1]
    # distance 2d
    return np.sqrt((x - xc) ** 2 + (y - yc) ** 2).ravel()


def _distance_matrix2d(xy: np.ndarray, xyi: np.ndarray) -> np.ndarray:
    """
    (2D only) return element-wise distance matrix (pair-wise)

    Parameters
    ----------
    xy : np.ndarray
        nx2 array of points (x, y)
    xyi : np.ndarray
        points pairs

    Returns
    -------
    np.ndarray
        distance matrix between pairwise observations
    """
    # Make a distance matrix between pairwise observations
    # Note: from <http://stackoverflow.com/questions/1871536>
    # (Yay for ufuncs!)
    d0 = np.subtract.outer(xy[:, 0], xyi[:, 0])  # size(xy) * size(xyi)
    d1 = np.subtract.outer(xy[:, 1], xyi[:, 1])

    # hypot : element-wise sqrt(d0**2 + d1**2)
    return np.hypot(d0, d1)


def weight_sigmod(
    xy: np.ndarray, xyi: np.ndarray, ratio: float = 0.05, s: float = 20.0
):
    """
    (2D only)
    local weight/interpolate by sigmod function (GREIT3D)

    Parameters
    ----------
    xy: np.ndarray
        (x, y) of values
    xyi: np.ndarray
        (xi, yi) of interpolated locations
    ratio: float
        R0 = d_max * ratio, by default 0.05.
    s: float
        control the decay ratio, by default 20.0.

    Returns
    -------
    w_mat: np.ndarray
        weighting matrix mapping from xy to xyi (xy meshgrid)
    """
    d_mat = _distance_matrix2d(xy, xyi)
    # normalize distance
    d_max = np.max(d_mat)
    d_mat = 5.0 * d_mat / d_max
    # desired radius (a ratio of max pairwise distance)
    r0 = 5.0 * ratio
    # weights is the sigmod function
    weight = 1.0 / (1 + np.exp(s * (d_mat - r0)))
    # weighting matrix normalized
    return weight / weight.sum(axis=0)


def weight_idw(xy: np.ndarray, xyi: np.ndarray, k: int = 6, p: float = 1.0):
    """
    (2D only)
    local weight/interpolate by inverse distance

    Parameters
    ----------
    xy: np.ndarray
        (x, y) of values
    xyi: np.ndarray
        (xi, yi) of interpolated locations
    k: int
        number of nearest neighbores, by default 6.
    p: float
        scaling distance, by default 1.0.

    Returns
    -------
    w_mat: np.ndarray
        weighting matrix mapping from xy to xy_mesh
    """
    d_mat = _distance_matrix2d(xy, xyi)
    # weight = 1.0 / d_mat**p
    weight = 1.0 / d_mat**p
    # keep only k largest neighbores (nearest)
    for w in weight.T:
        sort_indices = np.argsort(w)
        np.put(w, sort_indices[:-k], 0)
    # weighting matrix normalized
    # xy times xyi size, use w_mat.T to multiply
    return weight / weight.sum(axis=0)


def weight_linear_rbf(xy: np.ndarray, xyi: np.ndarray, z: np.ndarray):
    """
    (2D only)
    local weight/interpolate by linear rbf function (z value required)

    Parameters
    ----------
    xy: np.ndarray
        (x, y) of values
    xyi: np.ndarray
        (xi, yi) of interpolated locations
    z: np.ndarray
        z values

    Returns
    -------
    w_mat: np.ndarray
        weighting matrix mapping from xy to xy_mesh
    """
    internal_dist = _distance_matrix2d(xy, xy)
    weights = la.solve(internal_dist, z)
    interp_dist = _distance_matrix2d(xy, xyi)
    return np.dot(interp_dist.T, weights)


def weight_barycentric_gradient():
    """
    (2D only)
    local weight/interpolate by barycentric gradient

    Parameters
    ----------
    xy: np.ndarray
        (x, y) of values
    xyi: np.ndarray
        (xi, yi) of interpolated locations

    Returns
    -------
    w_mat: np.ndarray
        weighting matrix mapping from xy to xy_mesh
    """
    raise NotImplementedError()


def sim2pts(pts: np.ndarray, sim: np.ndarray, sim_values: np.ndarray):
    """
    (2D/3D) compatible.

    Interp values on points using values on simplex,
    a simplex can be triangle or tetrahedron.
    The areas/volumes are used as weights.

    f_n = (sum_e r_e*S_e) / (sum_e S_e)

    where r_e is the value on triangles who share the node n,
    S_e is the area of triangle e.

    Parameters
    ----------
    pts_values: np.ndarray
        Nx1 array, real/complex valued
    sim: np.ndarray
        Mx3, Mx4 array, elements or simplex
        triangles denote connectivity [[i, j, k]]
        tetrahedrons denote connectivity [[i, j, m, n]]
    sim_value: np.ndarray

    Notes
    -----
    This function is similar to pdeprtni of MATLAB pde.
    """
    N = pts.shape[0]
    M, dim = sim.shape
    # calculate the weights
    # triangle/tetrahedron must be CCW (recommended), then a is positive
    if dim == 3:
        weight_func = tri_area
    elif dim == 4:
        weight_func = tet_volume
    weights = weight_func(pts, sim)
    # build tri->pts matrix, could be accelerated using sparse matrix
    row = np.ravel(sim)
    col = np.repeat(np.arange(M), dim)  # [0, 0, 0, 1, 1, 1, ...]
    data = np.repeat(weights, dim)
    e2n_map = coo_matrix((data, (row, col)), shape=(N, M)).tocsr()
    # map values from elements to nodes
    # and re-weight by the sum of the areas/volumes of adjacent elements
    f = e2n_map.dot(sim_values)
    w = np.sum(e2n_map.toarray(), axis=1)

    return f / w


def pts2sim(sim: np.ndarray, pts_values: np.ndarray):
    """
    (2D/3D) compatible.

    Given values on nodes, calculate interpolated values on simplex,
    this function was tested and equivalent to MATLAB 'pdeintrp'
    except for the shapes of 'pts' and 'tri'

    Parameters
    ----------
    sim: np.ndarray
        Mx3, Mx4 array, elements or simplex
        triangles denote connectivity [[i, j, k]]
        tetrahedrons denote connectivity [[i, j, m, n]]
    pts_values: np.ndarray
        Nx1 array, real/complex valued

    Returns
    -------
    el_value: np.ndarray
        Mx1 array, real/complex valued

    Notes
    -----
    This function is similar to pdfinterp of MATLAB pde.
    """
    # averaged over 3 nodes of a triangle
    return np.mean(pts_values[sim], axis=1)


def tri_area(pts: np.ndarray, sim: np.ndarray) -> np.ndarray:
    """
    calculate the area of each triangle

    Parameters
    ----------
    pts: np.ndarray
        Nx2 array, (x,y) locations for points (can be also (x,y,z))
    sim: np.ndarray
        Mx3 array, elements (triangles) connectivity

    Returns
    -------
    a: np.ndarray
        Areas of triangles
    """
    pts_2D = pts[:, :2]  # get only x and y
    a = np.zeros(np.shape(sim)[0])
    for i, e in enumerate(sim):
        xy = pts_2D[e]
        # which can be simplified to
        # s = xy[[2, 0, 1]] - xy[[1, 2, 0]]
        s = xy[[2, 0]] - xy[[1, 2]]

        # a should be positive if triangles are CCW arranged
        a[i] = la.det(s)

    return a * 0.5


def tet_volume(pts: np.ndarray, sim: np.ndarray) -> np.ndarray:
    """
    calculate the area of each triangle

    Parameters
    ----------
    pts: np.ndarray
        Nx3 array, (x,y, z) locations for points
    sim: np.ndarray
        Mx4 array, elements (tetrahedrons) connectivity

    Returns
    -------
    v: np.ndarray
        Volumes of tetrahedrons
    """
    v = np.zeros(np.shape(sim)[0])
    for i, e in enumerate(sim):
        xyz = pts[e]
        s = xyz[[2, 3, 0]] - xyz[[1, 2, 3]]

        # a should be positive if triangles are CCW arranged
        v[i] = la.det(s)

    return v / 6.0


def pdetrg(pts: np.ndarray, tri: np.ndarray) -> Tuple[Any, Any, Any]:
    """
    (Deprecated)
    analytical calculate the Area and grad(phi_i) using
    barycentric coordinates (simplex coordinates)
    this function is tested and equivalent to MATLAB 'pdetrg'
    except for the shape of 'pts' and 'tri' and the output

    note: each node may have multiple gradients in neighbor
    elements' coordinates. you may averaged all the gradient to
    get one node gradient.

    Parameters
    ----------
    pts: np.ndarray
        Nx2 array, (x,y) locations for points
    tri: np.ndarray
        Mx3 array, elements (triangles) connectivity

    Returns
    -------
    a: np.ndarray
        Mx1 array, areas of elements
    grad_phi_x: np.ndarray
        Mx3 array, x-gradient on elements' local coordinate
    grad_phi_y: np.ndarray
        Mx3 array, y-gradient on elements' local coordinate
    """
    m = np.size(tri, 0)

    ix = tri[:, 0]
    iy = tri[:, 1]
    iz = tri[:, 2]

    s1 = pts[iz, :] - pts[iy, :]
    s2 = pts[ix, :] - pts[iz, :]
    s3 = pts[iy, :] - pts[ix, :]

    a = 0.5 * (s2[:, 0] * s3[:, 1] - s3[:, 0] * s2[:, 1])
    if any(a) < 0:
        raise ValueError("Triangles are not in CCW order")

    # note in python, reshape place elements first on the right-most index
    grad_phi_x = np.reshape(
        [-s1[:, 1] / (2.0 * a), -s2[:, 1] / (2.0 * a), -s3[:, 1] / (2.0 * a)], [-1, m]
    ).T
    grad_phi_y = np.reshape(
        [s1[:, 0] / (2.0 * a), s2[:, 0] / (2.0 * a), s3[:, 0] / (2.0 * a)], [-1, m]
    ).T

    return a, grad_phi_x, grad_phi_y


def pdegrad(
    pts: np.ndarray, tri: np.ndarray, node_value: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    (Deprecated)
    given values on nodes, calculate the averaged-grad on elements
    this function was tested and equivalent to MATLAB 'pdegrad'
    except for the shape of 'pts', 'tri'

    Parameters
    ----------
    pts: np.ndarray
        Nx2 array, (x,y) locations for points
    tri: np.ndarray
        Mx3 array, elements (triangles) connectivity
    node_value: np.ndarray
        Nx1 array, real/complex valued

    Returns
    -------
    el_grad: np.ndarray
        el_grad, Mx2 array, real/complex valued
    """
    m = np.size(tri, 0)
    _, grad_phi_x, grad_phi_y = pdetrg(pts, tri)
    tri_values = np.reshape(node_value[tri.ravel()], [m, -1])
    grad_el_x = np.sum(grad_phi_x * tri_values, axis=1)
    grad_el_y = np.sum(grad_phi_y * tri_values, axis=1)
    return grad_el_x, grad_el_y


def demo() -> None:
    from pyeit.mesh.wrapper import PyEITAnomaly_Circle

    """demo shows how to interpolate on regular/irregular grids"""
    # 1. create mesh
    mesh_obj = layer_circle(n_layer=8, n_fan=6)
    pts = mesh_obj.node
    tri = mesh_obj.element

    # set anomaly
    anomaly = PyEITAnomaly_Circle(center=[0.5, 0.5], r=0.2, perm=100.0)
    mesh_new = set_perm(mesh_obj, anomaly=anomaly)

    # 2. interpolate using averaged neighbor triangle area
    perm_node = sim2pts(pts, tri, mesh_new.perm_array)

    # plot mesh and interpolated mesh (tri2pts)
    fig_size = (6, 4)
    fig = plt.figure(figsize=fig_size, dpi=200)
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")
    ax.triplot(pts[:, 0], pts[:, 1], tri)
    ax.set_title("mesh_obj and anomaly")
    im1 = ax.tripcolor(pts[:, 0], pts[:, 1], tri, mesh_new.perm, alpha=0.8)
    fig.colorbar(im1, orientation="vertical")

    fig = plt.figure(figsize=fig_size, dpi=200)
    ax2 = fig.add_subplot(111)
    ax2.set_aspect("equal")
    ax2.triplot(pts[:, 0], pts[:, 1], tri)
    ax2.set_title("mesh_obj and anomaly on nodes")
    im2 = ax2.tripcolor(pts[:, 0], pts[:, 1], tri, perm_node, shading="flat")
    fig.colorbar(im2, orientation="vertical")

    # 3. interpolate on grids (irregular or regular) using IDW, sigmod
    xg, yg, mask = rasterize(pts, tri)
    im = np.ones_like(mask)
    # mapping from values on xy to values on xyi
    xy = np.mean(pts[tri], axis=1)
    xyi = np.vstack((xg.flatten(), yg.flatten())).T
    # w_mat = weight_idw(xy, xyi)
    w_mat = weight_sigmod(xy, xyi)
    im = np.dot(w_mat.T, mesh_new.perm)
    # im = weight_linear_rbf(xy, xyi, mesh_new['perm'])
    im[mask] = 0.0
    # reshape to grid size
    im = im.reshape(xg.shape)

    # plot interpolated values
    fig, ax = plt.subplots(figsize=fig_size, dpi=200)
    ax.set_aspect("equal")
    ax.triplot(pts[:, 0], pts[:, 1], tri, alpha=0.5)
    ax.set_title("mesh_obj and anomaly rasterized")
    im3 = ax.pcolor(xg, yg, im, edgecolors=None, linewidth=0, alpha=0.8)
    fig.colorbar(im3, orientation="vertical")
    plt.show()


if __name__ == "__main__":
    demo()
