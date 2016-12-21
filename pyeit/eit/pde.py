# coding: utf-8
# pylint: disable=invalid-name, no-member, too-many-locals
# author: benyuan liu
""" provides MATLAB compatible pde functions """
from __future__ import division, absolute_import, print_function
import numpy as np
import scipy.linalg as la


def pdeintrp(no2xy, el2no, node_value):
    """
    given values on nodes, calculate interpolated values on elements
    this function was tested and equivalent to MATLAB 'pdeintrp'
    except for the shapes of 'no2xy' and 'el2no'

    Parameters
    ----------
    no2xy : NDArray
        Nx2 array, (x,y) locations for points
    el2no : NDArray
        Mx3 array, elements (triangles) connectivity
    node_value : NDArray
        Nx1 array, real/complex valued

    Returns
    -------
    NDArray
        el_value, Mx1 array, real/complex valued
    """
    N = np.size(no2xy, 0)
    M = np.size(el2no, 0)
    # build e->n matrix, could be accelerated using sparse matrix
    e2n = np.zeros([M, N], dtype='int')
    for i in range(M):
        e2n[i, el2no[i, :]] = 1
    # for tri-mesh only
    el_value = np.dot(e2n, node_value) / 3.0
    return el_value


def pdetrg(no2xy, el2no):
    """
    analytical calculate the Area and grad(phi_i) using
    barycentric coordinates (simplex coordinates)
    this function is tested and equivalent to MATLAB 'pdetrg'
    except for the shape of 'no2xy' and 'el2no' and the output

    note: each node may have multiple gradients in neighbor
    elements' coordinates. you may averaged all the gradient to
    get one node gradient.

    Parameters
    ----------
    no2xy : NDArray
        Nx2 array, (x,y) locations for points
    el2no : NDArray
        Mx3 array, elements (triangles) connectivity

    Returns
    -------
    a : NDArray
        Mx1 array, areas of elements
    grad_phi_x : NDArray
        Mx3 array, x-gradient on elements' local coordinate
    grad_phi_y : NDArray
        Mx3 array, y-gradient on elements' local coordinate
    """
    m = np.size(el2no, 0)

    ix = el2no[:, 0]
    iy = el2no[:, 1]
    iz = el2no[:, 2]

    s1 = no2xy[iz, :] - no2xy[iy, :]
    s2 = no2xy[ix, :] - no2xy[iz, :]
    s3 = no2xy[iy, :] - no2xy[ix, :]

    a = 0.5*(s2[:, 0]*s3[:, 1] - s3[:, 0]*s2[:, 1])
    if any(a) < 0:
        exit("nodes are given in clockwise manner")

    # note in python, reshape place elements first on the right-most index
    grad_phi_x = np.reshape([-s1[:, 1] / (2. * a),
                             -s2[:, 1] / (2. * a),
                             -s3[:, 1] / (2. * a)], [-1, m]).T
    grad_phi_y = np.reshape([s1[:, 0] / (2. * a),
                             s2[:, 0] / (2. * a),
                             s3[:, 0] / (2. * a)], [-1, m]).T

    return a, grad_phi_x, grad_phi_y


def pdegrad(no2xy, el2no, node_value):
    """
    given values on nodes, calculate the averaged-grad on elements
    this function was tested and equivalent to MATLAB 'pdegrad'
    except for the shape of 'no2xy', 'el2no'

    Parameters
    ----------
    no2xy : NDArray
        Nx2 array, (x,y) locations for points
    el2no : NDArray
        Mx3 array, elements (triangles) connectivity
    node_value : NDArray
        Nx1 array, real/complex valued

    Returns
    -------
    NDArray
        el_grad, Mx2 array, real/complex valued
    """
    m = np.size(el2no, 0)
    _, grad_phi_x, grad_phi_y = pdetrg(no2xy, el2no)
    tri_values = np.reshape(node_value[el2no.ravel()], [m, -1])
    grad_el_x = np.sum(grad_phi_x * tri_values, axis=1)
    grad_el_y = np.sum(grad_phi_y * tri_values, axis=1)
    return grad_el_x, grad_el_y


def pdeprtni(no2xy, el2no, el_value):
    """
    Notes
    -----
    given values on elements, interpolate values on nodes
    this code was tested and equivalent to MATLAB 'pdeprtni'
    except for the shape of 'no2xy' and 'el2no'
    ps: prtni is the reverse of interp :)

    Parameters
    ----------
    no2xy : NDArray
        Nx2 array, (x,y) locations for points
    el2no : NDArray
        Mx3 array, elements (triangles) connectivity
    el_value : NDArray
        Mx1 value, real/complex valued on elements

    Returns
    -------
    NDArray
        no_value, piecewise reverse-interpolate of el_value on nodes
    """
    n = np.size(no2xy, 0)
    m = np.size(el2no, 0)
    # build n->e matrix, this could be accelerated using sparse matrix
    n2e = np.zeros([n, m], dtype='int')
    for i in range(m):
        n2e[el2no[i, :], i] = 1
    # equivalent to,
    # pick a node, find all the triangles sharing this node,
    # and average all the values on these triangles
    node_value = np.dot(n2e, el_value) / np.sum(n2e, axis=1)
    return node_value


def pde_area(no2xy, el2no):
    """
    calculate the area of each triangle

    Parameters
    ----------
    no2xy : NDArray
        Nx2 array, (x,y) locations for points
    el2no : NDArray
        Mx3 array, elements (triangles) connectivity

    Returns
    -------
    NDArray
        a, areas of triangles
    """
    a = np.zeros(np.shape(el2no)[0])
    for i, e in enumerate(el2no):
        xy = no2xy[e]
        # s1 = xy[2, :] - xy[1, :]
        # s2 = xy[0, :] - xy[2, :]
        # s3 = xy[1, :] - xy[0, :]
        s = xy[[2, 0, 1]] - xy[[1, 2, 0]]

        #
        a[i] = 0.5 * la.det(s[[0, 1]])

    return a
