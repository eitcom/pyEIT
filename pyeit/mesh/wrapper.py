# coding: utf-8
# pylint: disable=invalid-name, no-member, too-many-arguments
""" wrapper function of distmesh for EIT """
from __future__ import absolute_import

import numpy as np

from .distmesh import build
from .utils import check_order
from .shape import unit_circle, unit_ball, huniform
from .shape import pfix_fd, pfix_ball


def create(numEl=16, fd=None, fh=None, pfix=None, bbox=None, h0=0.1):
    """ wrapper for pyEIT interface

    Parameters
    ----------
    numEl : int, optional
        number of electrodes
    h0 : float, optional
        initial mesh size

    Returns
    -------
    dict
        {'element', 'node', 'alpha'}
    """
    if bbox is None:
        bbox = [[-1, -1], [1, 1]]
    # infer ndim
    bbox = np.array(bbox)
    ndim = bbox.shape[1]
    if ndim not in [2, 3]:
        raise TypeError('distmesh only support 2D or 3D')
    if bbox.shape[0] != 2:
        raise TypeError('please specify lower and upper bound of bbox')

    if ndim == 2:
        if fd is None:
            fd = unit_circle
        if pfix is None:
            pfix = pfix_fd(fd, numEl=numEl)
    elif ndim == 3:
        if fd is None:
            fd = unit_ball
        if pfix is None:
            pfix = pfix_ball(numEl=numEl)

    if fh is None:
        fh = huniform

    # 1. build mesh
    p, t = build(fd, fh, pfix=pfix, bbox=bbox, h0=h0)
    # 2. check whether t is counter-clock-wise, otherwise reshape it
    t = check_order(p, t)
    # 3. generate electrodes, the same as pfix (top numEl)
    elPos = np.arange(numEl)
    # 4. init uniform element sigma
    alpha = 1. * np.ones(t.shape[0])
    # 5. build output structure
    mesh = {'element': t,
            'node': p,
            'alpha': alpha}
    return mesh, elPos


def set_alpha(mesh, anom=None, background=None):
    """ wrapper for pyEIT interface

    Note
    ----
    update alphas of mesh structure, if specified,

    Parameters
    ----------
    mesh : dict
        mesh structure
    anom : dict, optional
        anom is a dictionary (or arrays of dictionary) contains,
        {'x': val, 'y': val, 'd': val, 'alpha': val}
        all alphas on triangles whose distance to (x,y) are less than (d)
        will be replaced with a new alpha, alpha can have a complex dtype
    background : float, optional
        set background permitivities

    Returns
    -------
    dict
        updated mesh structure
    """
    el2no = mesh['element']
    no2xy = mesh['node']
    alpha = mesh['alpha']
    tri_centers = np.mean(no2xy[el2no], axis=1)

    # this code is equivalent to:
    # >>> N = np.shape(el2no)[0]
    # >>> for i in range(N):
    # >>>     tri_centers[i] = np.mean(no2xy[el2no[i]], axis=0)
    # >>> plt.plot(tri_centers[:,0], tri_centers[:,1], 'kx')
    N = np.size(mesh['alpha'])

    # reset background if needed
    if background is not None:
        alpha = background * np.ones(N, dtype='complex')

    if anom is not None:
        for _, attr in enumerate(anom):
            d = attr['d']
            # find elements whose distance to (cx,cy) is smaller than d
            if 'z' in attr:
                indice = np.sqrt((tri_centers[:, 0] - attr['x'])**2 +
                                 (tri_centers[:, 1] - attr['y'])**2 +
                                 (tri_centers[:, 2] - attr['z'])**2) < d
            else:
                indice = np.sqrt((tri_centers[:, 0] - attr['x'])**2 +
                                 (tri_centers[:, 1] - attr['y'])**2) < d
            # update alpha within indices
            alpha[indice] = attr['alpha']

    mesh_new = {'node': no2xy,
                'element': el2no,
                'alpha': alpha}
    return mesh_new
