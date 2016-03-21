# coding: utf-8
# pylint: disable=invalid-name, no-member, too-many-locals
""" wrapper function of distmesh for EIT """
from __future__ import absolute_import

import numpy as np

from .distmesh import build
from .shape import unit_circle, huniform, pfix_circle


def create(numEl=16, h0=0.1, fd=unit_circle, fh=huniform):
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
    pfix = pfix_circle(numEl=numEl)
    p, t = build(fd, fh, pfix=pfix, h0=h0, Fscale=1.2)
    # electrodes are the same as pfix (top numEl)
    elPos = np.arange(numEl)
    # build output dictionary, uniform element sigma
    alpha = 1. * np.ones(t.shape[0])
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
            cx = attr['x']
            cy = attr['y']
            diameter = attr['d']
            alpha_anomaly = attr['alpha']
            # find elements whose distance to (cx,cy) is smaller than d
            indice = np.sqrt((tri_centers[:, 0] - cx)**2 +
                             (tri_centers[:, 1] - cy)**2) < diameter
            alpha[indice] = alpha_anomaly

    mesh_new = {'node': no2xy,
                'element': el2no,
                'alpha': alpha}
    return mesh_new
