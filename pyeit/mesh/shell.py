# coding: utf-8
# pylint: disable=invalid-name, no-member, too-many-locals, too-many-arguments
# author: benyuan liu
""" create multi-shell mesh """
from __future__ import division, absolute_import, print_function

import numpy as np

from .shape import circle, fix_points_circle
from .distmesh import build
from .utils import check_order
from .mesh_circle import MeshCircle


def multi_shell(n_fan=8, n_layer=8, n_el=16,
                r_layer=None, alpha_layer=None):
    """
    create simple multi shell mesh

    Parameters
    ----------
    n_fan : int
        number of fans per layer
    n_layer : int
        number of layers
    n_el : int
        number of electrodes
    r_layer : NDArray
        int, anomaly layers
    alpha_layer : NDArray
        float, conductivity on each anomaly layer

    Notes
    -----
    The quality of meshes near the boundary is bad.
    (sharp angles, angle of 90, etc.)
    """
    if np.size(r_layer) != np.size(alpha_layer):
        raise ValueError('r_layer and alpha_layer must have same length')

    model = MeshCircle(n_fan=n_fan, n_layer=n_layer, n_el=n_el)
    p, e, el_pos = model.create()

    # tweak alpha
    delta_r = 1. / n_layer
    alpha = np.ones(e.shape[0], dtype=np.float)

    t_center = np.mean(p[e], axis=1)
    r_center = np.sqrt(np.sum(t_center**2, axis=1))
    for layer, a in zip(r_layer, alpha_layer):
        r0, r1 = delta_r*(layer-1), delta_r*layer
        idx = (r0 < r_center) & (r_center < r1)
        alpha[idx] = a

    # 5. build output structure
    mesh = {'element': e,
            'node': p,
            'alpha': alpha}
    return mesh, el_pos


def multi_circle(r=1., background=1., n_el=16, h0=0.006,
                 r_layer=None, alpha_layer=None, ppl=64):
    """
    create multi layer circle mesh

    Parameters
    ----------
    r : float
        radius of the circle
    background : float
        background conductivity
    n_el : int
        number of electrodes
    h0 : float
        initial area of meshes
    r_layer : NDArray
        n x 2 arrays, each row represents [r0, r1] where r0 < r < r1
    alpha_layer : NDArray
        n x 1 arrays, the conductivity on each layer
    ppl : int
        point per layer

    Notes
    -----
    Due to the size constraints the triangle mesh, layer may be discontinuous
    especially in the interior. However, the quality of meshes is superior
    to multi_shell.
    """

    if np.ndim(r_layer) != 2:
        raise ValueError('r_layer must be 2-dimension')

    if np.ndim(alpha_layer) != 1:
        raise ValueError('alpha_layer must be 1-dimension')

    if np.shape(r_layer)[0] != np.size(alpha_layer):
        raise ValueError('r_layer and alpha_layer must have same length')

    def _fd(pts):
        """ shape function """
        return circle(pts, pc=[0, 0], r=r)

    def _fh(pts):
        """ distance function """
        r2 = np.sum(pts**2, axis=1)
        return 0.6*(2.0 - r2)

    # build fix points, may be used as the position for electrodes
    p_fix = fix_points_circle(n_el=n_el)

    # 1. append fix points on layers
    for layer in r_layer:
        r0, r1 = layer
        p_fix_layer1 = r0 * r * fix_points_circle(n_el=ppl)
        p_fix_layer2 = r1 * r * fix_points_circle(n_el=ppl)
        p_fix = np.vstack([p_fix, p_fix_layer1, p_fix_layer2])

    # firs num nodes are the positions for electrodes
    el_pos = np.arange(n_el)

    # build triangle
    p, t = build(_fd, _fh, pfix=p_fix, h0=h0)

    # 2. check whether t is counter-clock-wise, otherwise reshape it
    t = check_order(p, t)

    # 3. generate electrodes, the same as p_fix (top n_el)
    el_pos = np.arange(n_el)

    # 4. init uniform element sigma
    alpha = background * np.ones(t.shape[0], dtype=np.float)
    t_center = np.mean(p[t], axis=1)
    r_center = np.sqrt(np.sum(t_center**2, axis=1))

    # update alphas
    for (layer, a) in zip(r_layer, alpha_layer):
        r0, r1 = layer
        idx = (r0 < r_center) & (r_center < r1)
        alpha[idx] = a

    # 5. build output structure
    mesh = {'element': t,
            'node': p,
            'alpha': alpha}
    return mesh, el_pos
