# coding: utf-8
""" demo on sensitivity analysis of 2D mesh"""
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import absolute_import, division, print_function

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pyeit.eit.protocol as protocol
import pyeit.mesh as mesh
from pyeit.eit.fem import EITForward
from pyeit.eit.interp2d import sim2pts, tri_area

""" 0. build mesh """
# Mesh shape is specified with fd parameter in the instantiation, e.g : fd=thorax , Default :fd=circle
n_el = 16  # nb of electrodes
mesh_obj = mesh.layer_circle(n_el, n_layer=8, n_fan=6)

# extract node, element, alpha
pts = mesh_obj.node
tri = mesh_obj.element
x, y = pts[:, 0], pts[:, 1]
mesh_obj.print_stats()


def calc_sens(fwd: EITForward):
    """
    see Adler2017 on IEEE TBME, pp 5, figure 6,
    Electrical Impedance Tomography: Tissue Properties to Image Measures
    """
    # solving EIT problem
    jac, v0 = fwd.compute_jac()
    # normalized jacobian (note: normalize affect sensitivity)
    v0 = v0[:, np.newaxis]
    jac = jac  # / v0  # (normalize or not)
    # calculate sensitivity matrix
    s = np.linalg.norm(jac, axis=0)
    ae = tri_area(pts, tri)
    s = np.sqrt(s) / ae
    assert any(s >= 0)

    se = np.log10(s)
    return sim2pts(pts, tri, se)


""" 1. FEM forward setup """

# loop over EIT scan settings: vary the distance of stimulation nodes, AB
ex_list = [1, 2, 4, 8]
N = len(ex_list)
s = []
for ex_dist in ex_list:
    # setup EIT scan conditions
    protocol_obj = protocol.create(
        n_el, dist_exc=ex_dist, step_meas=1, parser_meas="fmmu"
    )
    # calculate simulated data using FEM with different protocol
    fwd = EITForward(mesh_obj, protocol_obj)
    # Note: ex_mat can also be stacked, see demo_dynamic_stack.py
    s0 = calc_sens(fwd)
    s.append(s0)

""" 2. Plot (elements) sensitivity """
vmin = np.min(s)
vmax = np.max(s)
fig = plt.figure(figsize=(12, 2.5))
gs = gridspec.GridSpec(1, N)
for ix in range(N):
    ax = fig.add_subplot(gs[ix])
    sn = s[ix]
    ex_dist = ex_list[ix]
    # statistics, it seems like ex_dist=4 yields the minimal std
    std = np.std(sn)
    print(f"std ({ex_dist=}) = {std}")
    im = ax.tripcolor(
        x,
        y,
        tri,
        sn,
        edgecolors="none",
        shading="gouraud",
        cmap=plt.cm.Reds,
        antialiased=True,
        vmin=vmin,
        vmax=vmax,
    )
    # annotate
    ax.set_title(f"ex_dist={str(ex_dist)}")
    ax.set_aspect("equal")
    ax.set_ylim([-1.2, 1.2])
    ax.set_xlim([-1.2, 1.2])
    ax.axis("off")
    plt.colorbar(im)

# fig.savefig('demo_sens.png', dpi=96)
plt.show()
