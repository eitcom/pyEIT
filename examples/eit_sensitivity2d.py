# coding: utf-8
""" demo on sensitivity analysis of 2D mesh"""
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import division, absolute_import, print_function

# numeric
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# pyEIT
import pyeit.mesh as mesh
from pyeit.eit.interp2d import tri_area, sim2pts
from pyeit.mesh import quality
from pyeit.eit.fem import Forward
from pyeit.eit.utils import eit_scan_lines

""" 0. build mesh """
mesh_obj, el_pos = mesh.layer_circle(n_layer=8, n_fan=6)
# mesh_obj, el_pos = mesh.create()

# extract node, element, alpha
pts = mesh_obj["node"]
tri = mesh_obj["element"]
x, y = pts[:, 0], pts[:, 1]
quality.stats(pts, tri)


def calc_sens(fwd, ex_mat):
    """
    see Adler2017 on IEEE TBME, pp 5, figure 6,
    Electrical Impedance Tomography: Tissue Properties to Image Measures
    """
    # solving EIT problem
    p = fwd.solve_eit(ex_mat=ex_mat, parser="fmmu")
    v0 = p.v
    # normalized jacobian (note: normalize affect sensitivity)
    v0 = v0[:, np.newaxis]
    jac = p.jac / v0
    # calculate sensitivity matrix
    s = np.linalg.norm(jac, axis=0)
    ae = tri_area(pts, tri)
    s = np.sqrt(s) / ae
    assert any(s >= 0)

    se = np.log10(s)
    sn = sim2pts(pts, tri, se)
    return sn


""" 1. FEM forward setup """
# calculate simulated data using FEM
fwd = Forward(mesh_obj, el_pos)
# loop over EIT scan settings: vary the distance of stimulation nodes, AB
ex_list = [1, 2, 4, 8]
N = len(ex_list)
s = []
for ex_dist in ex_list:
    ex_mat = eit_scan_lines(16, ex_dist)
    # Note: ex_mat can also be stacked, see demo_dynamic_stack.py
    s0 = calc_sens(fwd, ex_mat)
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
    print("std (ex_dist=%d) = %f" % (ex_dist, std))
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
    ax.set_title("ex_dist=" + str(ex_dist))
    ax.set_aspect("equal")
    ax.set_ylim([-1.2, 1.2])
    ax.set_xlim([-1.2, 1.2])
    ax.axis("off")
    plt.colorbar(im)

# fig.savefig('demo_sens.png', dpi=96)
plt.show()
