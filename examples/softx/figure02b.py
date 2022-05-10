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
from pyeit.eit.fem import EITForward
import pyeit.eit.protocol as protocol

""" 0. build mesh """
# mesh_obj, el_pos = mesh.layer_circle(n_layer=8, n_fan=6)
n_el = 16  # nb of electrodes
mesh_obj = mesh.create(n_el, h0=0.05)

# extract node, element, alpha
pts = mesh_obj.node
tri = mesh_obj.element
el_pos = mesh_obj.el_pos
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
    jac = jac  # / v0
    # calculate sensitivity matrix
    s = np.linalg.norm(jac, axis=0)
    ae = tri_area(pts, tri)
    s = np.sqrt(s) / ae
    assert any(s >= 0)

    se = np.log10(s)
    return sim2pts(pts, tri, se)


""" 1. FEM forward setup """
# loop over EIT scan settings: vary the distance of stimulation nodes, AB
ex_list = [1, 2, 5, 8]
N = len(ex_list)
s = []
for ex_dist in ex_list:
    protocol_obj = protocol.create(
        n_el, dist_exc=ex_dist, step_meas=1, parser_meas="fmmu"
    )
    fwd = EITForward(mesh_obj, protocol_obj)
    # TODO: ex_mat can also be stacked, see eit_dynamic_stack.py
    s0 = calc_sens(fwd)
    s.append(s0)

""" 2. Plot (elements) sensitivity """
vmax = np.max(s)
# vmin = np.min(s)
vmin = vmax - vmax * 0.5
fig = plt.figure(figsize=(9, 3))
gs = gridspec.GridSpec(1, N)
ax_array = []
for ix in range(N):
    ax = fig.add_subplot(gs[0, ix])
    ax_array.append(ax)
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
    ax.set_title("skip=" + str(ex_dist - 1))
    ax.set_aspect("equal")
    ax.set_ylim([-1.2, 1.2])
    ax.set_xlim([-1.2, 1.2])
    ax.axis("off")

plt.colorbar(im, ax=ax_array, orientation="horizontal", shrink=0.7)

# fig.savefig('demo_sens.png', dpi=96)
# fig.set_size_inches(5, 5)
# fig.tight_layout()
fig.subplots_adjust(top=0.975, bottom=0.275, left=0.01, right=0.975)
fig.savefig("figure02b.pdf", dpi=300)
plt.show()
