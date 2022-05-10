# coding: utf-8
# pylint: disable=invalid-name
""" demo for (multi) shell.py """
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np
import pyeit.mesh as mesh
from pyeit.eit.fem import Forward
from pyeit.eit.protocol import build_exc_pattern_std

# (a) using multi-shell (calls layer_circle, fast, inaccurate)
n_el = 16  # nb of electrodes
n_fan = 6
n_layer = 12
r_layers = [n_layer - 1]
perm_layers = [0.01]
mesh_obj = mesh.multi_shell(
    n_el=n_el,
    n_fan=n_fan,
    n_layer=n_layer,
    r_layer=r_layers,
    perm_per_layer=perm_layers,
)

# (b) using multi-circle (calls create, slow, high-quality)
r_layers = [[0.85, 0.925]]
perm_layers = [0.01]
mesh_obj = mesh.multi_circle(
    r=1.0,
    background=1.0,
    n_el=n_el,
    h0=0.04,
    r_layer=r_layers,
    perm_per_layer=perm_layers,
    ppl=64,
)

""" 0. Visualizing mesh structure """
pts = mesh_obj.node
tri = mesh_obj.element
tri_perm = mesh_obj.perm
el_pos = mesh_obj.el_pos
figsize = (6, 6)

# plot
fig, ax = plt.subplots(figsize=figsize)
ax.triplot(pts[:, 0], pts[:, 1], tri)
ax.plot(pts[el_pos, 0], pts[el_pos, 1], "ro")
plt.axis("equal")
plt.axis([-1.5, 1.5, -1.25, 1.25])
plt.show()

fig, ax = plt.subplots(figsize=figsize)
ax.tripcolor(
    pts[:, 0], pts[:, 1], tri, tri_perm, edgecolor="k", alpha=0.6, cmap=plt.cm.Greys
)
ax.plot(pts[el_pos, 0], pts[el_pos, 1], "ro")
plt.axis("equal")
plt.axis([-1.5, 1.5, -1.25, 1.25])
plt.show()

""" 1. FEM forward simulations """
# setup EIT scan conditions
ex_mat = build_exc_pattern_std(n_el, dist=7)

# calculate simulated data
fwd = Forward(mesh_obj)

# in python, index start from 0
ex_line = ex_mat[0].ravel()

# solving once using fem
f = fwd.solve(ex_line)
f = np.real(f)
vf = np.linspace(min(f), max(f), 32)

# plot
fig = plt.figure(figsize=figsize)
ax1 = fig.add_subplot(111)
ax1.tricontour(pts[:, 0], pts[:, 1], tri, f, vf, cmap=plt.cm.viridis)
ax1.tripcolor(
    pts[:, 0],
    pts[:, 1],
    tri,
    np.real(tri_perm),
    edgecolors="k",
    shading="flat",
    alpha=0.4,
    cmap=plt.cm.Greys,
)
ax1.plot(pts[el_pos, 0], pts[el_pos, 1], "ro")
ax1.set_title("equi-potential lines")
ax1.axis("equal")
plt.axis([-1.5, 1.5, -1.25, 1.25])
# fig.set_size_inches(6, 4)
# fig.savefig('demo_bp.png', dpi=96)
plt.show()
