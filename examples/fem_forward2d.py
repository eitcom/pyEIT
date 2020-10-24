# coding: utf-8
""" demo on forward 2D """
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import division, absolute_import, print_function

import numpy as np
import matplotlib.pyplot as plt

import pyeit.mesh as mesh
from pyeit.mesh import quality
from pyeit.eit.fem import Forward
from pyeit.eit.utils import eit_scan_lines

""" 0. build mesh """
mesh_obj, el_pos = mesh.create(16, h0=0.08)

# extract node, element, alpha
pts = mesh_obj["node"]
tri = mesh_obj["element"]
x, y = pts[:, 0], pts[:, 1]
quality.stats(pts, tri)

# change permittivity
anomaly = [{"x": 0.40, "y": 0.50, "d": 0.20, "perm": 100.0}]
mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly, background=1.0)
perm = mesh_new["perm"]

""" 1. FEM forward simulations """
# setup EIT scan conditions
ex_dist, step = 7, 1
ex_mat = eit_scan_lines(16, ex_dist)
ex_line = ex_mat[0].ravel()

# calculate simulated data using FEM
fwd = Forward(mesh_obj, el_pos)
f, _ = fwd.solve(ex_line, perm=perm)
f = np.real(f)

""" 2. plot """
fig = plt.figure()
ax1 = fig.add_subplot(111)
# draw equi-potential lines
vf = np.linspace(min(f), max(f), 32)
ax1.tricontour(x, y, tri, f, vf, cmap=plt.cm.viridis)
# draw mesh structure
ax1.tripcolor(
    x,
    y,
    tri,
    np.real(perm),
    edgecolors="k",
    shading="flat",
    alpha=0.5,
    cmap=plt.cm.Greys,
)
# draw electrodes
ax1.plot(x[el_pos], y[el_pos], "ro")
for i, e in enumerate(el_pos):
    ax1.text(x[e], y[e], str(i + 1), size=12)
ax1.set_title("equi-potential lines")
# clean up
ax1.set_aspect("equal")
ax1.set_ylim([-1.2, 1.2])
ax1.set_xlim([-1.2, 1.2])
fig.set_size_inches(6, 6)
# fig.savefig('demo_bp.png', dpi=96)
plt.show()
