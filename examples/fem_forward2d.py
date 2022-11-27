# coding: utf-8
""" demo on forward 2D """
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np
import pyeit.eit.protocol as protocol
import pyeit.mesh as mesh
from pyeit.eit.fem import Forward
from pyeit.mesh.shape import thorax
from pyeit.mesh.wrapper import PyEITAnomaly_Circle
from pyeit.eit.interp2d import sim2pts, pdegrad

""" 0. build mesh """
n_el = 16  # nb of electrodes
use_customize_shape = False
if use_customize_shape:
    # Mesh shape is specified with fd parameter in the instantiation, e.g : fd=thorax
    mesh_obj = mesh.create(n_el, h0=0.1, fd=thorax)
else:
    mesh_obj = mesh.create(n_el, h0=0.05)
el_pos = mesh_obj.el_pos

# extract node, element, alpha
pts = mesh_obj.node
tri = mesh_obj.element
x, y = pts[:, 0], pts[:, 1]
mesh_obj.print_stats()

# change permittivity
anomaly = PyEITAnomaly_Circle(center=[0.4, 0.5], r=0.2, perm=100.0)
mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly, background=1.0)
perm = mesh_new.perm

""" 1. FEM forward simulations """
# setup EIT scan conditions
protocol_obj = protocol.create(n_el, dist_exc=7, step_meas=1, parser_meas="std")

# Define electrode current sink and current source
ex_line = protocol_obj.ex_mat[0].ravel()

# calculate simulated data using FEM
fwd = Forward(mesh_new)
f = fwd.solve(ex_line)
f = np.real(f)

""" 2. plot """
fig, ax1 = plt.subplots(figsize=(9, 6))
# draw equi-potential lines
vf = np.linspace(min(f), max(f), 32)
# vf = np.sort(f[el_pos])
# Draw contour lines on an unstructured triangular grid.
ax1.tricontour(x, y, tri, f, vf, cmap=plt.cm.viridis)

# draw mesh structure
# Create a pseudocolor plot of an unstructured triangular grid
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

ux, uy = pdegrad(pts, tri, f)
uf = ux**2 + uy**2
uf_pts = sim2pts(pts, tri, uf)
uf_logpwr = 10 * np.log10(uf_pts)

fig, ax = plt.subplots(figsize=(9, 6))
# Draw contour lines on an unstructured triangular grid.
ax.tripcolor(x, y, tri, uf_logpwr, cmap=plt.cm.viridis)
ax.tricontour(x, y, tri, uf_logpwr, 10, cmap=plt.cm.hot)
ax.set_aspect("equal")
ax.set_ylim([-1.2, 1.2])
ax.set_xlim([-1.2, 1.2])
ax.set_title("E field (logmag)")
plt.show()
