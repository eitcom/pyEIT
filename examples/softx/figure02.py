# coding: utf-8
""" Figure02 for softx """
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import division, absolute_import, print_function

import numpy as np
import matplotlib.pyplot as plt

import pyeit.mesh as mesh
from pyeit.eit.fem import Forward
from pyeit.mesh.wrapper import PyEITAnomaly_Circle

""" 0. build mesh """
n_el = 16  # nb of electrodes
mesh_obj = mesh.create(n_el, h0=0.08)

# extract node, element, alpha
pts = mesh_obj.node
tri = mesh_obj.element
el_pos = mesh_obj.el_pos
x, y = pts[:, 0], pts[:, 1]
mesh_obj.print_stats()

# change permittivity
anomaly = PyEITAnomaly_Circle(center=[0.4, 0.5], r=0.2, perm=100.0)
mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly, background=1.0)
perm = mesh_new.perm

""" 1. FEM forward simulations """
# setup (AB) current path
ex_line = [0, 7]

# calculate simulated data using FEM
fwd = Forward(mesh_new)
f = fwd.solve(ex_line)
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
ax1.set_xlabel("x")
ax1.set_ylabel("y")

fig.set_size_inches(5, 5)
fig.subplots_adjust(top=0.975, bottom=0.02, left=0.15)
fig.savefig("figure02.pdf", dpi=300)
plt.show()
