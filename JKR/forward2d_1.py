# coding: utf-8
""" forward 2D """

# JKR July 2021. Based on pyeit/examples/fem_forward2d.py



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
def myrectangle(pts):
    return mesh.shape.rectangle(pts,p1=[-1,0])
n_el = 11
p_fix = np.array([[x,0] for x in np.arange(-1+1/n_el,1,2/n_el)])
mesh_obj, el_pos = mesh.create(n_el, fd=myrectangle, p_fix=p_fix, h0=0.05)

# extract node, element, alpha
pts = mesh_obj["node"]
tri = mesh_obj["element"]
x, y = pts[:, 0], pts[:, 1]
quality.stats(pts, tri)

# change permittivity
anomaly = [{"x": 0.10, "y": 0.25, "d": 0.20, "perm": 10}]
mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly, background=1.0)
perm = mesh_new["perm"]

""" 1. FEM forward simulations """
# setup EIT scan conditions
#ex_dist, step = 1, 3
#ex_mat = eit_scan_lines(16, ex_dist)
ex_mat = np.array( [ [4,9],
                     [1,6],
                     [2,6],
                     [3,6],
                     [4,6],
                     [5,6],
                     [7,6],
                     [8,6],
                     [9,6],
                     [10,6],
                     [11,6]
                     ] )
ex_line = ex_mat[0].ravel()

# calculate simulated data using FEM
fwd = Forward(mesh_obj, el_pos)
f, _ = fwd.solve(ex_line, perm=perm)
f = np.real(f)


# calculate the gradient to plot electric field lines
from matplotlib.tri import (
    Triangulation, CubicTriInterpolator)
triang = Triangulation(x, y, triangles=tri)
tci = CubicTriInterpolator(triang, -f)
# Gradient requested here at the mesh nodes but could be anywhere else:
(Ex, Ey) = tci.gradient(triang.x, triang.y)
E_norm = np.sqrt(Ex**2 + Ey**2)



""" 2. plot """
fig = plt.figure()
ax1 = fig.add_subplot(121)
# draw equi-potential lines
vf = np.linspace(min(f), max(f), 16)   # list of contour voltages
ax1.tricontour(x, y, tri, f, vf, cmap=plt.cm.viridis)
# draw mesh structure
ax1.tripcolor(
    x,
    y,
    tri,
    np.real(perm),
    edgecolors="k",
    shading="flat",
    alpha=0.2,
    cmap=plt.cm.Greys,
)
# draw electrodes
ax1.plot(x[el_pos], y[el_pos], "ro")
for i, e in enumerate(el_pos):
    ax1.text(x[e], y[e], str(i + 1), size=12)
ax1.set_title("equi-potential lines")
# clean up
ax1.set_aspect("equal")
ax1.set_ylim([-0.2, 1.2])
ax1.set_xlim([-1.2, 1.2])


ax2 = fig.add_subplot(122)
# draw equi-potential lines
E_norm_list = np.linspace(min(E_norm), max(E_norm), 16)   # list of contour voltages
ax2.tricontour(x, y, tri, E_norm, E_norm_list, cmap=plt.cm.Reds_r)
# draw mesh structure
ax2.tripcolor(
    x,
    y,
    tri,
    np.real(perm),
    edgecolors="k",
    shading="flat",
    alpha=0.2,
    cmap=plt.cm.Greys,
)
# draw electrodes
ax2.plot(x[el_pos], y[el_pos], "ro")
for i, e in enumerate(el_pos):
    ax2.text(x[e], y[e], str(i + 1), size=12)
ax2.set_title("estimated electric field lines")
# clean up
ax2.set_aspect("equal")
ax2.set_ylim([-0.2, 1.2])
ax2.set_xlim([-1.2, 1.2])

fig.set_size_inches(12, 12)
# fig.savefig('demo_bp.png', dpi=96)
plt.show()
