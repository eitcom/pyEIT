# coding: utf-8
# pylint: disable=invalid-name
# author: benyuan liu
""" demo for (multi) shell.py """
from __future__ import division, absolute_import, print_function

import numpy as np
import matplotlib.pyplot as plt

from pyeit.mesh import multi_shell, multi_circle
from pyeit.eit.fem import Forward
from pyeit.eit.utils import eit_scan_lines


# (a) using multi-shell (fast, in-accurate)
n_fan = 6
n_layer = 16
r_layers = [n_layer-2]
alpha_layers = [0.01]
ms, el_pos = multi_shell(n_fan=n_fan, n_layer=n_layer,
                         r_layer=r_layers, alpha_layer=alpha_layers)

# (b) using multi-circle (slow, high-quality)
r_layers = [[0.8, 0.9]]
alpha_layers = [0.01]
ms, el_pos = multi_circle(r=1., background=1., n_el=16, h0=0.006,
                          r_layer=r_layers, alpha_layer=alpha_layers, ppl=64)

""" 0. Visualizing mesh structure """
no2xy = ms['node']
el2no = ms['element']
tri_perm = ms['alpha']

# plot
fig, ax = plt.subplots()
ax.triplot(no2xy[:, 0], no2xy[:, 1], el2no)
ax.plot(no2xy[el_pos, 0], no2xy[el_pos, 1], 'ro')
plt.axis('equal')
plt.axis([-1.5, 1.5, -1.1, 1.1])
plt.show()

fig, ax = plt.subplots()
ax.tripcolor(no2xy[:, 0], no2xy[:, 1], el2no, tri_perm, alpha=0.6,
             cmap=plt.cm.viridis)
ax.plot(no2xy[el_pos, 0], no2xy[el_pos, 1], 'ro')
plt.axis('equal')
plt.axis([-1.5, 1.5, -1.1, 1.1])
plt.show()

""" 1. FEM forward simulations """
# setup EIT scan conditions
ex_dist, step = 7, 1
ex_mat = eit_scan_lines(16, ex_dist)

# calculate simulated data
fwd = Forward(ms, el_pos)

# in python, index start from 0
ex_line = ex_mat[0].ravel()

# solving once using fem
f, _ = fwd.solve_once(ex_line, tri_perm)
f = np.real(f)
vf = np.linspace(min(f), max(f), 32)

# plot
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.tricontour(no2xy[:, 0], no2xy[:, 1], el2no, f, vf,
               linewidth=0.5, cmap=plt.cm.viridis)
ax1.tripcolor(no2xy[:, 0], no2xy[:, 1], el2no, np.real(tri_perm),
              edgecolors='k', shading='flat', alpha=0.5,
              cmap=plt.cm.Greys)
ax1.plot(no2xy[el_pos, 0], no2xy[el_pos, 1], 'ro')
ax1.set_title('equi-potential lines')
ax1.axis('equal')
fig.set_size_inches(6, 4)
# fig.savefig('demo_bp.png', dpi=96)
plt.show()
