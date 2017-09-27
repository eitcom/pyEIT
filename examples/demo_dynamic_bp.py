# coding: utf-8
# author: benyuan liu
""" demo code for back-projection """
from __future__ import division, absolute_import, print_function

import numpy as np
import matplotlib.pyplot as plt

import pyeit.mesh as mesh
from pyeit.eit.fem import Forward
from pyeit.eit.utils import eit_scan_lines
import pyeit.eit.bp as bp

""" 0. build mesh """
ms, el_pos = mesh.create(16, h0=0.1)

# extract node, element, alpha
no2xy = ms['node']
el2no = ms['element']

""" 1. problem setup """
anomaly = [{'x': 0.5, 'y': 0.5, 'd': 0.1, 'alpha': 10.0}]
ms1 = mesh.set_alpha(ms, anomaly=anomaly, background=1.0)

# draw
delta_alpha = np.real(ms1['alpha'] - ms['alpha'])
fig, ax = plt.subplots()
im = ax.tripcolor(no2xy[:, 0], no2xy[:, 1], el2no, delta_alpha,
                  shading='flat', cmap=plt.cm.viridis)
ax.set_title(r'$\Delta$ Conductivities')
fig.colorbar(im)
ax.axis('equal')
fig.set_size_inches(6, 4)
# fig.savefig('demo_bp_0.png', dpi=96)
plt.show()

""" 2. FEM forward simulations """
# setup EIT scan conditions
el_dist, step = 1, 1
ex_mat = eit_scan_lines(16, el_dist)

# calculate simulated data
fwd = Forward(ms, el_pos)
f0 = fwd.solve(ex_mat, step=step, perm=ms['alpha'])
f1 = fwd.solve(ex_mat, step=step, perm=ms1['alpha'])

"""
3. naive inverse solver using back-projection
"""
eit = bp.BP(ms, el_pos, ex_mat=ex_mat, step=1, parser='std')
eit.setup(weight='none')
ds = 192.0 * eit.solve(f1.v, f0.v)

# plot
fig = plt.figure()
ax1 = fig.add_subplot(111)
im = ax1.tripcolor(no2xy[:, 0], no2xy[:, 1], el2no, ds, cmap=plt.cm.viridis)
ax1.set_title(r'$\Delta$ Conductivities')
ax1.axis('equal')
fig.colorbar(im)
""" for production figures, use dpi=300 or render pdf """
fig.set_size_inches(6, 4)
# fig.savefig('../figs/demo_bp.png', dpi=96)
plt.show()
