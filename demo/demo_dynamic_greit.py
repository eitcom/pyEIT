""" demo on GREIT """

import numpy as np
import matplotlib.pyplot as plt

import pyeit.mesh as mesh
from pyeit.eit.fem import Forward
from pyeit.eit.utils import eit_scan_lines
import pyeit.eit.greit as greit

""" 0. construct mesh """
ms, el_pos = mesh.create(16, h0=0.1)

# extract node, element, alpha
no2xy = ms['node']
el2no = ms['element']

""" 1. problem setup """
# this step is not needed, actually
ms0 = mesh.set_alpha(ms, background=1.0)

# test function for altering the 'alpha' in mesh dictionary
anomaly = [{'x': 0.4,  'y': 0,    'd': 0.1, 'alpha': 10},
           {'x': -0.4, 'y': 0,    'd': 0.1, 'alpha': 10},
           {'x': 0,    'y': 0.5,  'd': 0.1, 'alpha': 0.1},
           {'x': 0,    'y': -0.5, 'd': 0.1, 'alpha': 0.1}]
ms1 = mesh.set_alpha(ms, anomaly=anomaly, background=1.0)
alpha = np.real(ms1['alpha'] - ms0['alpha'])

# show alpha
fig, ax = plt.subplots(figsize=(6, 4))
im = ax.tripcolor(no2xy[:, 0], no2xy[:, 1], el2no, alpha,
                  shading='flat', cmap=plt.cm.viridis)
fig.colorbar(im)
ax.axis('equal')
ax.set_title(r'$\Delta$ Conductivity')
# fig.set_size_inches(6, 4)

""" 2. FEM forward simulations """
# setup EIT scan conditions
el_dist, step = 1, 1
ex_mat = eit_scan_lines(16, el_dist)

# calculate simulated data
fwd = Forward(ms, el_pos)
f0 = fwd.solve(ex_mat, step=step, perm=ms0['alpha'])
f1 = fwd.solve(ex_mat, step=step, perm=ms1['alpha'])

""" 3. Construct using GREIT
"""
eit = greit.GREIT(ms, el_pos, ex_mat=ex_mat, step=step, parser='std')
eit.setup(p=0.50, lamb=1e-4)
ds = eit.solve(f1.v, f0.v)
x, y, ds = eit.mask_value(ds, mask_value=np.NAN)

# plot
"""
imshow will automatically set NaN (bad values) to 'w',
if you want to manually do so

import matplotlib.cm as cm
cmap = cm.gray
cmap.set_bad('w', 1.)
plt.imshow(np.real(ds), interpolation='nearest', cmap=cmap)
"""
fig, ax = plt.subplots(figsize=(6, 4))
im = ax.imshow(np.real(ds), interpolation='none', cmap=plt.cm.viridis)
fig.colorbar(im)
ax.axis('equal')
# fig.set_size_inches(6, 4)
# fig.savefig('../figs/demo_greit.png', dpi=96)
plt.show()
