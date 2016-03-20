""" demo on GREIT """

import numpy as np
import matplotlib.pyplot as plt

import pyeit.mesh as mesh
from pyeit.eit.fem import forward
from pyeit.eit.utils import eit_scan_lines
import pyeit.eit.greit as greit

""" 0. construct mesh """
ms, elPos = mesh.create(16, h0=0.1)

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
ms1 = mesh.set_alpha(ms, anom=anomaly, background=1.0)
alpha = np.real(ms1['alpha'] - ms0['alpha'])

# show alpha
fig = plt.figure()
plt.tripcolor(no2xy[:, 0], no2xy[:, 1], el2no, alpha, shading='flat')
plt.colorbar()
plt.title(r'$\Delta$ Conductivity')
fig.set_size_inches(6, 4.5)

""" 2. FEM forward simulations """
# setup EIT scan conditions
elDist, step = 1, 1
exMtx = eit_scan_lines(16, elDist)

# calculate simulated data
fwd = forward(ms, elPos)
f0 = fwd.solve(exMtx, step=step, perm=ms0['alpha'])
f1 = fwd.solve(exMtx, step=step, perm=ms1['alpha'])

""" 3. Construct using GREIT
"""
eit = greit.GREIT(ms, elPos, exMtx=exMtx, step=step, parser='std')
ds = eit.solve(f1.v, f0.v)
x, y, ds = eit.mask_value(ds, mask_value=np.NAN)

fig = plt.figure()
"""
imshow will automatically set NaN (bad values) to 'w',
if you want to manually do so

import matplotlib.cm as cm
cmap = cm.gray
cmap.set_bad('w', 1.)
plt.imshow(np.real(ds), interpolation='nearest', cmap=cmap)
"""
plt.imshow(np.real(ds), interpolation='nearest')
plt.colorbar()
fig.set_size_inches(4, 3)
# fig.savefig('demo_greit.png', dpi=96)
