""" demo on bp """

import numpy as np
import matplotlib.pyplot as plt

import pyeit.mesh as mesh
from pyeit.eit.fem import forward, pdeprtni
from pyeit.eit.utils import eit_scan_lines
import pyeit.eit.bp as bp

""" 0. build mesh """
ms, elPos = mesh.create(16, h0=0.075)

# extract node, element, alpha
no2xy = ms['node']
el2no = ms['element']

""" 1. problem setup """
# test function for altering the 'alpha' in mesh dictionary
anomaly = [{'x': 0.5, 'y': 0.5, 'd': 0.1, 'alpha': 10.0}]
ms0 = mesh.set_alpha(ms, anom=anomaly, background=1.0)

# test function for altering the 'alpha' in mesh dictionary
anomaly = [{'x': 0.5, 'y': 0.5, 'd': 0.1, 'alpha': 100.0}]
ms1 = mesh.set_alpha(ms, anom=anomaly, background=1.0)

# draw
delta_alpha = np.real(ms1['alpha'] - ms0['alpha'])
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
elDist, step = 1, 1
exMtx = eit_scan_lines(16, elDist)

# calculate simulated data
fwd = forward(ms, elPos)
f0 = fwd.solve(exMtx, step=step, perm=ms0['alpha'])
f1 = fwd.solve(exMtx, step=step, perm=ms1['alpha'])

"""
3. naive inverse solver using back-projection
"""
eit = bp.BP(ms, elPos, exMtx, step=1, parser='std', weight='none')
ds = eit.solve(f1.v, f0.v, normalize=True)
ds = 10000. * pdeprtni(no2xy, el2no, ds)

# draw
fig = plt.figure()
ax1 = fig.add_subplot(111)
im = ax1.tripcolor(no2xy[:, 0], no2xy[:, 1], el2no, ds, cmap=plt.cm.viridis)
ax1.set_title(r'$\Delta$ Conductivities')
ax1.axis('equal')
fig.colorbar(im)
""" for production figures, use dpi=300 or render pdf """
fig.set_size_inches(6, 4)
# fig.savefig('demo_bp.png', dpi=96)
plt.show()
