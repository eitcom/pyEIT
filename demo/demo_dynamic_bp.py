""" demo on bp """

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from pyeit.mesh import distmesh2d
from pyeit.eit.fem import forward, pdeprtni
from pyeit.eit.utils import eit_scan_lines
import pyeit.eit.bp as bp

""" 0. build mesh """
ms, elPos = distmesh2d.create(16, h0=0.08)

# extract node, element, alpha
no2xy = ms['node']
el2no = ms['element']

""" 1. problem setup """
# test function for altering the 'alpha' in mesh dictionary
anomaly = [{'x': 0.5, 'y': 0.5, 'd': 0.1, 'alpha': 10.0}]
ms0 = distmesh2d.set_alpha(ms, anom=anomaly, background=1.0)

# test function for altering the 'alpha' in mesh dictionary
anomaly = [{'x': 0.5, 'y': 0.5, 'd': 0.1, 'alpha': 100.0}]
ms1 = distmesh2d.set_alpha(ms, anom=anomaly, background=1.0)

# draw delta alpha
delta_alpha = np.real(ms1['alpha'] - ms0['alpha'])
fig, ax = plt.subplots()
im = ax.tripcolor(no2xy[:, 0], no2xy[:, 1], el2no, delta_alpha,
                  shading='flat')
ax.set_title(r'$\Delta$ Conductivities')
fig.colorbar(im)
fig.set_size_inches(6, 4.5)
plt.show()

""" 2. FEM forward simulations """
# setup EIT scan conditions
elDist, step = 1, 1
exMtx = eit_scan_lines(16, elDist)

# calculate simulated data
fwd = forward(ms, elPos)
f0 = fwd.solve(exMtx, step=step, perm=ms0['alpha'])
f1 = fwd.solve(exMtx, step=step, perm=ms1['alpha'])

""" 2.1 show equi-potential lines """

# in python, index start from 0
exLine = exMtx[1].ravel()

# change alpha
anomaly = [{'x': 0.5, 'y': 0.5, 'd': 0.1, 'alpha': 100.0}]
ms_test = distmesh2d.set_alpha(ms, anom=anomaly, background=1.0)
tri_perm = ms_test['alpha']

# solving once using fem
f, _ = fwd.solve_once(exLine, tri_perm)

# draw
fig = plt.figure()
gs = gridspec.GridSpec(1, 2)
# subplot
ax1 = fig.add_subplot(gs[0, 0])
ax1.tricontour(no2xy[:, 0], no2xy[:, 1], el2no, np.real(f),
               60, linewidth=2,
               cmap=plt.cm.viridis)
ax1.tripcolor(no2xy[:, 0], no2xy[:, 1], el2no, np.real(tri_perm),
              edgecolors='k', shading='flat', alpha=0.5,
              cmap=plt.cm.Greys)
ax1.plot(no2xy[elPos, 0], no2xy[elPos, 1], 'ro')
ax1.set_title('equi-potential lines')
ax1.axis('equal')
ax1.axis([-1, 1, -1, 1])

"""
3. naive inverse solver using back-projection
"""
eit = bp.BP(ms, elPos, exMtx, step=1, parser='std', weight='none')
ds = eit.solve(f1.v, f0.v, normalize=True)
ds = 10000. * pdeprtni(no2xy, el2no, ds)

# draw
ax2 = fig.add_subplot(gs[0, 1])
im = ax2.tripcolor(no2xy[:, 0], no2xy[:, 1], el2no, ds)
ax2.set_title(r'$\Delta$ Conductivities')
ax2.axis('equal')
fig.colorbar(im)
""" for production figures, use dpi=300 or render pdf """
fig.set_size_inches(6, 3)
# fig.savefig('demo_bp.png', dpi=96)
plt.show()
