""" demo on static solving using JAC (experimental) """

import numpy as np
import matplotlib.pyplot as plt

# pyEIT 2D algorithms modules
from pyeit.mesh import create, set_alpha
from pyeit.eit.fem import Forward
from pyeit.eit.utils import eit_scan_lines
import pyeit.eit.jac as jac

""" 1. setup """
n_el = 16
ms, el_pos = create(n_el, h0=0.1)

# test function for altering the 'alpha' in mesh dictionary
anomaly = [{'x': 0.4, 'y': 0.4, 'd': 0.2, 'alpha': 10},
           {'x': -0.4, 'y': -0.4, 'd': 0.2, 'alpha': 0.1}]
# TODO: even if background changed to values other than 1.0 will fail
ms1 = set_alpha(ms, anomaly=anomaly, background=1.)

# extract node, element, alpha
no2xy = ms['node']
el2no = ms['element']
alpha = ms1['alpha']

# show
fig, ax = plt.subplots(figsize=(6, 4))
im = ax.tripcolor(no2xy[:, 0], no2xy[:, 1], el2no,
                  np.real(alpha), shading='flat', cmap=plt.cm.viridis)
fig.colorbar(im)
ax.axis('equal')
ax.set_title(r'$\Delta$ Conductivities')
plt.show()

""" 2. calculate simulated data """
el_dist, step = 1, 1
ex_mat = eit_scan_lines(n_el, el_dist)
fwd = Forward(ms, el_pos)
f1 = fwd.solve(ex_mat, step, perm=ms1['alpha'], parser='std')

""" 3. solve using gaussian-newton """
# number of excitation lines & excitation patterns
eit = jac.JAC(ms, el_pos, ex_mat, step, perm=1.0, parser='std')
eit.setup(p=0.25, lamb=0.1, method='kotre')
ds = eit.gn(f1.v, lamb_decay=0.1, lamb_min=1e-4, maxiter=8, verbose=True)

# plot
fig, ax = plt.subplots(figsize=(6, 4))
im = ax.tripcolor(no2xy[:, 0], no2xy[:, 1], el2no, np.real(ds),
                  shading='flat', alpha=1.0, cmap=plt.cm.viridis)
fig.colorbar(im)
ax.axis('equal')
ax.set_title('Conductivities Reconstructed')
# fig.savefig('../figs/demo_static.png', dpi=96)
plt.show()
