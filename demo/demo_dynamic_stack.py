# demo using stacked exMtx (the devil is in the details)
# liubenyuan@gmail.com
# 2015-07-23

import numpy as np
import matplotlib.pyplot as plt

# pyEIT 2D algorithm modules
import pyeit.mesh as mesh
from pyeit.eit.fem import Forward
from pyeit.eit.utils import eit_scan_lines
import pyeit.eit.jac as jac

""" 1. setup """
ms, el_pos = mesh.create(16)

# test function for altering the 'alpha' in mesh dictionary
anomaly = [{'x': 0.4, 'y': 0.4, 'd': 0.2, 'alpha': 100}]
ms1 = mesh.set_alpha(ms, anomaly=anomaly, background=1.)

# extract node, element, alpha
no2xy = ms['node']
el2no = ms['element']
alpha = ms1['alpha'] - ms['alpha']

# show alpha
fig, ax = plt.subplots(figsize=(6, 4))
im = ax.tripcolor(no2xy[:, 0], no2xy[:, 1], el2no,
                  np.real(alpha), shading='flat')
fig.colorbar(im)
ax.axis('tight')
ax.set_title(r'$\Delta$ Permitivity')

""" 2. calculate simulated data using stack ex_mat """
el_dist, step = 7, 1
n_el = len(el_pos)
ex_mat1 = eit_scan_lines(n_el, el_dist)
ex_mat2 = eit_scan_lines(n_el, 1)
ex_mat = np.vstack([ex_mat1, ex_mat2])

# forward solver
fwd = Forward(ms, el_pos)
f0 = fwd.solve(ex_mat, step, perm=ms['alpha'])
f1 = fwd.solve(ex_mat, step, perm=ms1['alpha'])

""" 3. solving using dynamic EIT """
# number of excitation lines & excitation patterns
eit = jac.JAC(ms, el_pos, ex_mat=ex_mat, step=step, parser='std')
eit.setup(p=0.40, lamb=1e-3, method='kotre')
ds = eit.solve(f1.v, f0.v)

""" 4. plot """
fig, ax = plt.subplots(figsize=(6, 4))
im = ax.tripcolor(no2xy[:, 0], no2xy[:, 1], el2no, np.real(ds),
                  shading='flat', alpha=0.90, cmap=plt.cm.viridis)
fig.colorbar(im)
ax.axis('tight')
ax.set_title(r'$\Delta$ Permitivity Reconstructed')
# plt.savefig('quasi-demo-eit.pdf')
