""" reproducible code for EIT2016 """
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# pyEIT 2D algo modules
import pyeit.mesh as mesh
from pyeit.eit.fem import forward, pdeprtni
from pyeit.eit.utils import eit_scan_lines
import pyeit.eit.greit as greit
import pyeit.eit.bp as bp
import pyeit.eit.jac as jac

""" 0. construct mesh structure """
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

""" ax1. FEM forward simulations """
# setup EIT scan conditions
elDist, step = 1, 1
exMtx = eit_scan_lines(16, elDist)

# calculate simulated data
fwd = forward(ms, elPos)
f0 = fwd.solve(exMtx, step=step, perm=ms0['alpha'])
f1 = fwd.solve(exMtx, step=step, perm=ms1['alpha'])

""" ax2. BP """
eit = bp.BP(ms, elPos, exMtx, step=1, parser='std')
ds = eit.solve(f1.v, f0.v, normalize=True)
ds_bp = pdeprtni(no2xy, el2no, ds)

""" ax3. JAC """
eit = jac.JAC(ms, elPos, exMtx=exMtx, step=step,
              perm=1., parser='std',
              p=0.2, lamb=0.001, method='kotre')
# parameter tuning is needed for better display
ds = eit.solve(f1.v, f0.v)
ds_jac = pdeprtni(no2xy, el2no, ds)

""" ax4. GREIT """
eit = greit.GREIT(ms, elPos, exMtx=exMtx, step=step, parser='std')
ds = eit.solve(f1.v, f0.v)
x, y, ds_greit = eit.mask_value(ds, mask_value=np.NAN)

""" build for EIT2016b (orig: 300p x 300p, 150dpi) """
size = (6, 6)
axis_size = [-1.2, 1.2, -1.2, 1.2]
fig = plt.figure(figsize=size)
gs = gridspec.GridSpec(2, 2)

# simulation
ax1 = fig.add_subplot(gs[0, 0])
ax1.tripcolor(no2xy[:, 0], no2xy[:, 1], el2no, alpha, shading='flat')
ax1.set_title(r'(a) $\Delta$ Conductivity')
ax1.axis(axis_size)
ax1.axis('off')

# Filtered BP
ax2 = fig.add_subplot(gs[0, 1])
ax2.tripcolor(no2xy[:, 0], no2xy[:, 1], el2no, np.real(ds_bp))
ax2.set_title(r'(b) BP')
ax1.axis(axis_size)
ax2.axis('off')

# JAC
ax3 = fig.add_subplot(gs[1, 0])
ax3.tripcolor(no2xy[:, 0], no2xy[:, 1], el2no, np.real(ds_jac))
ax3.set_title(r'(c) JAC')
ax1.axis(axis_size)
ax3.axis('off')

# GREIT
ax4 = fig.add_subplot(gs[1, 1])
ax4.imshow(np.real(ds_greit), interpolation='nearest')
ax4.set_title(r'(d) GREIT')
ax1.axis(axis_size)
ax4.axis('off')

# save
# fig.tight_layout()
# import matplotlib
# matplotlib.rcParams.update({'font.size': 3})
# fig.subplots_adjust(top=0.875, bottom=0.01)
# fig.set_size_inches(1, 1)
# fig.savefig('eit2016b.png', dpi=300)
