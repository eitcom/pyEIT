""" forward 3D (computation on tetrahedrons) """
from __future__ import division, absolute_import, print_function
import numpy as np

import pyeit.mesh as mesh
import pyeit.mesh.plot as mplot
from pyeit.eit.fem import Forward
from pyeit.eit.fem import EITForward
from pyeit.eit.interp2d import sim2pts
import pyeit.eit.protocol as protocol
from pyeit.mesh.wrapper import PyEITAnomaly_Ball
from pyeit.mesh.shape import ball

# 3D Mesh shape is specified with fd parameter in the instantiation, e.g : fd=ball , Default in 3D :fd=ball
n_el = 16  # number of electrodes
mesh_obj = mesh.create(n_el, h0=0.12, fd=ball)

# report the status of the 2D mesh
pts = mesh_obj.node
tri = mesh_obj.element
mesh_obj.print_stats()

""" 1. FEM forward simulations """
# setup EIT scan conditions
protocol_obj = protocol.create(n_el, dist_exc=2, step_meas=2, parser_meas="std")

# calculate simulated data
fwd = EITForward(mesh_obj, protocol_obj)

# Create permittivity map with anomaly
anomaly = PyEITAnomaly_Ball(center=[0.4, 0.2, 0], r=0.2, perm=10.0)
mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly, background=1.0)
tri_perm = mesh_new.perm
node_perm = sim2pts(pts, tri, np.real(tri_perm))

# calculate simulated surface potential data
v1 = fwd.solve_eit(perm=mesh_new.perm)

# Plot the 3D permittivity (conductivity) map
mplot.tetplot(pts, tri, vertex_color=node_perm, alpha=0.5)
