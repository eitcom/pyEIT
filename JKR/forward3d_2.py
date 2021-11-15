# coding: utf-8


# JKR November 2021. Based on pyeit/examples/eit_dynamic_jac3d.py


# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import division, absolute_import, print_function

import numpy as np
import itertools

# add path to find pyeit if run directly
import sys
sys.path.append('../')  

import pyeit.mesh as mesh
from pyeit.mesh import quality
import pyeit.mesh.plot as mplot
from pyeit.eit.fem import Forward
from pyeit.eit.interp2d import sim2pts
from pyeit.eit.utils import eit_scan_lines
import pyeit.eit.jac as jac
from pyeit.mesh.shape import ball


meshwidth = 200e-6
meshheight = 100e-6
meshsize = meshwidth/5

n_el=11
elec_spacing=10e-6


#p_fix = np.array([[x,0] for x in np.arange(-(n_el//2*elec_spacing),(n_el//2+1)*elec_spacing,elec_spacing)])  # electrodes
dx = np.arange(-(n_el//2*elec_spacing),(n_el//2+1)*elec_spacing,elec_spacing)
p_fix = list(itertools.product(dx,dx,[0]))
print('locations of electrodes',p_fix)

# build tetrahedron
# 3D tetrahedron must have a bbox
bbox = np.array([[-meshwidth/2, -meshwidth/2, 0], [meshwidth/2, meshwidth/2, meshheight]])

# save calling convention as distmesh 2D
mesh_obj, el_pos = mesh.create(h0=meshsize, bbox=bbox, p_fix=p_fix, fd=ball)

pts = mesh_obj["node"]
tri = mesh_obj["element"]

# report the status of the 2D mesh
quality.stats(pts, tri)

""" 1. FEM forward simulations """
# setup EIT scan conditions
#el_dist, step = 7, 1
#ex_mat = eit_scan_lines(16, el_dist)

# array of electrode pairs

ex_mat = list(itertools.product(range(n_el),[int((n_el+1)/2),int((n_el+5)/2),int((n_el-3)/2)]))
ex_mat = np.array([e for e in ex_mat if e[0]!=e[1]])  # remove doubles




# calculate simulated data
fwd = Forward(mesh_obj, el_pos)

# in python, index start from 0
#ex_line = ex_mat[2].ravel()

# change alpha
anomaly = [{"x": 0.40, "y": 0.40, "z": 0.0, "d": 0.30, "perm": 100.0}]
mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly, background=1.0)
tri_perm = mesh_new["perm"]
node_perm = sim2pts(pts, tri, np.real(tri_perm))

# solving once using fem
# f, _ = fwd.solve(ex_line, tri_perm)
# f = np.real(f)

# calculate simulated data
step=1
f0 = fwd.solve_eit(ex_mat, step=step, perm=mesh_obj["perm"])
f1 = fwd.solve_eit(ex_mat, step=step, perm=mesh_new["perm"])


meas_mat = np.hstack([ex_mat,f1.v[:,np.newaxis]])
print('simulated measurements\n',meas_mat)



