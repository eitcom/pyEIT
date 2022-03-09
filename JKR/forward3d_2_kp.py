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
import matplotlib.pyplot as plt


meshwidth = 200e-6
meshheight = 100e-6
meshsize = meshwidth/5

n_el=11
elec_spacing=10e-6
epsilon = 8.85*1e-12

background = 80 * epsilon


#p_fix = np.array([[x,0] for x in np.arange(-(n_el//2*elec_spacing),(n_el//2+1)*elec_spacing,elec_spacing)])  # electrodes
dx = np.arange(-(n_el//2*elec_spacing),(n_el//2+1)*elec_spacing,elec_spacing)
p_fix = list(itertools.product(dx,dx,[0]))
print('locations of electrodes',p_fix)

# build tetrahedron
# 3D tetrahedron must have a bbox
bbox = np.array([[-meshwidth/2, -meshwidth/2, 0], [meshwidth/2, meshwidth/2, meshheight]])

# save calling convention as distmesh 2D
mesh_obj, el_pos = mesh.create(n_el=len(p_fix), h0=meshsize, bbox=bbox, p_fix=p_fix, fd=ball)

pts = mesh_obj["node"]
tri = mesh_obj["element"]

# report the status of the 2D mesh
quality.stats(pts, tri)

""" 1. FEM forward simulations """
# setup EIT scan conditions
#el_dist, step = 7, 1
#ex_mat = eit_scan_lines(16, el_dist)

# array of electrode pairs

#ex_mat = list(itertools.product(range(n_el),[int((n_el+1)/2),int((n_el+5)/2),int((n_el-3)/2)]))
#ex_mat = np.array([e for e in ex_mat if e[0]!=e[1]])  # remove doubles

# define the measurement matrix    
p_fix_microns = np.int64(np.float32(p_fix)*1e6)
p_fix_line_indices = np.array([x for x in range(len(p_fix_microns)) if p_fix_microns[x][0]==0])

ex_mat = np.array( [ [x,x+1] for x in range(0,n_el-1) ] )
ex_mat = np.append( ex_mat, np.array( [ [x,x+2] for x in range(0,n_el-3) ] ), axis=0 )
ex_mat = np.append( ex_mat, np.array( [ [x,x+3] for x in range(0,n_el-4) ] ), axis=0 )
ex_mat = np.append( ex_mat, np.array( [ [x,x+4] for x in range(0,n_el-5) ] ), axis=0 )
ex_mat = np.append( ex_mat, np.array( [ [x,x+5] for x in range(0,n_el-6) ] ), axis=0 )
ex_mat = np.array( [ [p_fix_line_indices[x1],p_fix_line_indices[x2]] for (x1,x2) in ex_mat ] )

# calculate simulated data
fwd = Forward(mesh_obj, el_pos)

# in python, index start from 0
#ex_line = ex_mat[2].ravel()

# change alpha
anomaly = [{"x": 0, "y": 0, "z": 10e-6, "d": 20e-6, "perm": 2.5*epsilon}]
mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly, background=background)
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

fig, axs = plt.subplots(5, figsize=(5,10))
axs[0].plot(meas_mat[0:9, 2])
axs[0].set_title('1-pixel spacing')

axs[1].plot(meas_mat[9:17, 2])
axs[1].set_title('2-pixel spacing')

axs[2].plot(meas_mat[17:24, 2])
axs[2].set_title('3-pixel spacing')

axs[3].plot(meas_mat[24:30, 2])
axs[3].set_title('4-pixel spacing')

axs[4].plot(meas_mat[30:35, 2])
axs[4].set_title('5-pixel spacing')

plt.tight_layout()




