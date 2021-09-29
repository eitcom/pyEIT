# coding: utf-8
""" forward 2D """

# Sept 2021
# simulates measured impedance for various permittivity slices



# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import division, absolute_import, print_function

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.tri import Triangulation, CubicTriInterpolator, LinearTriInterpolator

import itertools
import h5py

# add path to find pyeit if run directly
import sys
sys.path.append('../')  

import pyeit.mesh as mesh
from pyeit.mesh import quality
from pyeit.eit.fem import Forward
from pyeit.eit.utils import eit_scan_lines

# calculate the distance between bead centers
def cal_bead_cent_dist(x1,y1,x2,y2):
    return ((((x2 - x1 )**2) + ((y2-y1)**2) )**0.5)


meshwidth = 250e-6
meshheight = 150e-6
meshsize = meshwidth/50
n_el = 15
elec_spacing = 10e-6


""" 0. build mesh """
def myrectangle(pts):
    return mesh.shape.rectangle(pts,p1=[-meshwidth/2,-10e-6],p2=[meshwidth/2,meshheight])
p_fix = np.array([[x,0] for x in np.arange(-(n_el//2*elec_spacing),(n_el//2+1)*elec_spacing,elec_spacing)])  # electrodes
p_fix = np.append(p_fix, np.array([[x,meshwidth] for x in np.arange(-meshwidth/2,meshwidth/2,meshsize)]), axis=0)   # dirichlet nodes (const voltage)
mesh_obj, el_pos = mesh.create(len(p_fix), 
                               fd=myrectangle, 
                               p_fix=p_fix, 
                               h0=meshsize,
                               bbox = np.array([[-meshwidth/2, 0], [meshwidth/2, meshheight]]),
                               )


# rectangular grid when needed
rgrid_size=1e-6
x_rgrid,y_rgrid = np.meshgrid(np.arange(-meshwidth/2,meshwidth/2,rgrid_size),
                              np.arange(0,meshheight,rgrid_size))

# constant voltage boundary conditions
# applied to all electrodes after n_el
vbias = 0
dirichlet = [ [el_pos[x],vbias] for x in range(n_el,len(p_fix)) ]
#dirichlet = []

    
pts = mesh_obj["node"]
tri = mesh_obj["element"]
quality.stats(pts, tri)

tri_centers = np.mean(pts[tri], axis=1)    
x_tri, y_tri = tri_centers[:, 0], tri_centers[:, 1]



num_trials = 10000

for j in range(1790, num_trials, 1):
    mesh_new=mesh_obj.copy()
    perm = mesh_new["perm"]
    background=1.0
    
    # num_beads = np.random.randint(3)
    num_beads = 3

    # list of bead position
    bead_x_list = []
    bead_y_list = []

    for b in range(num_beads):
        # bead_diameter = 30e-6 + 10e-6*np.random.randn()
        # bead_diameter = max(bead_diameter,2e-6)
        # fix bead diameter
        bead_diameter = 40e-6
        # bead_x = -meshwidth/4 + meshwidth/2*np.random.rand()
        # bead_y = meshheight/2*np.random.rand()
        # bead_y = min(bead_y,bead_diameter/2)

        # randomize bead center
        bead_x = np.random.uniform(-60e-6,60e-6)
        bead_y = np.random.uniform(20e-6,100e-6)

        # not the first bead
        if bead_x_list:
            flag = 1

            while flag:
                # check the separation:
                flag = 0
                for i in range(len(bead_x_list)):
                    if cal_bead_cent_dist(bead_x_list[i], bead_y_list[i], bead_x, bead_y) < 45e-6:
                        flag = 1

                # generate a new one
                if flag == 1:
                    bead_x = np.random.uniform(-60e-6,60e-6)
                    bead_y = np.random.uniform(20e-6,80e-6)                
        
        bead_x_list.append(bead_x)
        bead_y_list.append(bead_y)


        # print('bead d=%2.2e, x=%2.2e, y=%2.2e' % (bead_diameter,bead_x,bead_y))
        anomaly = [{"x": bead_x, "y": bead_y, "d": bead_diameter/2, "perm": 0.25}]
        mesh_new = mesh.set_perm(mesh_new, anomaly=anomaly, background=background)
        perm = mesh_new["perm"]

        background=None  # for loop>1 do not reset background
        
    
    # array of electrode pairs
    #ex_mat = list(itertools.product(range(n_el),[int((n_el+1)/2),]))
    #ex_mat = list(itertools.product(range(n_el),range(n_el)))
    ex_mat = list(itertools.product(range(n_el),[int((n_el+1)/2),int((n_el+5)/2),int((n_el-3)/2)]))
    ex_mat = np.array([e for e in ex_mat if e[0]!=e[1]])  # remove doubles

    # calculated simulated capacitances    
    fwd = Forward(mesh_obj, el_pos)
    f1 = fwd.solve_eit(ex_mat, perm=mesh_new["perm"], parser="std")
    meas_mat = np.hstack([ex_mat,f1.v[:,np.newaxis]])
    # print('simulated measurements\n',f1.v)
    # print(len(f1.v))
    
    # create interpolator for permittivity
    # and get permittivity on a rectangular mesh
    triang = Triangulation(x_tri, y_tri)
    tci = LinearTriInterpolator(triang, perm)
    perm_xy = tci(x_rgrid,y_rgrid)

    # plot
    plt.figure(figsize=(6,6))
    plt.imshow(perm_xy,cmap='Greys_r')
    plt.ylim([0,y_rgrid.shape[0]])
    plt.title('interpolated permittivity (rectangular grid)')
    plt.savefig('figure/permittivity_'+str(num_beads)+'_beads_'+str(j)+'.png')
    # plt.show()
    plt.close()

    # save the output file
    X_name = 'train/permittivity_'+str(num_beads)+'_beads_'+str(j)+'.npy'
    Y_name = 'train/capacitance_'+str(num_beads)+'_beads_'+str(j)+'.npy'
    with open(X_name, 'wb') as f:
        np.save(f, np.asarray(perm_xy))

    with open(Y_name, 'wb') as f:
        np.save(f, np.asarray(f1.v))


