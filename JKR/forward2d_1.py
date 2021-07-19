# coding: utf-8
""" forward 2D """

# JKR July 2021. Based on pyeit/examples/fem_forward2d.py



# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import division, absolute_import, print_function

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# add path to find pyeit if run directly
import sys
sys.path.append('../')  

import pyeit.mesh as mesh
from pyeit.mesh import quality
from pyeit.eit.fem import Forward
from pyeit.eit.utils import eit_scan_lines


meshwidth = 200e-6
meshheight = 200e-6
meshsize = meshwidth/50
n_el = 11
elec_spacing = 10e-6


""" 0. build mesh """
def myrectangle(pts):
    return mesh.shape.rectangle(pts,p1=[-meshwidth/2,0],p2=[meshwidth/2,meshheight])
p_fix = np.array([[x,0] for x in np.arange(-(n_el//2*elec_spacing),(n_el//2+1)*elec_spacing,elec_spacing)])  # electrodes
p_fix = np.append(p_fix, np.array([[x,meshwidth] for x in np.arange(-meshwidth/2,meshwidth/2,meshsize)]), axis=0)   # dirichlet nodes (const voltage)
mesh_obj, el_pos = mesh.create(len(p_fix), 
                               fd=myrectangle, 
                               p_fix=p_fix, 
                               h0=meshsize,
                               bbox = np.array([[-meshwidth/2, 0], [meshwidth/2, meshheight]]),
                               )
                               #subdivideregions = [ np.array([[-50e-6, 0], [50e-6, 100e-6]]),
                               #                     np.array([[-50e-6, 0], [50e-6, 100e-6]]) ])

# rectangular grid when needed
x_rgrid,y_rgrid = np.meshgrid(np.linspace(-meshwidth/2,meshwidth/2,400),np.linspace(0,meshheight,400))

# constant voltage boundary conditions
# applied to all electrodes after n_el
vbias = 0
dirichlet = [ [el_pos[x],vbias] for x in range(n_el,len(p_fix)) ]
#dirichlet = []

    
# extract node, element, alpha
pts = mesh_obj["node"]
tri = mesh_obj["element"]
x, y = pts[:, 0], pts[:, 1]
quality.stats(pts, tri)



singlefig = plt.figure()
singlefig_ax1 = singlefig.add_subplot(111)
singlefig_cms = [plt.cm.Blues_r, plt.cm.Reds_r, plt.cm.Greens_r]

# change permittivity
for j,bead_diameter in enumerate([0,15e-6,30e-6]):
    print('bead diameter',bead_diameter)
    anomaly = [{"x": 10e-6, "y": 25e-6, "d": bead_diameter/2, "perm": 0.25}]
    mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly, background=1.0)
    perm = mesh_new["perm"]
    
    """ 1. FEM forward simulations """
    # setup EIT scan conditions
    #ex_dist, step = 1, 3
    #ex_mat = eit_scan_lines(16, ex_dist)
    ex_mat = np.array( [ [4,9],
                         [1,6],
                         [2,6],
                         [3,6],
                         [4,6],
                         [5,6],
                         [7,6],
                         [8,6],
                         [9,6],
                         [10,6],
                         [11,6]
                         ] )
    ex_line = ex_mat[0].ravel()
    
    # calculate simulated data using FEM
    fwd = Forward(mesh_obj, el_pos)
    f, _ = fwd.solve(ex_line, perm=perm)
    f = np.real(f)
    
    print('solved potential min=%4.4f  max=%4.4f  ' % (np.min(f),np.max(f)))
    
    
    # calculate the gradient to plot electric field lines
    from matplotlib.tri import (
        Triangulation, CubicTriInterpolator)
    triang = Triangulation(x, y, triangles=tri)
    tci = CubicTriInterpolator(triang, f)
    # Gradient requested here at the mesh nodes but could be anywhere else:
    #(Ex, Ey) = tci.gradient(triang.x, triang.y)

    # get gradient on a rectangular mesh for plotting
    (Ex, Ey) = tci.gradient(x_rgrid,y_rgrid)
    #E_norm = np.sqrt(Ex**2 + Ey**2)
    
    
    
    """ 2. plot """
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    # draw equi-potential lines
    vf = np.linspace(min(f), max(f), 32)   # list of contour voltages
    ax1.tricontour(x, y, tri, f, vf, cmap=plt.cm.inferno)
    # draw mesh structure
    ax1.tripcolor(
        x,
        y,
        tri,
        np.real(perm),
        edgecolors="k",
        shading="flat",
        alpha=0.2,
        cmap=plt.cm.Greys,
    )
    # draw electrodes
    ax1.plot(x[el_pos], y[el_pos], "ro")
    for i in range(n_el):
        e = el_pos[i]
        ax1.text(x[e], y[e]-1e-6, str(i + 1), size=12, horizontalalignment='center', verticalalignment='top')
    ax1.set_title("equi-potential lines")
    # clean up
    ax1.set_aspect("equal")
    ax1.set_ylim([-0.05*meshwidth, 0.55*meshwidth])
    ax1.set_xlim([-0.55*meshwidth, 0.55*meshwidth])
    scale_x,scale_y = 1e-6,1e-6
    ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale_x))
    ax1.xaxis.set_major_formatter(ticks_x)        
    ticks_y = ticker.FuncFormatter(lambda y, pos: '{0:g}'.format(y/scale_y))
    ax1.yaxis.set_major_formatter(ticks_y)      
    ax1.set_xlabel('microns')
    ax1.set_ylabel('microns (height)')
    
    
    ax2 = fig.add_subplot(122)
    # draw electric field vectors
    #E_norm_list = np.linspace(min(E_norm), max(E_norm), 32)   # list of contour voltages
    #ax2.tricontour(x, y, tri, E_norm, E_norm_list, cmap=plt.cm.Reds_r)
    color = 2 * np.log(np.hypot(Ex, Ey))
    ax2.streamplot(x_rgrid,y_rgrid, Ex, Ey, color=color, linewidth=1, cmap=plt.cm.inferno,
              density=1, arrowstyle='->', arrowsize=1.5)
    # draw mesh structure
    ax2.tripcolor(
        x,
        y,
        tri,
        np.real(perm),
        edgecolors="k",
        shading="flat",
        alpha=0.2,
        cmap=plt.cm.Greys,
    )
    # draw electrodes
    ax2.plot(x[el_pos], y[el_pos], "ro")
    for i in range(n_el):
        e = el_pos[i]
        ax2.text(x[e], y[e]-1e-6, str(i + 1), size=12, horizontalalignment='center', verticalalignment='top')
    ax2.set_title("estimated electric field lines")
    # clean up
    ax2.set_aspect("equal")
    ax2.set_ylim([-0.05*meshwidth, 0.55*meshwidth])
    ax2.set_xlim([-0.55*meshwidth, 0.55*meshwidth])
    scale_x,scale_y = 1e-6,1e-6
    ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale_x))
    ax2.xaxis.set_major_formatter(ticks_x)        
    ticks_y = ticker.FuncFormatter(lambda y, pos: '{0:g}'.format(y/scale_y))
    ax2.yaxis.set_major_formatter(ticks_y)      
    ax2.set_xlabel('microns')
    ax2.set_ylabel('microns (height)')
    
    fig.set_size_inches(12, 12)
    
    # fig.savefig('demo_bp.png', dpi=96)
    #plt.show()




    singlefig_ax1.tricontour(x, y, tri, f, vf, cmap=singlefig_cms[j])
    #singlefig_ax1.tricontour(x, y, tri, E_norm, E_norm_list, cmap=plt.cm.Reds_r)


