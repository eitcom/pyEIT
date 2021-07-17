# coding: utf-8
""" forward 2D """

# JKR July 2021. Based on pyeit/examples/eit_sensitivity2d.py



# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import division, absolute_import, print_function

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d

import pyeit.mesh as mesh
from pyeit.eit.interp2d import tri_area, sim2pts
from pyeit.mesh import quality
from pyeit.eit.fem import Forward
from pyeit.eit.utils import eit_scan_lines



meshwidth = 200e-6
meshsize = meshwidth/50
n_el = 11
elec_spacing = 10e-6


""" 0. build mesh """
def myrectangle(pts):
    return mesh.shape.rectangle(pts,p1=[-meshwidth/2,0],p2=[meshwidth/2,meshwidth])
p_fix = np.array([[x,0] for x in np.arange(-(n_el//2*elec_spacing),(n_el//2+1)*elec_spacing,elec_spacing)])
mesh_obj, el_pos = mesh.create(n_el, 
                               fd=myrectangle, 
                               p_fix=p_fix, 
                               h0=meshsize,
                               bbox = np.array([[-meshwidth/2, 0], [meshwidth/2, meshwidth]]))

# rectangular grid when needed
x_rgrid,y_rgrid = np.meshgrid(np.linspace(-meshwidth/2,meshwidth/2,100),np.linspace(0,meshwidth,100))
    
# extract node, element, alpha
pts = mesh_obj["node"]
tri = mesh_obj["element"]
x, y = pts[:, 0], pts[:, 1]
quality.stats(pts, tri)



def calc_sens(fwd, ex_mat):
    """
    see Adler2017 on IEEE TBME, pp 5, figure 6,
    Electrical Impedance Tomography: Tissue Properties to Image Measures
    """
    # solving EIT problem
    p = fwd.solve_eit(ex_mat=ex_mat, parser="fmmu")
    v0 = p.v
    # normalized jacobian (note: normalize affect sensitivity)
    v0 = v0[:, np.newaxis]
    jac = p.jac / v0
    # calculate sensitivity matrix
    s = np.linalg.norm(jac, axis=0)
    ae = tri_area(pts, tri)
    s = np.sqrt(s) / ae
    assert any(s >= 0)

    se = np.log10(s)
    sn = sim2pts(pts, tri, se)
    return sn





for elec_sep in [-5,-4,-3,-2,-1,1,2,3,4,5]:

    print('elec_sep',elec_sep)
    
    """ 1. FEM forward simulations """
    # setup EIT scan conditions
    #ex_dist, step = 1, 3
    #ex_mat = eit_scan_lines(16, ex_dist)
    #ex_mat = np.array( [ [0,5],
    #                     [1,5],                     
    #                     [2,5],
    #                     [3,5],
    #                     [4,5],
    #                     [5,5],
    #                     [7,5],
    #                     [8,5],
    #                     [9,5],
    #                     [10,5]
    #                     ] )
    ex_mat = np.array( [ [5+elec_sep,5], ] )
    ex_line = ex_mat[0].ravel()
    
    perm = mesh_obj["perm"]
    
    # calculate simulated data using FEM
    fwd = Forward(mesh_obj, el_pos)
    f, _ = fwd.solve(ex_line,perm=perm)
    f = np.real(f)
    
    print('solved potential min=%4.4f  max=%4.4f  ' % (np.min(f),np.max(f)))
    
    
    # calculate the gradient to plot electric field lines
    from matplotlib.tri import (
        Triangulation, CubicTriInterpolator)
    triang = Triangulation(x, y, triangles=tri)
    tci = CubicTriInterpolator(triang, -f)
    # Gradient requested here at the mesh nodes but could be anywhere else:
    #(Ex, Ey) = tci.gradient(triang.x, triang.y)
    
    # get gradient on a rectangular mesh for plotting
    (Ex, Ey) = tci.gradient(x_rgrid,y_rgrid)
    #E_norm = np.sqrt(Ex**2 + Ey**2)
    
    
    
    
    # calculate sensitivity 
    sensitivity = calc_sens(fwd, ex_mat)
    
    
    
    
    
    
    """ 2. plot """
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
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
    for i, e in enumerate(el_pos):
        ax1.text(x[e], y[e]-1e-6, str(i + 1), size=12, horizontalalignment='center', verticalalignment='top')
    ax1.set_title("equi-potential lines")
    # clean up
    ax1.set_aspect("equal")
    ax1.set_ylim([-0.05*meshwidth, 1.05*meshwidth])
    ax1.set_xlim([-0.55*meshwidth, 0.55*meshwidth])
    
    
    ax2 = fig.add_subplot(222)
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
    for i, e in enumerate(el_pos):
        ax2.text(x[e], y[e]-1e-6, str(i + 1), size=12, horizontalalignment='center', verticalalignment='top')
    ax2.set_title("estimated electric field lines")
    # clean up
    ax2.set_aspect("equal")
    ax2.set_ylim([-0.05*meshwidth, 1.05*meshwidth])
    ax2.set_xlim([-0.55*meshwidth, 0.55*meshwidth])
    
    
    # fig.savefig('demo_bp.png', dpi=96)
    #plt.show()
    
    
    
    ax3 = fig.add_subplot(223)
    
    im = ax3.tripcolor(
        x,
        y,
        tri,
        sensitivity,
        edgecolors="none",
        shading="gouraud",
        cmap=plt.cm.inferno,
        antialiased=True,
        vmin=np.min(sensitivity),
        vmax=np.max(sensitivity)
    )
    # draw electrodes
    ax3.plot(x[el_pos], y[el_pos], "ro")
    for i, e in enumerate(el_pos):
        ax3.text(x[e], y[e]-1e-6, str(i + 1), size=12, horizontalalignment='center', verticalalignment='top')
    ax3.set_title("sensitivity")
    # clean up
    ax3.set_aspect("equal")
    ax3.set_ylim([-0.05*meshwidth, 1.05*meshwidth])
    ax3.set_xlim([-0.55*meshwidth, 0.55*meshwidth])
    fig.colorbar(im,ax=ax3,orientation='horizontal')
    
    
    
    ax4 = fig.add_subplot(224,projection='3d')    
#    ax = plt.axes(projection='3d')
    ax4.scatter3D(x, y, sensitivity, c=sensitivity, cmap=plt.cm.inferno);
    ax4.set_title('surface of sensitivity');



    
    fig.set_size_inches(12, 12)
    
    
    
    plt.show()




#singlefig_ax1.tricontour(x, y, tri, f, vf, cmap=singlefig_cms[j])
#singlefig_ax1.tricontour(x, y, tri, E_norm, E_norm_list, cmap=plt.cm.Reds_r)


