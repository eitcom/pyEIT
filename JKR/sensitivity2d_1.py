# coding: utf-8
""" forward 2D """

# JKR July 2021. Based on pyeit/examples/eit_sensitivity2d.py



# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import division, absolute_import, print_function

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from mpl_toolkits import mplot3d

# add path to find pyeit if run directly
import sys
sys.path.append('../')  

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
p_fix = np.array([[x,0] for x in np.arange(-(n_el//2*elec_spacing),(n_el//2+1)*elec_spacing,elec_spacing)])  # electrodes
p_fix = np.append(p_fix, np.array([[x,meshwidth] for x in np.arange(-meshwidth/2,meshwidth/2,meshsize)]), axis=0)   # dirichlet nodes (const voltage)
mesh_obj, el_pos = mesh.create(len(p_fix), 
                               fd=myrectangle, 
                               p_fix=p_fix, 
                               h0=meshsize,
                               bbox = np.array([[-meshwidth/2, 0], [meshwidth/2, meshwidth]]))

# rectangular grid when needed
x_rgrid,y_rgrid = np.meshgrid(np.linspace(-meshwidth/2,meshwidth/2,400),np.linspace(0,meshwidth,400))

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



def calc_sens(fwd, ex_mat):
    """
    see Adler2017 on IEEE TBME, pp 5, figure 6,
    Electrical Impedance Tomography: Tissue Properties to Image Measures
    """
    # solving EIT problem
    p = fwd.solve_eit(ex_mat=ex_mat, parser="fmmu", dirichlet=dirichlet)
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
    sn = sim2pts(pts, tri, se)   # log scale
    #sn = sim2pts(pts, tri, s)   # linear scale
    return sn





#for elec_sep in [-5,-4,-3,-2,-1,1,2,3,4,5]:
for elec_sep in [3,]:

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
    f, _ = fwd.solve(ex_line,perm=perm, dirichlet=dirichlet)
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
    E_norm = np.sqrt(Ex**2 + Ey**2)
    
    
    
    
    # calculate sensitivity 
    sensitivity = calc_sens(fwd, ex_mat)
    
    
    
    def overlay_grid_plot(ax):
        # draw mesh structure
        ax.tripcolor(
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
        ax.plot(x[el_pos], y[el_pos], "ko")
        for i in range(n_el):
            e = el_pos[i]
            ax1.text(x[e], y[e]-5e-6, str(i), size=8, horizontalalignment='center', verticalalignment='top')
        # clean up
        ax.set_aspect("equal")
        ax.set_ylim([-0.1*meshwidth, 1.05*meshwidth])
        ax.set_xlim([-0.55*meshwidth, 0.55*meshwidth])
        scale_x,scale_y = 1e-6,1e-6
        ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale_x))
        ax.xaxis.set_major_formatter(ticks_x)        
        ticks_y = ticker.FuncFormatter(lambda y, pos: '{0:g}'.format(y/scale_y))
        ax.yaxis.set_major_formatter(ticks_y)      
        ax.set_xlabel('microns')
        ax.set_ylabel('microns (height)')
        
    
    
    """ 2. plot """
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    # draw equi-potential lines
    vf = np.linspace(min(f), max(f), 32)   # list of contour voltages
    ax1.tricontour(x, y, tri, f, vf, cmap=plt.cm.inferno)
    overlay_grid_plot(ax1)
    ax1.set_title("equi-potential lines")
    
    
    ax2 = fig.add_subplot(222)
    # draw electric field vectors
    #E_norm_list = np.linspace(min(E_norm), max(E_norm), 32)   # list of contour voltages
    #ax2.tricontour(x, y, tri, E_norm, E_norm_list, cmap=plt.cm.Reds_r)
    color = 2 * np.log(np.hypot(Ex, Ey))
    ax2.streamplot(x_rgrid,y_rgrid, Ex, Ey, color=color, linewidth=1, cmap=plt.cm.inferno,
              density=1, arrowstyle='->', arrowsize=1.5)
    overlay_grid_plot(ax2)
    ax2.set_title("electric field")
    
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
        cmap=plt.cm.Reds,
        antialiased=True,
        vmin=np.min(sensitivity),
        vmax=np.max(sensitivity)
    )
    fig.colorbar(im,ax=ax3,orientation='horizontal')
    overlay_grid_plot(ax3)    
    ax3.set_title("log(sensitivity)")
    
    
#    ax4 = fig.add_subplot(224,projection='3d')    
##    ax = plt.axes(projection='3d')
#    ax4.scatter3D(x, y, sensitivity, c=sensitivity, cmap=plt.cm.Reds);
#    ax4.set_title('surface of sensitivity');
#

    # calculate Efield on rectangular mesh points
    (Ex, Ey) = tci.gradient(x_rgrid,y_rgrid)
    E_norm = np.sqrt(Ex**2 + Ey**2)
    
    ax4 = fig.add_subplot(224)
    
#    im = ax4.tripcolor(
#        x,
#        y,
#        tri,
#        E_norm,
#        edgecolors="none",
#        shading="gouraud",
#        cmap=plt.cm.Reds,
#        antialiased=True,
#        vmin=np.min(E_norm),
#        vmax=np.max(E_norm)
#    )
#    fig.colorbar(im,ax=ax4,orientation='horizontal')
#    overlay_grid_plot(ax4)    

    E_norm = np.array(E_norm)
    E_norm = E_norm / np.mean(E_norm,axis=1)[0]
    ax4.semilogy(np.linspace(0,meshwidth,400), np.mean(E_norm,axis=1))
    ax4.set_xlabel('distance from the sensor (microns)')
    ax4.set_ylabel('mean electric field strength')
    scale_x,scale_y = 1e-6,1e-6
    ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale_x))
    ax4.xaxis.set_major_formatter(ticks_x)        
    
    fig.set_size_inches(12, 12)
    
    
    
    plt.show()



    # plotting electric field strength
    (Ex, Ey) = tci.gradient(triang.x, triang.y)
    E_norm = np.sqrt(Ex**2 + Ey**2)
    fig=plt.figure()
    ax = fig.add_subplot(221,projection='3d')
    ax.scatter3D(x, y, E_norm, c=E_norm, cmap=plt.cm.bwr);
    ax.set_title('surface of E_norm');
    ax = fig.add_subplot(222)
    ax.plot(E_norm)
    ax.set_title('E_norm of all nodes')
    ax = fig.add_subplot(223)
    ax.semilogy(np.hypot(x,y),E_norm,'.')
    ax.set_title('distance vs E_norm')    
    fig.set_size_inches(8, 8)
    plt.show()
    

#singlefig_ax1.tricontour(x, y, tri, f, vf, cmap=singlefig_cms[j])
#singlefig_ax1.tricontour(x, y, tri, E_norm, E_norm_list, cmap=plt.cm.Reds_r)


