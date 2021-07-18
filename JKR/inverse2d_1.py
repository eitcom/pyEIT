# coding: utf-8
""" forward 2D """

# JKR July 2021. Based on pyeit/examples/eit_static_jac.py



# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import division, absolute_import, print_function

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import imageio


from mpl_toolkits import mplot3d

# add path to find pyeit if run directly
import sys
sys.path.append('../')  

import pyeit.mesh as mesh
from pyeit.eit.interp2d import tri_area, sim2pts
from pyeit.mesh import quality
from pyeit.eit.fem import Forward
from pyeit.eit.utils import eit_scan_lines
import pyeit.eit.jac as jac


meshwidth = 200e-6
meshsize = meshwidth/25
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



def overlay_grid_plot(ax,solidmesh=True):
    if(solidmesh):
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
    else:
        plt.triplot(x,y,tri,'-',color='0.9',alpha=0.2,)
    # draw electrodes
    ax.plot(x[el_pos], y[el_pos], "ko")
    for i in range(n_el):
        e = el_pos[i]
        ax1.text(x[e], y[e]-5e-6, str(i), size=8, horizontalalignment='center', verticalalignment='top')
    ax.set_title("equi-potential lines")
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





myframes=[]

# add object
bead_diameter = 50e-6
bead_height = 50e-6

for bead_height in np.arange(30e-6,110e-6,10e-6):
    print('bead diameter',bead_diameter)
    print('bead height',bead_height)
    anomaly = [{"x": 10e-6, "y": bead_height, "d": bead_diameter/2, "perm": 0.25}]
    mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly, background=1.0)
    perm = mesh_new["perm"]
    
    
    ###################
    # INVERSE SOLVER
    
    print('setting up inverse problem')
    
    
    # define the measurement matrix    
    ex_mat = np.array( [ [x,x+1] for x in range(0,10) ] )
    ex_mat = np.append( ex_mat, np.array( [ [x,x+2] for x in range(0,9) ] ), axis=0 )
    ex_mat = np.append( ex_mat, np.array( [ [x,x+3] for x in range(0,8) ] ), axis=0 )
    ex_mat = np.append( ex_mat, np.array( [ [x,x+4] for x in range(0,7) ] ), axis=0 )
    ex_mat = np.append( ex_mat, np.array( [ [x,x+5] for x in range(0,6) ] ), axis=0 )
    
    """ 2. calculate simulated data """    
    fwd = Forward(mesh_obj, el_pos)
    f1 = fwd.solve_eit(ex_mat, perm=mesh_new["perm"], parser="std")
    #print('simulated measurements',f1)
    
    
    """ 3. solve_eit using gaussian-newton (with regularization) """
    # number of stimulation lines/patterns
    eit = jac.JAC(mesh_obj, el_pos, ex_mat, perm=1.0, parser="std")
    eit.setup(p=0.25, lamb=1.0, method="lm")
    # lamb = lamb * lamb_decay
    ds = eit.gn(f1.v, lamb_decay=0.1, lamb_min=1e-5, maxiter=20, verbose=True)
    
    
    # plot results
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    overlay_grid_plot(ax1)  
    ax1.set_title('original')
    
    ax2 = fig.add_subplot(122)
    im = ax2.tripcolor(
        x,
        y,
        tri,
        np.real(ds),
        cmap=plt.cm.Reds,
        antialiased=True,
        #vmin=np.min(ds),
        #vmax=np.max(ds)
    )
    #fig.colorbar(im,ax=ax2,orientation='horizontal')
    overlay_grid_plot(ax2,solidmesh=False)  
    ax2.set_title('reconstruction')
    
    fig.set_size_inches(8, 8)
    
    
    
    
    # add to frames for animation
    fig.canvas.draw()       # draw the canvas, cache the renderer
    im = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    im  = im.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    myframes.append(im)
    
    
    
    
    
    plt.show()


if 0:
    # create .mp4 video file
    imageio.mimsave('EIT_test_1a.mp4', 
                    myframes, 
                    fps=5)

if 0:
    # create animated .gif
    imageio.mimsave('EIT_test_1a.gif', 
                    myframes, 
                    fps=5)