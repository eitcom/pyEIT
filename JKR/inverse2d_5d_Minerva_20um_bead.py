# coding: utf-8
""" forward 2D """

# JKR July 2021. Based on pyeit/examples/eit_static_jac.py



from __future__ import division, absolute_import, print_function



import h5py
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import glob
from datetime import datetime
from itertools import chain

#os.chdir(r"R:\projects\minerva")

#from run_files import Check_connection
#from run_files import Measure
#from run_files import Decode

import imageio



logdir = r"/Users/jkrosens/Dropbox/HyBiScIs_data/Larkin Lab Data/ECT_Training_Dataset_04172022/ECT_1D_dataset"
#logdir = r"/Users/jkrosens/Dropbox/HyBiScIs_data/Rosenstein Lab Data/F0011_04172022_confocal_ECT_beads/ECT_1D_dataset"
#logdir = r"/Users/jkrosens/Dropbox/HyBiScIs_data/Rosenstein Lab Data/F0002_09062021_20u_bead_ECT"


#lognames = glob.glob(logdir+'/redo_post_confocal_ECT_scan_1D_04172022_set_0.h5')
#lognames = glob.glob(logdir+'/redo_pre_confocal_ECT_scan_1D_04172022_set_2.h5')
#lognames = glob.glob(logdir+'/redo_pre_confocal_ECT_scan_1D_04172022_set_0.h5')
#lognames = glob.glob(logdir+'/20u_bead_wet_20u_bead_VCM_500_VSW_100.h5')
lognames = glob.glob(logdir+'/redo_pre_confocal_ECT_scan_1D_04172022_set_1.h5')


mycolormap='Blues'
#mycolormap='viridis'




minerva_data = []



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def Get_Data(logname, exp_name, dataname='image'):
    hf = h5py.File(logname, 'r')
    grp_data = hf.get(exp_name)
    image = grp_data[dataname][:]
    return image    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   
def Get_Attr(logname, exp_name, attrname):
    hf = h5py.File(logname, 'r')
    grp_data = hf.get(exp_name)
    return grp_data.attrs[attrname]
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   
def Get_Time(logname, exp_name):
    return datetime.strptime(Get_Attr(logname, exp_name, 'timestamp'), "%Y%m%d_%H%M%S")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   
def Get_List(logname,filterstring=None,sortby=None):
    hf = h5py.File(logname, 'r')
    base_items = list(hf.items())
    grp_list = []
    for i in range(len(base_items)):
        grp = base_items[i]
        grp_list.append(grp[0])
    if filterstring is not None:
        grp_list = [x for x in grp_list if filterstring in x]
    if sortby is 'time':
        grp_list = sorted(grp_list,key=lambda x: Get_Time(logname,x))
    return grp_list	
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    


plotslices=[]

for lognum,logname in enumerate(lognames):

    fullname = os.path.join(logdir,logname)
    
    list_all = Get_List(fullname,sortby='time')
    print('\n\nall\n\n',list_all)
    
    list_ect = Get_List(fullname,filterstring='ECT')
    print('\n\nECT\n\n',list_ect)
    
    if(lognum==0):
        t0 = Get_Time(fullname,list_all[0])
    
    
    
    # plot images
    if(lognum==0):
        myframes=[]
        colonysizes=[[],]*4
        startindex=0
        endindex=len(list_all)
        image_1_ref=None

    for i in range(startindex,endindex,1):

        if 'ECT_' in list_all[i]:        

            V_STBY = Get_Attr(fullname,list_all[i],'V_STBY')
            V_CM = Get_Attr(fullname,list_all[i],'V_CM')
            f_sw = Get_Attr(fullname,list_all[i],'f_sw')
            T_int = Get_Attr(fullname,list_all[i],'T_int')
            C_int = Get_Attr(fullname,list_all[i],'C_int')
            
            gain_swcap = np.abs(V_STBY-V_CM)*1e-3*f_sw  # Iout/Cin
            gain_integrator = T_int/C_int  # Vout/Iin
            gain_overall = gain_swcap*gain_integrator

        
            row_offset = Get_Attr(fullname,list_all[i],'row_offset')        
            col_offset = Get_Attr(fullname,list_all[i],'col_offset')                
        
            print('row/col = ',row_offset,col_offset)
            #if(abs(row_offset) > 5):
            #    continue
            #if(abs(col_offset) > 5):
            #    continue
            
            print('row/col = ',row_offset,col_offset)
            
        
        
            image_2d_ph1 = Get_Data(fullname,
                                    list_all[i],
                                    dataname='image_2d_ph1')
            image_2d_ph2 = Get_Data(fullname,
                                    list_all[i],
                                    dataname='image_2d_ph2')
            
            #image_1 = image_1[100:350,100:200]


            # fix column alignment ECT offset
            coloffset=15
            for ch in range(8):
                image_2d_ph1[:, ch*32:(ch+1)*32] = image_2d_ph1[:, ch*32+np.mod(-coloffset + np.arange(32),32)]
                image_2d_ph2[:, ch*32:(ch+1)*32] = image_2d_ph2[:, ch*32+np.mod(-coloffset + np.arange(32),32)]




            image_2d_ph1 = image_2d_ph1 / gain_overall
            image_2d_ph2 = image_2d_ph2 / gain_overall

            normrows = range(100,200)
                
            # normalize by channel
            def normalize_by_channel(image):
                ch0mean = np.mean(image[normrows, :32])
                for ch in range(8):
                    image[:, ch*32:(ch+1)*32] = image[:, ch*32:(ch+1)*32] / np.mean(image[normrows, ch*32:(ch+1)*32]) * ch0mean
                    #image[:, ch*32:(ch+1)*32] = image[:, ch*32:(ch+1)*32] - np.mean(image[normrows, ch*32:(ch+1)*32])
                image = np.abs(image)
                return image
            image_2d_ph1 = normalize_by_channel(image_2d_ph1)
            image_2d_ph2 = normalize_by_channel(image_2d_ph2)    
    
            # ~~~~~~~~~~~~~~~~~~
            # remove outliers
            def remove_outliers(data,Nstd=5):
                med=np.median(np.ravel(data))
                std=np.std(np.ravel(data))
                data[np.abs(data-med)>(Nstd*std)] = med-std
                return data
            image_2d_ph1 = remove_outliers(image_2d_ph1,Nstd=4)
            image_2d_ph2 = remove_outliers(image_2d_ph2,Nstd=4)    
            # ~~~~~~~~~~~~~~~~~~
            
            # re-normalize again
            #image_2d_ph1 = normalize_by_channel(image_2d_ph1)
            #image_2d_ph2 = normalize_by_channel(image_2d_ph2)    
#    
    
    
    
    
            image_1 = np.flip(np.transpose(image_2d_ph1))

            ##########################
            # normalize every column
            #for c in range(256):
            #    image_1[:,c] = image_1[:,c] - image_1[300,c]
            ##########################
            
            if image_1_ref is None:
                image_1_ref = image_1
                #continue
            #image_1 = image_1-image_1_ref
        
            tx = Get_Time(fullname,list_all[i])
            
            # subtract first image as a baseline
            #image_1 = image_1 - image_1_ref
            
            # subtract top row as baseline
            #for c in range(512):
            #    image_1[c,:] = image_1[c,:] - np.median(image_1[c,:])
        
        
            slicecol = 73
            xmin,xmax=[0,511]
            ymin,ymax=[0,255]
            
            # slicecol = 135
            # xmin,xmax=[24,34]
            # ymin,ymax=[120,150]
            
            # slicecol = 75
            # xmin,xmax=[38,48]
            # ymin,ymax=[50,100]
            
            # slicecol = 110
            # xmin,xmax=[300,511]
            # ymin,ymax=[100,200]
            
            # slicecol = 71
            # xmin,xmax=[120,150]
            # ymin,ymax=[64,95]
            
            #slicecol = 65 # 89
            #xmin,xmax=[225,280]
            #ymin,ymax=[40,150]

            slicecol = 180
            xmin,xmax=[150,280]
            ymin,ymax=[150,200]
            
            fig = plt.figure(figsize=(12,9))
            grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)
            ax = fig.add_subplot(grid[:3, :])
            mycolormap='Blues'    #'Blues' #'Greys'
            
            
            im1 = ax.imshow(image_1, #-np.median(image_1)), # [50:100,:40]),
                                vmin=np.mean(image_1[ymin:ymax,xmin:xmax])-6*np.std(image_1[ymin:ymax,xmin:xmax]), 
                                vmax=np.mean(image_1[ymin:ymax,xmin:xmax])+1*np.std(image_1[ymin:ymax,xmin:xmax]), 
                                cmap=mycolormap)
            #fig.colorbar(im1,ax=ax)
            ax.set_title('%u %u (%u,%u) %s time elapsed %s' % (lognum, 
                                                            i, 
                                                            row_offset,
                                                            col_offset,
                                                            list_all[i], 
                                                            str(tx-t0) ))
            
            plt.xlim([xmin,xmax])
            plt.ylim([ymin,ymax])
            plt.plot([0,512],[slicecol,slicecol],'r')


            myslice_x = range(xmin,xmax)
            myslice_y = image_1[slicecol,xmin:xmax]
            
            myslice_y = myslice_y - myslice_y[-1]
            #myslice_y = 1e-15 + 0.75 * myslice_y

            ax = fig.add_subplot(grid[-1,:])            
                
            im1 = ax.plot(myslice_x, myslice_y,'r')
            plt.xlim([xmin,xmax])
            #plt.ylim([0,1.5e-15])


            plt.show()
            


            print('row/col',row_offset, col_offset)
            print('slice',myslice_y)
            print('')
    
            if(row_offset >= -5):
                if(row_offset==-1):
                    npts = range(11+row_offset)
                else:
                    npts = range(10+row_offset)
                for x in npts:
                    minerva_data.append( [x, x-row_offset, myslice_y[x]] )
                    
            
            
print(minerva_data)
                
minerva_data = np.array(minerva_data)                
                
minerva_data[:,2] = minerva_data[:,2] * 1e15 * 0.5

#for x in range(minerva_data.shape[0]):
    

#minerva_data[:,2] = minerva_data[:,2] - np.min(minerva_data[:,2])
                
                
                
                
######################################################################   
######################################################################   
######################################################################   
######################################################################   
######################################################################   
######################################################################   
######################################################################   
######################################################################   
######################################################################   
######################################################################   
######################################################################   
######################################################################                
                





# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
#from __future__ import division, absolute_import, print_function

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import imageio


# MINERVA 
#minerva_data = np.loadtxt('minerva_output4.txt',delimiter=',')
# minerva_data[:,2] = -minerva_data[:,2]
# minerva_data[:,2] = minerva_data[:,2] - minerva_data[0,2]
# minerva_data[:,2] = 1 + 10*minerva_data[:,2]

#minerva_data[:,2] = 1/minerva_data[:,2]
#minerva_data[:,2] = minerva_data[:,2] - minerva_data[0,2]
#minerva_data[:,2] = 1.15 + 0.04*minerva_data[:,2]


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


meshwidth = 150e-6
meshheight = 80e-6
meshsize = meshwidth/50
n_el = 11
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
                               ) #subdivideregions = [ np.array([[-50e-6, 0], [50e-6, 100e-6]]),
                                 #                   np.array([[-50e-6, 0], [50e-6, 100e-6]]) ])

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
    ax.set_ylim([-0.1*meshheight, 1.05*meshheight])
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
bead_diameter = 20e-6
#bead_height = 30e-6

for bead_height in [11e-6,]:
    print('bead diameter',bead_diameter)
    print('bead height',bead_height)
    anomaly = [{"x": 0e-6, "y": bead_height, "d": bead_diameter/2, "perm": 0.1},
               {"bbox": np.array([[-meshwidth/2, -10e-6], [meshwidth, 0e-6]]), "perm": 0.5},]
#               {"x": 0e-6, "y": bead_height, "d": bead_diameter/3, "perm": 1}]
    mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly, background=1.0)
    perm = mesh_new["perm"]
    
    
    ###################
    # INVERSE SOLVER
    
    print('setting up inverse problem')
    
    
    # define the measurement matrix    
#    ex_mat = np.array( [ [x,x+1] for x in range(0,n_el-1) ] )
#    ex_mat = np.append( ex_mat, np.array( [ [x,x+2] for x in range(0,n_el-3) ] ), axis=0 )
#    ex_mat = np.append( ex_mat, np.array( [ [x,x+3] for x in range(0,n_el-4) ] ), axis=0 )
#    ex_mat = np.append( ex_mat, np.array( [ [x,x+4] for x in range(0,n_el-5) ] ), axis=0 )
#    ex_mat = np.append( ex_mat, np.array( [ [x,x+5] for x in range(0,n_el-6) ] ), axis=0 )

    # use the same measurements as the experimental data
    ex_mat = minerva_data[:,:2].astype(np.int)

    
    """ 2. calculate simulated data """    
    fwd = Forward(mesh_obj, el_pos)
    f1 = fwd.solve_eit(ex_mat, perm=mesh_new["perm"], parser="std")
    #print('simulated measurements',f1)
    

    
    # ~~~~~~~~~~~~~~~~~~
    meas_mat = np.hstack([ex_mat,f1.v[:,np.newaxis]])
    plt.figure(figsize=(8,4))    
    for spacing in range(6):
        pats = np.where(np.abs(meas_mat[:,1]-meas_mat[:,0])==spacing)[0]
        if(len(pats)>0):
            toplot = meas_mat[pats,2]
            #toplot = toplot - np.min(toplot)
            #toplot = toplot / np.max(toplot)
            plt.subplot(1,2,1)
            plt.plot(spacing + toplot,'.-')
            plt.title('simulated')


            # ~~~~~~~~~~~~~~~~
            # FORCE OFFSET TO MATCH SIMULATION
            minerva_data[pats,2] = minerva_data[pats,2] - minerva_data[pats,2][0]
            minerva_data[pats,2] = minerva_data[pats,2] + meas_mat[pats,2][0]
            # ~~~~~~~~~~~~~~~~
            
            toplot = minerva_data[pats,2] 
            #toplot = toplot - np.min(toplot)
            #toplot = toplot / np.max(toplot)
            plt.subplot(1,2,2)
            plt.plot(spacing + toplot,'.-')        
            plt.title('minerva, measured & rescaled')
    # ~~~~~~~~~~~~~~~~~~    
    plt.show()


    ect_gap = minerva_data[:,1] - minerva_data[:,0]
#    for spacing in range(1,6):
#        minerva_data[:,2][ect_gap==spacing] = minerva_data[:,2][ect_gap==spacing] - min(minerva_data[:,2][ect_gap==spacing])
#        minerva_data[:,2][ect_gap==spacing] = minerva_data[:,2][ect_gap==spacing] / (max(minerva_data[:,2][ect_gap==spacing]) - min(minerva_data[:,2][ect_gap==spacing])) * (max(meas_mat[:,2][ect_gap==spacing]) - min(meas_mat[:,2][ect_gap==spacing]))
#        minerva_data[:,2][ect_gap==spacing] = minerva_data[:,2][ect_gap==spacing] + min(meas_mat[:,2][ect_gap==spacing])
  

    """ 3. solve_eit using gaussian-newton (with regularization) """
    # number of stimulation lines/patterns
    eit = jac.JAC(mesh_obj, el_pos, ex_mat, perm=1.0, parser="std")
    eit.setup(p=0.25, lamb=10.0, method="lm")
    #eit.setup(p=0.25, lamb=10.0, method="kotre")
    # lamb = lamb * lamb_decay

    # dump f1.v file
    print(f1.v)

    
    print('solving from Minerva bead data')    
    ds_from_minerva = eit.gn(minerva_data[:,2], 
                             lamb_decay=0.5, 
                             lamb_min=1e-5, 
                             maxiter=10, 
                             verbose=True)

    print('solving from simulated data')
    ds_from_sim = eit.gn(f1.v, 
                         lamb_decay=0.1, 
                         lamb_min=1e-5, 
                         maxiter=5, 
                         verbose=True)

    
    
    # plot results
    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    overlay_grid_plot(ax1)  
    ax1.set_title('original')
    
    ax2 = fig.add_subplot(132)
    im = ax2.tripcolor(
        x,
        y,
        tri,
        np.real(ds_from_sim),
        cmap=plt.cm.Reds,
        antialiased=True,
        vmin=np.min(np.real(ds_from_sim)),
        vmax=np.max(np.real(ds_from_sim))
    )
    #fig.colorbar(im,ax=ax2,orientation='horizontal')
    overlay_grid_plot(ax2,solidmesh=False)  
    ax2.set_title('reconstruction from simulated data')

    ax3 = fig.add_subplot(133)
    im = ax3.tripcolor(
        x,
        y,
        tri,
        np.real(ds_from_minerva),
        #np.real(ds_from_minerva) - np.median(np.real(ds_from_minerva)) + np.median(np.real(ds_from_sim)),
        cmap=plt.cm.Reds,
        antialiased=True,
        #vmin=np.min(np.real(ds_from_sim)),
        #vmax=np.max(np.real(ds_from_sim))
    )
    #fig.colorbar(im,ax=ax2,orientation='horizontal')
    overlay_grid_plot(ax3,solidmesh=False)  
    ax3.set_title('reconstruction from minerva data')


    
    fig.set_size_inches(12, 8)
    

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    # plot results
    fig = plt.figure()
    ax = fig.add_subplot(111)

    im = ax.tripcolor(
        x,
        y,
        tri,
        np.real(ds_from_minerva),
        cmap=plt.cm.autumn,
        antialiased=True,
        vmin=0.6,
        vmax=0.95
        #vmin=np.mean(np.real(ds_from_minerva)) - 1*np.std(np.real(ds_from_minerva)),
        #vmax=np.mean(np.real(ds_from_minerva)) + 3*np.std(np.real(ds_from_minerva))
    )
    #fig.colorbar(im,ax=ax2,orientation='horizontal')
    #overlay_grid_plot(ax,solidmesh=False)  
    # draw electrodes
    ax.plot(x[el_pos], y[el_pos], "ks")
    for i in range(n_el):
        e = el_pos[i]
        ax1.text(x[e], y[e]-5e-6, str(i), size=8, horizontalalignment='center', verticalalignment='top')
    ax.set_title("equi-potential lines")
    # clean up
    ax.set_aspect("equal")
    scale_x,scale_y = 1e-6,1e-6
    ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale_x))
    ax.xaxis.set_major_formatter(ticks_x)        
    ticks_y = ticker.FuncFormatter(lambda y, pos: '{0:g}'.format(y/scale_y))
    ax.yaxis.set_major_formatter(ticks_y)      
    ax.set_xlabel('microns')
    ax.set_ylabel('microns (height)')
    
    ax.set_title('reconstruction from minerva data')
    ax.set_ylim([0, 60e-6])
    ax.set_xlim([-0.35*meshwidth, 0.35*meshwidth])

    fig.set_size_inches(6, 3)
    
    fig.savefig('reconstruction.pdf')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    



    
    # add to frames for animation
    fig.canvas.draw()       # draw the canvas, cache the renderer
    im = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    im  = im.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    myframes.append(im)
    



if 0:
    # create .mp4 video file
    imageio.mimsave('EIT_test_1a.mp4', 
                    myframes, 
                    fps=5)

if 0:
    # create animated .gif
    imageio.mimsave('EIT_test_5a.gif', 
                    myframes, 
                    fps=2)