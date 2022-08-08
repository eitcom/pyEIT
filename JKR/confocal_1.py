#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 10:53:30 2022

@author: jkrosens
"""

import glob
import numpy as np
import matplotlib.pyplot as plt


logdir = r"/Users/jkrosens/Dropbox/HyBiScIs_data/Rosenstein Lab Data/F0012_06132922_confocal_ECT_biofilm_JSSC/confocal data"


lognames = glob.glob(logdir+'/ECT_Cell_Training_06132022_CFP-002.npy')




for myfile in lognames:
    x = np.load(myfile)
    

    
    im = x[:,:,7]
    
    plt.figure(figsize=(4,8))
    plt.imshow(np.transpose(im), 
               vmin=np.mean(im)-1.5*np.std(im),
               vmax=np.mean(im)+1.5*np.std(im),
               cmap='Blues_r')
    #plt.colorbar()
    plt.show()



    #im = x[:,:,7]
    im = x[:,400,:]
    
    plt.figure(figsize=(4,8))
    plt.imshow(np.transpose(im), 
               vmin=np.mean(im)-0.6*np.std(im),
               vmax=np.mean(im)+0.25*np.std(im),
               cmap='Reds')
    plt.ylim(0,120)
    plt.colorbar()
    plt.show()
    


