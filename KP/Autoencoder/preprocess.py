# -*- coding: utf-8 -*-
from torch.utils.data import Dataset
import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.transform import downscale_local_mean



class MyDataset(Dataset):
    def __init__(self, input_dir, data_type):
        super().__init__()

        # initialize dataset
        self.cap_pair = []
        self.perm_img = []
        self.name = []

        num_beads_list = [1,2,3]

        if data_type == 'train':
            d_range = 10000
            
        elif data_type == 'test':
            d_range = 1000
            
        else:
            d_range = 10

        # read in the data
        for num_beads in num_beads_list:
            for i in range(d_range):
                f_perm = input_dir + '/permittivity_'+str(num_beads)+'_beads_'+str(i)+'.npy'
                f_cap = input_dir + '/capacitance_'+str(num_beads)+'_beads_'+str(i)+'.npy'
    
                with open(f_perm, 'rb') as f:
                    # down sample the image
                    perm_downsample = downscale_local_mean(np.load(f), (3,3))
                    
                    # replace nan with 1
                    perm_downsample[np.isnan(perm_downsample)] = 1
                    self.perm_img.append(perm_downsample.reshape(-1))
    
                with open(f_cap, 'rb') as f:
                    self.cap_pair.append(np.load(f))
                    
                self.name.append(str(num_beads)+'_beads_'+str(i))

        # print(self.perm_img[0].mask)
        # plt.figure(figsize=(6,6))
        # plt.imshow(self.perm_img[0],cmap='Greys_r')
        # plt.show()


    def __len__(self):
        """
        __len__ should return a the length of the dataset

        :return: an integer length of the dataset
        """
        return len(self.cap_pair)
    

    def __getitem__(self, idx):
        """
        __getitem__ should return a tuple or dictionary of the data at some index

        :param idx: the index for retrieval

        :return: tuple or dictionary of the data
        """
        item = {
            "cap_pair": self.cap_pair[idx],
            "perm_img": self.perm_img[idx],
            "name": self.name[idx]
            }
        
        return item



if __name__ == "__main__":
    
    input_dir = "train/"
    data = MyDataset(input_dir, 'train')
