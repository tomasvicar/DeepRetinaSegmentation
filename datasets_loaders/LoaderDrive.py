import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import h5py

from LoaderGeneric import LoaderGeneric

from skimage.morphology import binary_closing, disk
from scipy.ndimage import binary_fill_holes
from skimage.measure import label
from skimage.io import imread
import re
import pandas as pd

import sys
sys.path.append("..")
from utils.local_contrast_and_clahe import local_contrast_and_clahe


class LoaderDrive(LoaderGeneric):
    
    @property
    def fnames_imgs(self):
        
        return glob(self.data_path + '/DRIVE/**/*.bmp', recursive=True)
        
    
    @property
    def masks_getfname_fcns(self):
        
        masks_getfname_fcns = dict()
        masks_getfname_fcns[LoaderGeneric.VESSEL] = lambda x : x.replace('images', '1st_manual').replace('.bmp', '_manual1.gif').replace('test_','').replace('training_','')
        
        def name_vessel2(x):
            
            x = x.replace('images', '2nd_manual').replace('.bmp', '_manual2.gif').replace('test_','').replace('training_','')
            
            if os.path.exists(x):
                return x
            else:
                return None
            
        
        masks_getfname_fcns[LoaderGeneric.VESSEL2] = name_vessel2
        
        return  masks_getfname_fcns
    
    @property
    def preprocess_fcns(self):
        preprocess_fcns = dict()
        preprocess_fcns[LoaderGeneric.VESSEL] = lambda x : x[0] > 128
        preprocess_fcns[LoaderGeneric.VESSEL2] = lambda x : x[0,:,:,0] > 128
        
        return preprocess_fcns
    
    
                                

    def get_fov(self, fname_img, img):
        
        fov = imread(fname_img.replace('images', 'mask').replace('.bmp', '_mask.gif'))
        fov = fov[0]
        
        return fov
    
    
    @property
    def dataset_fov_deg(self):
        return 45
    
    def get_savename(self, fname_img):
        


        if 'training' in fname_img:
            if int(os.path.split(fname_img)[1][:2]) in [25, 26, 32]:
                diagnose = 'dr'
            else:
                diagnose = 'normal'
            
            return 'drive_train_' + diagnose + '_'
        elif 'test' in fname_img:
            if int(os.path.split(fname_img)[1][:2]) in [3, 8, 14, 17]:
                diagnose = 'dr'
            else:
                diagnose = 'normal'
            
            return 'drive_test_' + diagnose + '_'
        else:
            raise(Exception('neni'))
            
        
        
        
        
    
if __name__ == "__main__":
    
    data_path = '../../databases'
    pix_per_deg = 25
    preprocess_f = lambda img, fov, pix_per_deg : img
    # preprocess_f = local_contrast_and_clahe
    out_fname = '../../data_' + str(pix_per_deg) + '.hdf5'
    
    if os.path.exists(out_fname):
        os.remove(out_fname)

    loader = LoaderDrive(data_path, pix_per_deg, out_fname, preprocess_f)
    # loader.preprocess()
    # loader.preprocess(show_fov=True)
    loader.preprocess(show_masks=True)
    
    with h5py.File(out_fname, "r") as f:
        image_names = [n for n in f.keys()]
        print(image_names)
        image_types = [n for n in f[image_names[0]].keys()]
        print(image_types)
        
        image_dts = f[image_names[0]][image_types[1]]
        image = image_dts[...]
        
        print(image_dts.attrs['orig_name'])
        plt.imshow(image)


