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
import json
import cv2


import sys
sys.path.append("..")
from utils.local_contrast_and_clahe import local_contrast_and_clahe


class LoaderHrf(LoaderGeneric):
    
    @property
    def fnames_imgs(self):
        
        return glob(self.data_path + '/HRF/**/*.jpg', recursive=True)
        
    
    @property
    def masks_getfname_fcns(self):
        
        masks_getfname_fcns = dict()
        masks_getfname_fcns[LoaderGeneric.VESSEL] = lambda x : x.replace('.JPG','.jpg').replace('.jpg','.tif').replace('images','manual1')
        masks_getfname_fcns[LoaderGeneric.VESSEL_CLASS] = lambda x : x.replace('.JPG','.jpg').replace('.jpg','.png').replace('images','clasified')    
        return  masks_getfname_fcns
    
    @property
    def preprocess_fcns(self):
        preprocess_fcns = dict()
        preprocess_fcns[LoaderGeneric.VESSEL] = lambda x : x > 0
        preprocess_fcns[LoaderGeneric.VESSEL_CLASS] = lambda x : np.stack((x[:,:,0] > 128, x[:,:,2] > 128, x[:,:,1] > 128), axis=2)
        
        return preprocess_fcns
    
                              


    def get_fov(self, fname_img, img):
        
        fov = imread(fname_img.replace('.JPG','.jpg').replace('.jpg','_mask.tif').replace('images','mask'))
        
        return fov
    
    
    @property
    def dataset_fov_deg(self):
        return 60
    
    def get_savename(self, fname_img):
        if '_g.' in fname_img:
            return 'hrf_na_glaucoma_'
        if '_h.' in fname_img:
            return 'hrf_na_normal_'
        if '_dr.' in fname_img:
            return 'hrf_na_dr_'
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

    loader = LoaderHrf(data_path, pix_per_deg, out_fname, preprocess_f)
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


