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

import sys
sys.path.append("..")
from utils.local_contrast_and_clahe import local_contrast_and_clahe


class LoaderChasedb1(LoaderGeneric):
    
    @property
    def fnames_imgs(self):
        
        
        return glob(self.data_path + '/CHASEDB1/*.jpg') 
        
    
    @property
    def masks_getfname_fcns(self):
        
        masks_getfname_fcns = dict()
        masks_getfname_fcns[LoaderGeneric.VESSEL] = lambda x : x.replace('.jpg', '_1stHO.png')
        masks_getfname_fcns[LoaderGeneric.VESSEL2] = lambda x : x.replace('.jpg', '_2ndHO.png')
        
        return  masks_getfname_fcns
    
    @property
    def preprocess_fcns(self):
        preprocess_fcns = dict()
        preprocess_fcns[LoaderGeneric.VESSEL] = lambda x : x
        preprocess_fcns[LoaderGeneric.VESSEL2] = lambda x : x  

        return preprocess_fcns
    
    
                                

    def get_fov(self, fname_img, img):
        
        fov = self.get_fov_auto(img, t1=7, t2=15)
        return fov
    
    @property
    def dataset_fov_deg(self):
        return 30
    
    def get_savename(self, fname_img):
        return 'chasedb1_na_child_'
        
        
        
        
    
if __name__ == "__main__":
    
    data_path = '../../databases'
    pix_per_deg = 25
    preprocess_f = lambda img, fov, pix_per_deg : img
    # preprocess_f = local_contrast_and_clahe
    out_fname = '../../data_' + str(pix_per_deg) + '.hdf5'
    
    if os.path.exists(out_fname):
        os.remove(out_fname)

    loader = LoaderChasedb1(data_path, pix_per_deg, out_fname, preprocess_f)
    # loader.preprocess()
    loader.preprocess(show_fov=True)
    # loader.preprocess(show_masks=True)
    
    with h5py.File(out_fname, "r") as f:
        image_names = [n for n in f.keys()]
        print(image_names)
        image_types = [n for n in f[image_names[0]].keys()]
        print(image_types)
        
        image_dts = f[image_names[0]][image_types[1]]
        image = image_dts[...]
        
        print(image_dts.attrs['orig_name'])
        plt.imshow(image)


