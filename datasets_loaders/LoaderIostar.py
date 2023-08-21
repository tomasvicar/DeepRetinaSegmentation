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
from scipy.ndimage import binary_fill_holes

import sys
sys.path.append("..")
from utils.local_contrast_and_clahe import local_contrast_and_clahe


class LoaderIostar(LoaderGeneric):
    
    @property
    def fnames_imgs(self):
        

        
        return glob(self.data_path + '/IOSTAR/image/*.jpg', recursive=True)
        
    
    @property
    def masks_getfname_fcns(self):
        
        masks_getfname_fcns = dict()
        masks_getfname_fcns[LoaderGeneric.VESSEL] = lambda x : x.replace('.jpg', '_GT.tif').replace('image','GT')
        masks_getfname_fcns[LoaderGeneric.VESSEL_CLASS] = lambda x : x.replace('.jpg', '_AV.tif').replace('image','AV_GT')
        masks_getfname_fcns[LoaderGeneric.DISK] = lambda x : x.replace('.jpg', '_ODMask.tif').replace('image','mask_OD')
        

        return  masks_getfname_fcns
    
    @property
    def preprocess_fcns(self):
        preprocess_fcns = dict()
        preprocess_fcns[LoaderGeneric.VESSEL] = lambda x : x > 0
        
        def get_vessel_class(x):
            r = x[:, :, 0]
            g = x[:, :, 1]
            b = x[:, :, 2]
            art = (r == 255) & (b == 0)
            vei = (b == 255) & (r == 0)
            
            return np.stack((art, vei, np.zeros_like(vei)),axis=2)
        
        preprocess_fcns[LoaderGeneric.VESSEL_CLASS] = get_vessel_class
        preprocess_fcns[LoaderGeneric.DISK] = lambda x : binary_fill_holes(x  > 0) & (x == 0)

        return preprocess_fcns
    
  

    def get_fov(self, fname_img, img):
        fov = imread(fname_img.replace('.jpg', '_Mask.tif').replace('image','mask')) > 0
        return fov
    
    @property
    def dataset_fov_deg(self):
        return 45
    
    def get_savename(self, fname_img):
        return 'iostar_na_na_'
        
        
        
        
    
if __name__ == "__main__":
    
    data_path = '../../databases'
    pix_per_deg = 25
    preprocess_f = lambda img, fov, pix_per_deg : img
    # preprocess_f = local_contrast_and_clahe
    out_fname = '../../data_' + str(pix_per_deg) + '.hdf5'
    
    if os.path.exists(out_fname):
        os.remove(out_fname)

    loader = LoaderIostar(data_path, pix_per_deg, out_fname, preprocess_f)
    # loader.preprocess()
    # loader.preprocess(show_fov=True)
    loader.preprocess(show_masks=True)
    
    with h5py.File(out_fname, "r") as f:
        image_names = [n for n in f.keys()]
        print(image_names)
        image_types = [n for n in f[image_names[0]].keys()]
        print(image_types)
        
        image_dts = f[image_names[0]][image_types[2]]
        image = image_dts[...]
        
        print(image_dts.attrs['orig_name'])
        plt.imshow(image)


