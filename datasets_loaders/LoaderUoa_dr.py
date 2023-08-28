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
from skimage.measure import label, regionprops

import sys
sys.path.append("..")
from utils.local_contrast_and_clahe import local_contrast_and_clahe


def zero_region(image):
    rows, cols, _ = image.shape
    start_row = round(0.97 * rows)
    end_col = round(0.25 * rows)
    
    image[start_row:, :end_col, ...] = 0
    return image


class LoaderUoa_dr(LoaderGeneric):
    
    @property
    def fnames_imgs(self):
        
        names = glob(self.data_path + '/UoA_DR/*') 
        
        
        return [name + '/' + os.path.split(name)[1] + '.jpg' for name in names if len(os.path.split(name)[1]) <= 3]
        
    
            
    
    @property
    def masks_getfname_fcns(self):
        
        def none_if_not_exists(name):
            if os.path.exists(name):
                return name
            else:
                return None
        
        masks_getfname_fcns = dict()
        masks_getfname_fcns[LoaderGeneric.VESSEL] = lambda x : x.replace('.jpg','.1.jpg')
        masks_getfname_fcns[LoaderGeneric.DISK] = lambda x : x.replace('.jpg','.2.jpg')
        masks_getfname_fcns[LoaderGeneric.CUP] = lambda x : x.replace('.jpg','.3.jpg')

        return  masks_getfname_fcns
    
    
    
    @property
    def preprocess_fcns(self):
        preprocess_fcns = dict()
        preprocess_fcns[LoaderGeneric.VESSEL] = lambda x : x[:, :, 0] > 128
        preprocess_fcns[LoaderGeneric.DISK] = lambda x : x[:, :, 0] > 128
        preprocess_fcns[LoaderGeneric.CUP] = lambda x : x[:, :, 0] > 128
        return preprocess_fcns
    
    @property 
    def read_img_fcn(self):
        
        def read_im(name):
            return zero_region(imread(name))
        
        return read_im

    def get_fov(self, fname_img, img):
        fov = self.get_fov_auto(img, t1=7, t2=15)
        return fov
    
    
    @property
    def dataset_fov_deg(self):
        return 45
    
    def get_savename(self, fname_img):
        my_list_npdr = list(range(1, 82)) + list(range(83, 95)) + [130] + list(range(132, 144)) + [168, 169] + list(range(171, 175)) + [179, 193]
        my_list_pdr = [82] + list(range(95, 101)) + [131, 167] + list(range(176, 179)) + list(range(182, 193)) + list(range(194, 201))
        my_list_normal = list(range(101, 130)) + list(range(144, 167)) + [170, 175, 180, 181]

        num = int(os.path.split(fname_img)[-1][:-4])
        if num in my_list_npdr:  
            return 'uoadr_na_npdr_'
        elif num in my_list_pdr:  
            return 'uoadr_na_pdr_'
        elif num in my_list_normal:  
            return 'uoadr_na_normal_'
        else:
            raise(Exception('nenalezena_patologie'))
        
        
    
if __name__ == "__main__":
    
    data_path = '../../databases'
    pix_per_deg = 25
    preprocess_f = lambda img, fov, pix_per_deg : img
    # preprocess_f = local_contrast_and_clahe
    out_fname = '../../data_' + str(pix_per_deg) + '.hdf5'
    
    if os.path.exists(out_fname):
        os.remove(out_fname)

    loader = LoaderUoa_dr(data_path, pix_per_deg, out_fname, preprocess_f)
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


