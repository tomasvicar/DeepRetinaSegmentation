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


def filter_largest_object(image, rank=1):
    """
    Filters the largest or second-largest object from a binary image.

    Parameters:
        image (numpy.ndarray): Binary image containing the objects.
        rank (int): Rank of the object size to keep (1 for largest, 2 for second-largest, etc.).

    Returns:
        numpy.ndarray: Binary image containing only the object of specified rank.
    """
    labeled_image = label(image)
    regions = regionprops(labeled_image)
    
    if len(regions) == 0:
        return np.zeros_like(image)
    
    # Sort regions by area in descending order
    regions = sorted(regions, key=lambda x: x.area, reverse=True)
    
    if len(regions) < rank:
        return np.zeros_like(image)

    # Keep only the object of specified rank
    selected_region = regions[rank - 1]
    filtered_image = np.zeros_like(image)
    filtered_image[labeled_image == selected_region.label] = True
    
    return filtered_image





class LoaderRiga(LoaderGeneric):
    
    @property
    def fnames_imgs(self):
        
        return glob(self.data_path + '/RIGA/**/*prime.jpg', recursive=True) + glob(self.data_path + '/RIGA/**/*prime.tif', recursive=True)
        
    
            
    
    @property
    def masks_getfname_fcns(self):
        
        masks_getfname_fcns = dict()
        masks_getfname_fcns[LoaderGeneric.DISK] = lambda x : x.replace('.tif', '.jpg').replace('prime.jpg', '-1.jpg') + '_disk'
        masks_getfname_fcns[LoaderGeneric.DISK2] = lambda x : x.replace('.tif', '.jpg').replace('prime.jpg', '-2.jpg') + '_disk'
        masks_getfname_fcns[LoaderGeneric.DISK3] = lambda x : x.replace('.tif', '.jpg').replace('prime.jpg', '-3.jpg') + '_disk'
        masks_getfname_fcns[LoaderGeneric.DISK4] = lambda x : x.replace('.tif', '.jpg').replace('prime.jpg', '-4.jpg') + '_disk'
        masks_getfname_fcns[LoaderGeneric.DISK5] = lambda x : x.replace('.tif', '.jpg').replace('prime.jpg', '-5.jpg') + '_disk'
        masks_getfname_fcns[LoaderGeneric.DISK6] = lambda x : x.replace('.tif', '.jpg').replace('prime.jpg', '-6.jpg') + '_disk'
        
        
        masks_getfname_fcns[LoaderGeneric.CUP] = lambda x : x.replace('.tif', '.jpg').replace('prime.jpg', '-1.jpg') + '_cup'
        masks_getfname_fcns[LoaderGeneric.CUP2] = lambda x : x.replace('.tif', '.jpg').replace('prime.jpg', '-2.jpg') + '_cup'
        masks_getfname_fcns[LoaderGeneric.CUP3] = lambda x : x.replace('.tif', '.jpg').replace('prime.jpg', '-3.jpg') + '_cup'
        masks_getfname_fcns[LoaderGeneric.CUP4] = lambda x : x.replace('.tif', '.jpg').replace('prime.jpg', '-4.jpg') + '_cup'
        masks_getfname_fcns[LoaderGeneric.CUP5] = lambda x : x.replace('.tif', '.jpg').replace('prime.jpg', '-5.jpg') + '_cup'
        masks_getfname_fcns[LoaderGeneric.CUP6] = lambda x : x.replace('.tif', '.jpg').replace('prime.jpg', '-6.jpg') + '_cup'

        return  masks_getfname_fcns
    
    
    @property 
    def read_img_fcn(self):
        
        def read_disk_or_cup(name):
            
            
            if ('_disk' in name) or ('_cup' in name):
                name_img = '-'.join(name.split('-')[:-1]) +'prime.jpg'
                name_mask = name.replace('_disk', '').replace('_cup', '')
                
                if not os.path.exists(name_img):
                    name_img =  name_img.replace('.jpg', '.tif')
                if not os.path.exists(name_mask):
                    name_mask = name_mask.replace('.jpg', '.tif')
                
                img_orig = imread(name_img)
                img = imread(name_mask)
                mask = (img_orig[:,:,1].astype(np.float32) - img[:,:,1].astype(np.float32))>15
                
                if('_disk' in name):
                    rank = 1
                elif ('_cup' in name):
                    rank = 2
                
                mask = filter_largest_object(mask, rank=rank)
                return binary_fill_holes(mask)

            else:
                return imread(name)
            
        
        
        
        return read_disk_or_cup
    
    @property
    def preprocess_fcns(self):
        preprocess_fcns = dict()
        preprocess_fcns[LoaderGeneric.DISK] = lambda x : x > 0
        preprocess_fcns[LoaderGeneric.DISK2] = lambda x : x > 0
        preprocess_fcns[LoaderGeneric.DISK3] = lambda x : x > 0
        preprocess_fcns[LoaderGeneric.DISK4] = lambda x : x > 0
        preprocess_fcns[LoaderGeneric.DISK5] = lambda x : x > 0
        preprocess_fcns[LoaderGeneric.DISK6] = lambda x : x > 0
        
        preprocess_fcns[LoaderGeneric.CUP] = lambda x : x > 0
        preprocess_fcns[LoaderGeneric.CUP2] = lambda x : x > 0
        preprocess_fcns[LoaderGeneric.CUP3] = lambda x : x > 0
        preprocess_fcns[LoaderGeneric.CUP4] = lambda x : x > 0
        preprocess_fcns[LoaderGeneric.CUP5] = lambda x : x > 0
        preprocess_fcns[LoaderGeneric.CUP6] = lambda x : x > 0

        return preprocess_fcns
    
  

    def get_fov(self, fname_img, img):
        fov = self.get_fov_auto(img, t1=7, t2=15)
        return fov
    
    
    @property
    def dataset_fov_deg(self):
        return 45
    
    def get_savename(self, fname_img):
        
        return 'riga_na_na_'
        
        
        
        
    
if __name__ == "__main__":
    
    data_path = '../../databases'
    pix_per_deg = 25
    preprocess_f = lambda img, fov, pix_per_deg : img
    # preprocess_f = local_contrast_and_clahe
    out_fname = '../../data_' + str(pix_per_deg) + '.hdf5'
    
    if os.path.exists(out_fname):
        os.remove(out_fname)

    loader = LoaderRiga(data_path, pix_per_deg, out_fname, preprocess_f)
    # loader.preprocess()
    loader.preprocess(show_fov=True)
    # loader.preprocess(show_masks=True)
    
    with h5py.File(out_fname, "r") as f:
        image_names = [n for n in f.keys()]
        print(image_names)
        image_types = [n for n in f[image_names[0]].keys()]
        print(image_types)
        
        image_dts = f[image_names[0]][image_types[2]]
        image = image_dts[...]
        
        print(image_dts.attrs['orig_name'])
        plt.imshow(image)


