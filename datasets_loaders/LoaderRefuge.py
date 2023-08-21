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


class LoaderRefuge(LoaderGeneric):
    
    @property
    def fnames_imgs(self):
        
        self.mask_names = [x for x in glob(self.data_path + '/REFUGE/**/*.bmp', recursive=True) if 'Annotation' not in x]
        return [x for x in glob(self.data_path + '/REFUGE/**/*.jpg', recursive=True) if 'Annotation' not in x]
        
    
    def get_name_from_masknames(self, name):
        name_tail = os.path.split(name)[1].replace('.jpg', '')
        for mask_name in self.mask_names:
            if name_tail in mask_name:
                return mask_name
            
        raise(Exception('nenalezeno'))
            
    
    @property
    def masks_getfname_fcns(self):
        
        masks_getfname_fcns = dict()
        masks_getfname_fcns[LoaderGeneric.DISK] = lambda x : self.get_name_from_masknames(x) + '_disk'
        masks_getfname_fcns[LoaderGeneric.CUP] = lambda x : self.get_name_from_masknames(x) + '_cup'

        return  masks_getfname_fcns
    
    
    @property 
    def read_img_fcn(self):
        
        def read_disk_or_cup(name):
            if '_disk' in name:
                name = name.replace('_disk', '')
                img = imread(name)
                return img <= 128
            elif '_cup' in name:
                name = name.replace('_cup', '')
                img = imread(name)
                return img == 0
            else:
                return imread(name)
            
        
        
        
        return read_disk_or_cup
    
    @property
    def preprocess_fcns(self):
        preprocess_fcns = dict()
        preprocess_fcns[LoaderGeneric.DISK] = lambda x : x > 0
        preprocess_fcns[LoaderGeneric.CUP] = lambda x : x > 0

        return preprocess_fcns
    
  

    def get_fov(self, fname_img, img):
        fov = self.get_fov_auto(img, t1=7, t2=15)
        return fov
    
    @property
    def dataset_fov_deg(self):
        return 45
    
    def get_savename(self, fname_img):
        
        if 'Test' in fname_img:
            dataset = 'test'
        elif 'Training' in fname_img:
            dataset = 'train'
        elif 'Validation' in fname_img:
            dataset = 'valid'    
        else:
            raise(Exception('split not found'))
            
        mask_name = self.get_name_from_masknames(fname_img)
            
            
        if 'Glaucoma' in fname_img or 'Glaucoma' in mask_name:
            diagnosis = 'glaucoma'
        elif 'Non-Glaucoma' in fname_img or 'Non-Glaucoma' in mask_name:
            diagnosis = 'normal' 
        if '/G/' in fname_img or '/G/' in mask_name:
            diagnosis = 'glaucoma'
        elif '/N/' in fname_img or '/N/' in mask_name:
            diagnosis = 'normal'    
        elif 'Validation' in fname_img:
            diagnosis = 'na' 
        else:
            raise(Exception('diagnosis not found'))
        
        return 'refuge_' + dataset + '_' + diagnosis + '_'
        
        
        
        
    
if __name__ == "__main__":
    
    data_path = '../../databases'
    pix_per_deg = 25
    preprocess_f = lambda img, fov, pix_per_deg : img
    # preprocess_f = local_contrast_and_clahe
    out_fname = '../../data_' + str(pix_per_deg) + '.hdf5'
    
    if os.path.exists(out_fname):
        os.remove(out_fname)

    loader = LoaderRefuge(data_path, pix_per_deg, out_fname, preprocess_f)
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


