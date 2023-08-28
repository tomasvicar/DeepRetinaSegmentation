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


class LoaderG1020(LoaderGeneric):
    
    @property
    def fnames_imgs(self):
        
        return glob(self.data_path + '/G1020/**/*.jpg', recursive=True)
        
    
    @property
    def masks_getfname_fcns(self):
        
        masks_getfname_fcns = dict()
        
        def does_contain(name, text):
            with open(name, 'r') as f:
                output = f.read()
                return '"' + text + '"' in output
        
        
        def name_disk(x):
            if does_contain(x.replace('.jpg', '.json'), 'disc'):
                return x.replace('.jpg', '.json_disc')
            else:
                return None
        
        def name_cup(x):
            if does_contain(x.replace('.jpg', '.json'), 'cup'):
                return x.replace('.jpg', '.json_cup')
            else:
                return None
        
        masks_getfname_fcns[LoaderGeneric.DISK] = name_disk
        masks_getfname_fcns[LoaderGeneric.CUP] = name_cup
        
        return  masks_getfname_fcns
    
    @property
    def preprocess_fcns(self):
        preprocess_fcns = dict()
        preprocess_fcns[LoaderGeneric.DISK] = lambda x : x
        preprocess_fcns[LoaderGeneric.CUP] = lambda x : x
        
        return preprocess_fcns
    
                              
    @property 
    def read_img_fcn(self):
        
        def read_json_mask(name, data_type, img_shape):
            with open(name, 'r') as f:
                data = json.load(f)
                for json_shape in data['shapes']:
                    if json_shape['label'] == data_type:
                        img = np.zeros(img_shape, dtype=np.uint8)
                        points = np.array(json_shape['points'], np.int32) - 1
                        # -1 for python
                        cv2.fillPoly(img, [points], 255)
                        return img > 128
            return None
        
        
        def read_fcn(name):
        
            if  '.json' in name:
                type_ = name.split('_')[-1]
                name = '_'.join(name.split('_')[:-1])
                img_shape = imread(name.replace('.json', '.jpg')).shape[:2]
                return read_json_mask(name, type_, img_shape)
            else:
                return imread(name)
            
        return read_fcn
                
            
        



    def get_fov(self, fname_img, img):
        
        fov = self.get_fov_auto(img, t1=7, t2=15)
        return fov
    
    
    @property
    def dataset_fov_deg(self):
        return 45
    
    def get_savename(self, fname_img):
        return 'g1020_na_na_'
            
        
        
        
        
    
if __name__ == "__main__":
    
    data_path = '../../databases'
    pix_per_deg = 25
    preprocess_f = lambda img, fov, pix_per_deg : img
    # preprocess_f = local_contrast_and_clahe
    out_fname = '../../data_' + str(pix_per_deg) + '.hdf5'
    
    if os.path.exists(out_fname):
        os.remove(out_fname)

    loader = LoaderG1020(data_path, pix_per_deg, out_fname, preprocess_f)
    # loader.preprocess()
    # loader.preprocess(show_fov=True)
    loader.preprocess(show_masks=True)
    
    with h5py.File(out_fname, "r") as f:
        
        
        num = 1
        image_names = [n for n in f.keys()]
        print(image_names)
        image_types = [n for n in f[image_names[num]].keys()]
        print(image_types)
        
        for image_type in f[image_names[num]]:
            img = f[image_names[num]][image_type][...]
            print(image_type)
            print(np.max(img))
            print(np.min(img))
            print(img.shape)
            print(img.dtype)
            print(f[image_names[num]][image_type].attrs['orig_name'])
            plt.imshow(img)
            plt.show()

        
            


