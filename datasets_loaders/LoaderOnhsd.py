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
from scipy.io import loadmat
import cv2

import sys
sys.path.append("..")
from utils.local_contrast_and_clahe import local_contrast_and_clahe


class LoaderOnhsd(LoaderGeneric):
    
    @property
    def fnames_imgs(self):
        
        return glob(self.data_path + '/Onhsd/Images/*.bmp', recursive=True)
        
    
    @property
    def masks_getfname_fcns(self):
        
        masks_getfname_fcns = dict()
        
        
        def test_quality(name):
            img_shape = imread(('_'.join(name.split('_')[:-1]) + '.bmp').replace('Clinicians', 'Images')).shape[:2]
            name_center = '_'.join(name.split('_')[:-1]) + '_C.mat'
            center = loadmat(name_center)['ONHCentre'][0]
            dist = loadmat(name)['ONHEdge'][:, 0]
            
            x = []
            y = []
            for point_num in range(24):
                angle = np.pi / 12 * (point_num + 1)
                x.append(center[0] - 1 + np.cos(angle) * dist[point_num])
                y.append(center[1] - 1 + np.sin(angle) * dist[point_num])
                # -1 for python
                
            
            img = np.zeros(img_shape, dtype=np.uint8)
            points = np.round(np.stack((x, y), axis=1)).astype(np.int32)
            
            unique_points = np.unique(points, axis=0)
            
            if len(points) != len(unique_points):
                return None
            else:
                return name
            
        
        masks_getfname_fcns[LoaderGeneric.DISK] = lambda x : test_quality(x.replace('.bmp', '_AnsuONH.mat').replace('Images','Clinicians'))
        masks_getfname_fcns[LoaderGeneric.DISK2] = lambda x : test_quality(x.replace('.bmp', '_Bob.mat').replace('Images','Clinicians'))
        masks_getfname_fcns[LoaderGeneric.DISK3] = lambda x : test_quality(x.replace('.bmp', '_David.mat').replace('Images','Clinicians'))
        masks_getfname_fcns[LoaderGeneric.DISK4] = lambda x : test_quality(x.replace('.bmp', '_Lee.mat').replace('Images','Clinicians'))
        
        
        
        
        
        
        return  masks_getfname_fcns
    
    @property
    def preprocess_fcns(self):
        preprocess_fcns = dict()
        preprocess_fcns[LoaderGeneric.DISK] = lambda x : x
        preprocess_fcns[LoaderGeneric.DISK2] = lambda x : x
        preprocess_fcns[LoaderGeneric.DISK3] = lambda x : x
        preprocess_fcns[LoaderGeneric.DISK4] = lambda x : x

        return preprocess_fcns
    
  
    @property 
    def read_img_fcn(self):  
        def read_img_mat(name):
            if '.mat' in name:
                img_shape = imread(('_'.join(name.split('_')[:-1]) + '.bmp').replace('Clinicians', 'Images')).shape[:2]
                name_center = '_'.join(name.split('_')[:-1]) + '_C.mat'
                center = loadmat(name_center)['ONHCentre'][0]
                dist = loadmat(name)['ONHEdge'][:, 0]
                
                x = []
                y = []
                for point_num in range(24):
                    angle = np.pi / 12 * (point_num + 1)
                    x.append(center[0] - 1 + np.cos(angle) * dist[point_num])
                    y.append(center[1] - 1 + np.sin(angle) * dist[point_num])
                    # -1 for python
                    
                
                img = np.zeros(img_shape, dtype=np.uint8)
                points = np.round(np.stack((x, y), axis=1)).astype(np.int32)
                # print(points)
                img = cv2.fillPoly(img, [points], 255) > 128
                return img
                    
            else:
                return imread(name)
        
        return read_img_mat
     

    def get_fov(self, fname_img, img):
        fov = self.get_fov_auto(img, t1=7, t2=15)
        return fov
    
    @property
    def dataset_fov_deg(self):
        return 45
    
    def get_savename(self, fname_img):
        return 'onhsd_na_na_'
        
        
        
        
    
if __name__ == "__main__":
    
    data_path = '../../databases'
    pix_per_deg = 25
    preprocess_f = lambda img, fov, pix_per_deg : img
    # preprocess_f = local_contrast_and_clahe
    out_fname = '../../data_' + str(pix_per_deg) + '.hdf5'
    
    if os.path.exists(out_fname):
        os.remove(out_fname)

    loader = LoaderOnhsd(data_path, pix_per_deg, out_fname, preprocess_f)
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


