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


class LoaderAvrdb(LoaderGeneric):
    
    @property
    def fnames_imgs(self):
        
        pattern = re.compile(r'.*\.JPG$')
        
        return  [f for f in glob(self.data_path + '/AVRDB/**/*') if pattern.match(f)]
        
    
    @property
    def masks_getfname_fcns(self):
        
        masks_getfname_fcns = dict()
        masks_getfname_fcns[LoaderGeneric.VESSEL] = lambda x : x.replace('.JPG', '--vessels.jpg')
        masks_getfname_fcns[LoaderGeneric.VESSEL_CLASS] = lambda x : x.replace('.JPG', '--artery.jpg')
        

        return  masks_getfname_fcns
    
    @property
    def preprocess_fcns(self):
        preprocess_fcns = dict()
        preprocess_fcns[LoaderGeneric.VESSEL] = lambda x : x[:,:,0] < 128
        preprocess_fcns[LoaderGeneric.VESSEL_CLASS] = lambda x : x    

        return preprocess_fcns
    
    @property 
    def read_img_fcn(self):
        
        def read_artery_vein(name):
            if '--artery' in name:
                if os.path.exists(name):
                    artery = imread(name)
                elif os.path.exists(tmp_name := name.replace('--artery.jpg', '--arteries.jpg')):
                    artery = imread(tmp_name)
                elif os.path.exists(tmp_name := name.replace('--artery.jpg', '--atertries.jpg')):
                    artery = imread(tmp_name)
                else:
                    raise(Exception(name + 'art'))
                    
                if os.path.exists(tmp_name := name.replace('--artery.jpg', '--veins.jpg')):
                    veins = imread(tmp_name)
                elif os.path.exists(tmp_name := name.replace('--artery.jpg', '--vein.jpg')):
                    veins = imread(tmp_name)  
                elif os.path.exists(tmp_name := name.replace('--artery.jpg', '--veinds.jpg')):
                    veins = imread(tmp_name) 
                elif os.path.exists(tmp_name := name.replace('--artery.jpg', '--veisn.jpg')):
                    veins = imread(tmp_name) 
                else:
                    raise(Exception(name + 'vein'))
                    
                return np.stack((artery[:,:,2] < 128, veins[:,:,0] < 128, np.zeros_like(veins[:,:,0])), axis=2)
                                           
            else:
                return imread(name)
        
        return read_artery_vein
    
                                

    def get_fov(self, fname_img, img):
        
        fov = self.get_fov_auto(img, t1=7, t2=15)
        return fov
    
    @property
    def dataset_fov_deg(self):
        return 30
    
    def get_savename(self, fname_img):
        return 'avrdb_na_na_'
        
        
        
        
    
if __name__ == "__main__":
    
    data_path = '../../databases'
    pix_per_deg = 25
    preprocess_f = lambda img, fov, pix_per_deg : img
    # preprocess_f = local_contrast_and_clahe
    out_fname = '../../data_' + str(pix_per_deg) + '.hdf5'
    
    if os.path.exists(out_fname):
        os.remove(out_fname)

    loader = LoaderAvrdb(data_path, pix_per_deg, out_fname, preprocess_f)
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


