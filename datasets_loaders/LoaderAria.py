import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import h5py

from LoaderGeneric import LoaderGeneric

from skimage.morphology import binary_closing, disk
from scipy.ndimage import binary_fill_holes
from skimage.measure import label

import sys
sys.path.append("..")
from utils.local_contrast_and_clahe import local_contrast_and_clahe


class LoaderAria(LoaderGeneric):
    
    @property
    def fnames_imgs(self):
        return glob(self.data_path + '/ARIA/aria_*_markups/aria_*_markups/*.tif')
        
    
    @property
    def masks_getfname_fcns(self):
        
        masks_getfname_fcns = dict()
        masks_getfname_fcns[LoaderGeneric.VESSEL] = lambda x : x.replace('_markups', '_markup_vessel').replace('.tif', '_BDP.tif')
        masks_getfname_fcns[LoaderGeneric.VESSEL2] = lambda x : x.replace('_markups', '_markup_vessel').replace('.tif', '_BSS.tif')
        
        def get_name_disk(x):
            if  'aria_a_markup' not in x:
                xx = x.replace('_markups', '_markupdiscfovea').replace('.tif', '_dfs.tif')
                if os.path.isfile(xx):
                    return xx
                xx = x.replace('_markups', '_markupdiscfovea').replace('.tif', '.tif')
                if os.path.isfile(xx):
                    return xx
                xx = x.replace('_markups', '_markupdiscfovea').replace('.tif', '_dfd.tif')
                if os.path.isfile(xx):
                    return xx
                raise(Exception('mask not found'))
                
            else:
                return None 
        masks_getfname_fcns[LoaderGeneric.DISK] = get_name_disk 
        
        return  masks_getfname_fcns
    
    @property
    def preprocess_fcns(self):
        preprocess_fcns = dict()
        preprocess_fcns[LoaderGeneric.VESSEL] = lambda x : x
        preprocess_fcns[LoaderGeneric.VESSEL2] = lambda x : x
        
        def preprocess_disk(x):
            x = binary_closing(x > 0, disk(21))
            x = binary_fill_holes(x)
            
            x = label(x)
            largest_x = np.bincount(x.flat)[1:].argmax() + 1
            x = x == largest_x
            return x
        preprocess_fcns[LoaderGeneric.DISK] = preprocess_disk
        
        return preprocess_fcns
                                

    def get_fov(self, fname_img, img):
        
        fov = self.get_fov_auto(img, t1=1, t2=15)
        if np.sum(fov == 0) < (0.05 * fov.size):
            fov = self.get_fov_auto(img, t1=58, t2=80)
            
        return fov
    
    @property
    def dataset_fov_deg(self):
        return 50
    
    def get_savename(self, fname_img):
        if 'aria_d_markup' in fname_img:
            return 'aria_na_diabetes_'
        elif 'aria_a_markup' in fname_img:
            return 'aria_na_amd_'
        elif 'aria_c_markup' in fname_img:
            return 'aria_na_healthy_'
        else:
            raise(Exception('not any selection'))
        
        
        
        
    
if __name__ == "__main__":
    
    data_path = '../../databases'
    pix_per_deg = 25
    preprocess_f = lambda img, fov, pix_per_deg : img
    # preprocess_f = local_contrast_and_clahe
    out_fname = '../../data_' + str(pix_per_deg) + '.hdf5'
    
    if os.path.exists(out_fname):
        os.remove(out_fname)

    loader = LoaderAria(data_path, pix_per_deg, out_fname, preprocess_f)
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


