import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from LoaderGeneric import LoaderGeneric

from skimage.morphology import binary_closing, disk
from scipy.ndimage.morphology import binary_fill_holes
from skimage.measure import label

class LoaderAria(LoaderGeneric):
    
    @property
    def fnames_imgs(self):
        return glob(self.data_path + '/ARIA/aria_*_markups/aria_*_markups/*.tif')
        
    
    @property
    def masks_getfname_fcns(self):
        
        masks_getfname_fcns = dict()
        masks_getfname_fcns['vessel'] = lambda x : x.replace('_markups', '_markup_vessel').replace('.tif', '_BDP.tif')
        masks_getfname_fcns['vessel2'] = lambda x : x.replace('_markups', '_markup_vessel').replace('.tif', '_BSS.tif')
        masks_getfname_fcns['disc'] = lambda x : x.replace('_markups', '_markupdiscfovea').replace('.tif', '_dfs.tif')
        return  masks_getfname_fcns
    
    @property
    def preprocess_fcns(self):
        preprocess_fcns = dict()
        preprocess_fcns['vessel'] = lambda x : x
        preprocess_fcns['vessel2'] = lambda x : x
        
        def preprocess_disk(x):
            x = binary_closing(x > 0, disk(21))
            x = binary_fill_holes(x)
            
            x = label(x)
            largest_x = np.bincount(x.flat)[1:].argmax() + 1
            x = x == largest_x
            return x
        
        preprocess_fcns['disc'] = preprocess_disk
        return preprocess_fcns
                                

    def get_fov(self, fname_img, img):
        
        fov = self.get_fov_auto(img, t1=1, t2=15)
        if np.sum(fov == 0) < (0.05 * fov.size):
            fov = self.get_fov_auto(img, t1=58, t2=80)
            
        return fov
        
        
    
if __name__ == "__main__":
    
    data_path = '../../databases'
    pix_per_deg = 25
    preprocess_f = lambda x : x
    out_fname = '../../data_' + str(pix_per_deg) 
    
    loader = LoaderAria(data_path, pix_per_deg, out_fname, preprocess_f)
    loader.preprocess()
    




