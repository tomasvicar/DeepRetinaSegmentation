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

import sys
sys.path.append("..")
from utils.local_contrast_and_clahe import local_contrast_and_clahe


class LoaderDrishtigs(LoaderGeneric):
    
    @property
    def fnames_imgs(self):
        
        
        return [x for x in glob(self.data_path + '/Drishti-GS/**/*.png', recursive=True) if 'SoftMap' not in x]
        
    
    @property
    def masks_getfname_fcns(self):
        
        masks_getfname_fcns = dict()
        
        def disk_name(x):
            to_search = os.path.split(x)[0].replace('Images', '') + '/**/' + os.path.split(x)[1].replace('.png', '_ODsegSoftmap.png')
            
            return glob(to_search, recursive=True)[0]
            
        masks_getfname_fcns[LoaderGeneric.DISK] = disk_name
        
        
        def cup_name(x):
            to_search = os.path.split(x)[0].replace('Images', '') + '/**/' + os.path.split(x)[1].replace('.png', '_cupsegSoftmap.png')
            
            return glob(to_search, recursive=True)[0]
            
        
        masks_getfname_fcns[LoaderGeneric.CUP] = cup_name
        
        return  masks_getfname_fcns
    
    @property
    def preprocess_fcns(self):
        preprocess_fcns = dict()
        preprocess_fcns[LoaderGeneric.DISK] = lambda x : x
        preprocess_fcns[LoaderGeneric.CUP] = lambda x : x
        
        return preprocess_fcns
    
    
                                

    def get_fov(self, fname_img, img):
        
        fov = self.get_fov_auto(img, t1=7, t2=15)
        
        return fov
    
    
    @property
    def dataset_fov_deg(self):
        return 30
    
    def get_savename(self, fname_img):
        
        table = pd.read_excel(self.data_path + "/Drishti-GS/Drishti-GS1_diagnosis.xlsx")
        
        names = table.iloc[:, 1].tolist()
        diagnoses = table.iloc[:, 8].tolist()
        
        
        for name_tmp, diagnose_tmp in zip(names, diagnoses):
            if isinstance(name_tmp, str):
                if name_tmp[:-1] in fname_img:
                    diagnose = diagnose_tmp
                
        if diagnose == 'Glaucomatous':
            diagnose = 'glaucoma'
        if diagnose == 'Normal':
            diagnose = 'normal'    
            
        
        if 'Training' in fname_img:
            return 'drishtigs_train_' + diagnose + '_'
        elif 'Test' in fname_img:
            return 'drishtigs_test_'  + diagnose + '_'
        else:
            raise(Exception('neni'))
            
        
        
        
        
    
if __name__ == "__main__":
    
    data_path = '../../databases'
    pix_per_deg = 25
    preprocess_f = lambda img, fov, pix_per_deg : img
    # preprocess_f = local_contrast_and_clahe
    out_fname = '../../data_' + str(pix_per_deg) + '.hdf5'
    
    if os.path.exists(out_fname):
        os.remove(out_fname)

    loader = LoaderDrishtigs(data_path, pix_per_deg, out_fname, preprocess_f)
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


