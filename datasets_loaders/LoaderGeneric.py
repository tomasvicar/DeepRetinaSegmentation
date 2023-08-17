from abc import ABC, abstractmethod
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.filters import apply_hysteresis_threshold
import numpy as np
from skimage.morphology import binary_closing, disk
from skimage.measure import label
from scipy.ndimage.morphology import binary_fill_holes
from skimage import io, color, filters, morphology, measure, feature

import sys
sys.path.append("..")

from utils.visboundaries import visboundaries

class LoaderGeneric(ABC):
    
    def __init__(self, data_path, pix_per_deg, out_fname, preprocess_f):
        self.data_path = data_path
        self.pix_per_deg = pix_per_deg
        self.out_fname = out_fname
        self.preprocess_f = preprocess_f
        
        
    def preprocess(self):
        for fname_num, fname_img in enumerate(self.fnames_imgs):  
            
            img = imread(fname_img)
            fov = self.get_fov(fname_img, img)
            plt.imshow(img)
            visboundaries(fov)
            plt.show()
            
            # for mask_num, mask_name_key in  enumerate(self.masks_getfname_fcns):
            #     mask_name = self.masks_getfname_fcns[mask_name_key](fname_img)
            #     preprocess_fcn = self.preprocess_fcns[mask_name_key]
                
            #     print(mask_name)
            #     mask = imread(mask_name)
            #     mask = preprocess_fcn(mask)
                
                
        
    @abstractmethod
    def get_fov(self, fname_img):
        '''
        fucntion that will get fov
        '''
        pass
    
    
    def get_fov_auto(self, img, t1=7, t2=15):
        
        averageRGB = np.mean(img,axis=2)
        
        mask = apply_hysteresis_threshold(averageRGB,t1,t2)
        
        mask = binary_closing(mask, disk(10))
        
        mask = label(mask)
        largest_mask = np.bincount(mask.flat)[1:].argmax() + 1
        mask = mask == largest_mask
        
        mask = binary_fill_holes(mask)
        
        return mask
    
    
    
    
    
    
    def get_fov_auto_evca(self, im, database):
        
        im = color.rgb2gray(im)
        thresholds = filters.threshold_multiotsu(im, classes=4)
        prah1_dict = {
            'diaretdb': thresholds[0] * 0.1,
            'review': thresholds[0] * 0.1,
            'drishtigs': thresholds[2],
            # Add other databases and their conditions here...
        }
        prah1 = prah1_dict.get(database, 0)
        
        prah2 = im.max() * 0.4 if database == 'inspireavr' else im.max() * 0.8
        
        highmask = im > prah2
        lowmask = measure.label(im > prah1)
        unique_labels = np.unique(lowmask[highmask])
        final = np.isin(lowmask, unique_labels)
        se = morphology.disk(10)
        final_closed = morphology.binary_closing(final, se)

        # Further computations for `bpx`, `wpx`, etc.
        bpx = 1 - (np.sum(final_closed) / (im.shape[0] * im.shape[1]))
        wpx = 1 - bpx
        prah = 0.1  # default value
        
        if database == 'aria' or database == 'g1020':
            bpx = 0.1343  
            # You can add more conditions or variables specific to these databases here if needed
        
        elif database == 'diaretdb':
            bpx = 0.1634
        
        elif database == 'stare':
            bpx = 0.269
        
        elif database == 'drishtigs':
            bpx = 0.1568
            prah = 0.05
            if im.shape[1] > 2200:  # Assuming `im` is a numpy array, this gets its width
                bpx = 0.2929
        
        elif database == 'eophtha':
            bpx = 0.45
            prah = 0.15
        
        else:
            pass
        
        if np.sum(final_closed) / (im.shape[0] * im.shape[1]) > (wpx + prah) or np.sum(final_closed) / (im.shape[0] * im.shape[1]) < (wpx - prah):
            counts, edges = np.histogram(im, bins=250)
            cc = np.cumsum(counts)
            prah1 = edges[np.min(np.where(cc > bpx * (im.shape[0] * im.shape[1])))]
            prah2 = im.max() * 0.8
            highmask = im > prah2
            lowmask = measure.label(im > prah1)
            final = np.isin(lowmask, unique_labels)
            final_closed = morphology.binary_closing(final, se)

        fov = binary_fill_holes(final_closed)
        
        return fov
        
    
        
    @property
    @abstractmethod
    def masks_getfname_fcns(self):
        '''
        dict of functions how to get individual mask filenames for image filenames
        '''
        pass
    
    
    @property
    @abstractmethod
    def preprocess_fcns(self):
        '''
        dict of functions how to get individual preprocessed images from filenamse
        '''
        pass
    
    @property
    @abstractmethod
    def fnames_imgs(self):
        '''
        list of filenames of all images (get with glob)
        '''
        pass
    
    
        