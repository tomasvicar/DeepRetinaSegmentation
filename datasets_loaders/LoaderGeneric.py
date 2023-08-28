from abc import ABC, abstractmethod
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.filters import apply_hysteresis_threshold
import numpy as np
from skimage.morphology import binary_closing, disk, binary_erosion
from skimage.measure import label
from scipy.ndimage import binary_fill_holes
from skimage import io, color, filters, morphology, measure, feature
import h5py
from skimage.transform import rescale, resize

import sys
sys.path.append("..")

from utils.visboundaries import visboundaries
from utils.image_utils import get_bbox, crop_to_bbox

class LoaderGeneric(ABC):
    
    VESSEL = 'vessels'
    VESSEL2 = 'vessels2'
    VESSEL3 = 'vessels3'
    VESSEL_CLASS = 'ves_class'
    VESSEL_CLASS2 = 'ves_class2'
    DISK = 'disk'
    DISK2 = 'disk2'
    DISK3 = 'disk3'
    DISK4 = 'disk4'
    DISK5 = 'disk5'
    DISK6 = 'disk6'
    CUP = 'cup'
    CUP2 = 'cup2'
    CUP3 = 'cup3'
    CUP4 = 'cup4'
    CUP5 = 'cup5'
    CUP6 = 'cup6'
    
    def __init__(self, data_path, pix_per_deg, out_fname, preprocess_f):
        self.data_path = data_path
        self.pix_per_deg = pix_per_deg
        self.out_fname = out_fname
        self.preprocess_f = preprocess_f

        
        
    def preprocess(self, show_fov=False, show_masks=False):
        
        with h5py.File(self.out_fname, "a") as file:
        
            
            
            fnames_imgs = self.fnames_imgs
            # fnames_imgs = self.fnames_imgs[800:] ###########################################!!!!!!
            
            
            for fname_num, fname_img in enumerate(fnames_imgs): 
                
                print(str(fname_num) + ' / ' + str(len(fnames_imgs)))
                
                img = self.read_img_fcn(fname_img)
                img = self.prepare_img(img)
                fov = self.get_fov(fname_img, img)
                
                fov = fov > 0
                
                bbox = get_bbox(fov)
                rescale_factor = self.pix_per_deg * self.dataset_fov_deg / np.max(img.shape) 
                orig_size = img.shape[:2]
                
                img = crop_to_bbox(img, bbox)
                img = rescale(img, rescale_factor, channel_axis=2, preserve_range=True).astype(np.uint8)
                fov = crop_to_bbox(fov, bbox)
                fov = rescale(fov, rescale_factor, order=0)
                
                img = self.preprocess_f(img, fov, self.pix_per_deg)
                
                if show_fov:
                    print(fname_img)
                    plt.imshow(img)
                    visboundaries(fov)
                    plt.show()
                
                
                savename = self.get_savename(fname_img) + str(fname_num).zfill(4)
                original_filename = fname_img.replace(self.data_path, '')
                
                
                self.save_hdf5(file, img, savename + '/img', original_filename, bbox, orig_size, rescale_factor)
                self.save_hdf5(file, fov, savename + '/fov', original_filename, bbox, orig_size, rescale_factor)
                
                for mask_num, mask_name_key in  enumerate(self.masks_getfname_fcns):
                    
                    mask_name = self.masks_getfname_fcns[mask_name_key](fname_img)
                    if mask_name == None:
                        continue
                    preprocess_fcn = self.preprocess_fcns[mask_name_key]
                    
                    mask = self.read_img_fcn(mask_name)
                    mask = preprocess_fcn(mask)
                    
                    mask = mask > 0
                    
                    mask = crop_to_bbox(mask, bbox)
                    if len(mask.shape) == 3:
                        mask = rescale(mask, rescale_factor, channel_axis=2, order=0)
                    else:
                        mask = rescale(mask, rescale_factor, order=0)
                    
                    original_filename_mask = mask_name.replace(self.data_path, '')
                    
                    self.save_hdf5(file, mask, savename + '/' + mask_name_key, original_filename_mask, bbox, orig_size, rescale_factor)
                    
                    if show_masks:
                        print(mask_name)
                        plt.imshow(img)
                        if len(mask.shape) > 2:
                            mask = mask[:, :, 0]
                        visboundaries(mask)
                        plt.show()
                    
    def prepare_img(self, img):
        return img           
                
                
    def save_hdf5(self, file, data, name, orig_name, crop_position, orig_size, rescale_factor):
        
        chunks = [128,128]
        if len(data.shape) > 2:
            chunks.append(data.shape[2])
        chunks = tuple(chunks)    
        
        dts = file.create_dataset(name, data=data, chunks=chunks, compression="gzip", compression_opts=2)
        dts.attrs['orig_name'] = orig_name
        dts.attrs['crop_position'] = crop_position
        dts.attrs['orig_size'] = orig_size
        dts.attrs['rescale_factor'] = rescale_factor

    @property 
    def read_img_fcn(self):
        return imread
                
    @abstractmethod
    def get_savename(self, fname_img):
        '''
        name to save
        '''
        pass             
                
        
    @abstractmethod
    def get_fov(self, fname_img):
        '''
        fucntion that will get fov
        '''
        pass
    
    @staticmethod
    def get_fov_auto(img, t1=7, t2=15):
        
        averageRGB = np.mean(img,axis=2)
        
        mask = apply_hysteresis_threshold(averageRGB,t1,t2)
        
        mask = binary_closing(mask, disk(10))
        
        mask = binary_erosion(mask, disk(6))
        
        mask = label(mask)
        largest_mask = np.bincount(mask.flat)[1:].argmax() + 1
        mask = mask == largest_mask
        
        mask = binary_fill_holes(mask)
        
        return mask
    
    
    
    @property
    @abstractmethod
    def dataset_fov_deg(self):
        '''
        value of degrees of fov for this dataset
        '''
        pass
    
        
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
    
    
        