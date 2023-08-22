import numpy as np
import os

from DataSpliter import DataSpliter

class Config:
    
    
    dataset_fname = '../data_30.hdf5'
    
    model_save_dir = '../tmp'
    best_models_dir = '../best_models'
    
    seed = 42
    
    
    train_batch_size = 8
    train_num_workers = 8
    valid_batch_size = 4
    valid_num_workers = 2
    
    
    train_valid_test_frac = [0.7, 0.1, 0.2]
    
    
    init_lr = 1e-3
    lr_changes_list = np.cumsum([100,30,10,10])
    gamma = 0.1
    
    
    max_epochs = lr_changes_list[-1]
    
    patch_size = 32 * 8
    
    
    deformation = True
    scale_deform = 0.25
    shear_deform = 0.1
    rotate = True
    multipy = 0.2
    add = 0.1
    sharp = 0.5
    blur = 0.5
    p=0.
    
    weight_decay = 1e-5
    
    
    mask_type_use = [DataSpliter.VESSEL, DataSpliter.DISK, DataSpliter.CUP]
    
    device ='cuda:0'
    
    
   
    filters = 32
    drop_out = 0
    depth = 4
    
    
    
    
    
    
    
    