import numpy as np
import os

from DataSpliter import DataSpliter

class Config:
    
    
    dataset_fname = '../data_25.hdf5'
    
    
    # method = 'test'
    # method = 'test_vessels_seg'
    method = 'test_disk'
    
    tmp_folder = '../_method_/tmp'
    best_model_folder = '../_method_/best_models'
    
    
    
    seed = 42
    
    
    train_batch_size = 32
    valid_batch_size = 8
    # train_batch_size = 8
    # valid_batch_size = 4
    
    train_num_workers = 8
    valid_num_workers = 2
    # train_num_workers = 0
    # valid_num_workers = 0
    
    
    
    train_valid_test_frac = [0.7, 0.1, 0.2]
    
    
    init_lr = 1e-3
    lr_changes_list = np.cumsum([100,25,10,10])
    # lr_changes_list = np.cumsum([1,1])
    gamma = 0.1
    
    
    max_epochs = lr_changes_list[-1]
    
    patch_size = 32 * 8
    
    weight_decay = 1e-6
    
    
    # mask_type_use = [DataSpliter.VESSEL, DataSpliter.DISK, DataSpliter.CUP, DataSpliter.VESSEL_CLASS]
    # mask_type_use = [DataSpliter.VESSEL, DataSpliter.DISK, DataSpliter.CUP]
    # mask_type_use = [DataSpliter.VESSEL,]
    mask_type_use = [DataSpliter.DISK,]
    
    device ='cuda:0'
    
    
   
    filters = 32
    drop_out = 0
    depth = 4
    
    p = 0.3
    multipy = 0.2
    add = 0.2
    
    
    
    
    
    
    
    