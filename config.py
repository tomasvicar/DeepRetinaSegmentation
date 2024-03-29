import numpy as np
import os

class Config:
    
    model_save_dir='../tmp'
    
    best_models_dir='../best_models'
    
    
    if os.path.isdir("../data_preprocessed_hdf5"):
        data_path = "../data_preprocessed_hdf5" 
    elif os.path.isdir("../../data_preprocessed_hdf5"): 
        data_path = "../../data_preprocessed_hdf5"
    else:
        raise Exception('no data')
        
    # if os.path.isdir("../data_preprocessed_hdf5_12"):
    #     data_path = "../data_preprocessed_hdf5_12" 
    # elif os.path.isdir("../../data_preprocessed_hdf5_12"): 
    #     data_path = "../../data_preprocessed_hdf5_12"
    # else:
    #     raise Exception('no data')  
        
    
    results_folder = '../results'
    
    
    split_ratio_train_valid_test=[6.5,1.5,2] ##test only if test set does not exist
    

    split_ratio_pretrain_train_valid = [9.5,0.5]
    
    
    train_batch_size = 32
    train_num_workers = 8
    valid_batch_size = 8
    valid_num_workers = 2
    
    # train_batch_size = 32
    # train_num_workers = 12
    # valid_batch_size = 8
    # valid_num_workers = 4
    
    
    multiply_dataset = 100
    
    
    # train_batch_size = 4
    # train_num_workers = 0
    # valid_batch_size = 2
    # valid_num_workers = 0


    init_lr = 1e-3
    lr_changes_list = np.cumsum([30,10,5,5])
    # lr_changes_list = np.cumsum([2,1])
    # lr_changes_list = np.cumsum([5,2,2])
    
    gamma = 0.1
    max_epochs = lr_changes_list[-1]
    
    loss = 'bce'
    # loss = 'dice_loss'

    device ='cuda:0'
    
    
    # patch_size = 384
    patch_size = 32*8
    clahe = True

    clahe_grid = 4*8
    clahe_clip = 2
    
    
    
    deformation = True
    scale_deform = 0.25
    shear_deform = 0.1
    rotate = True
    multipy = 0.2
    add = 0.1
    sharp = 0.5
    blur = 0.5
    p=0.8
    
    
    
    net_name = "efficientnet-b2"
    
    # net_name = 'unet'
    # filters = 32
    # drop_out = 0
    # depth = 4
    
    weight_decay = 1e-5
    
    
    local_normalization = False
    
    # img_type = 'rgb'
    img_type = 'green'
    # img_type = 'gray'
    in_channels = 1
    
    
    pretrain_num_blocks = 5
    pretrain_max_block_size = 25
    pretrain_mean = -0.042632774 
    pretrain_std = 0.13752356
    pretrain_noise_std_fraction = 0
    pretrain_noise_pixel_p = 0.05
    pretrain_noise_pixel_std_fraction = 5
    
    
    model_name_load = None
    method = None















