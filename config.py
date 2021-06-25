import numpy as np
import os

class Config:
    
    model_save_dir='../tmp'
    
    best_models_dir='../best_models'
    
    
    if os.path.isdir("../data_preprocessed"):
        data_path = "../data_preprocessed" 
    elif os.path.isdir("../../data_preprocessed"): 
        data_path = "../../data_preprocessed"
    else:
        raise Exception('no data')
        
    
    results_folder = '../results'
    
    
    split_ratio_train_valid_test=[7.5,0.5,2] ##test only if test set does not exist
    

    split_ratio_pretrain_train_valid = [9.5,0.5]
    
    
    train_batch_size = 16
    train_num_workers = 8
    valid_batch_size = 4
    valid_num_workers = 2
    
    # train_batch_size = 4
    # train_num_workers = 0
    # valid_batch_size = 2
    # valid_num_workers = 0


    init_lr = 0.001
    lr_changes_list = np.cumsum([60,30,15,7])
    gamma = 0.1
    max_epochs = lr_changes_list[-1]
    
    loss = 'bce'
    # loss = 'dice_loss'

    device ='cuda:0'
    
    
    patch_size = 384
    clahe_grid = 16 #### for 384 patch
    clahe_clip = 5 #### for 384 patch
    
    
    net_name = "efficientnet-b0"
    in_channels = 1
    deformation = True
    scale_deform = 0.2
    shear_deform = 0.05
    multipy = 0.2
    add = 0.1
    p=0.7
    clahe = True
    local_normalization = False
    
    # img_type = 'rgb'
    img_type = 'green'
    # img_type = 'gray'
    
    
    
    pretrain_num_blocks = 0
    pretrain_max_block_size = 20
    pretrain_mean = -0.17507726083630085
    pretrain_std = 0.24185812541237633
    pretrain_noise_std_fraction = None
    pretrain_noise_pixel_p = None
    pretrain_noise_pixel_std_fraction = 5
    pretrain_chessboard_num_blocks = 0
    pretrain_chessboard_max_block_size = 20
    pretrain_rot_num_blocks = 0
    pretrain_rot_max_block_size = 20
    
    
    model_name_load = None
    method = None















