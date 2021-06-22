import numpy as np

class Config:
    
    model_save_dir='../tmp'
    
    best_models_dir='../best_models'
    
    data_path ="../data_preprocessed"
    
    results_folder = '../results'
    
    
    split_ratio_train_valid_test=[7.5,0.5,2] ##test only if test set does not exist
    

    split_ratio_pretrain_train_valid = [9.5,0.5]
    
    
    # train_batch_size = 16
    # train_num_workers = 8
    # valid_batch_size = 4
    # valid_num_workers = 2
    
    train_batch_size = 4
    train_num_workers = 0
    valid_batch_size = 2
    valid_num_workers = 0


    init_lr = 0.01
    lr_changes_list = np.cumsum([40,20,10,5])
    gamma = 0.1
    max_epochs = lr_changes_list[-1]
    
    

    filters=list((np.array([64,128,256,512,1024])/4).astype(np.int))
    in_size=1
    out_size=1
    
    
    device='cuda:0'
    
    
    patch_size=256
    
    
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
    border_width = None















