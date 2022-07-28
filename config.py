import numpy as np
import os

class Config:
    
    
    model_save_dir='../tmp'
    
    best_models_dir='../best_models'
    
    method = 'first_try'
    
    main_name = 'norm_all'
    
    
    data_path = '../retina_segmentation_databases_25_gauss_clahe.hdf5'
    
    
    results_folder = '../results'
    
    
    init_points = 5
    n_iter = 20
    
    split_ratio_train_valid_test=[8,2,0] 
    

    train_batch_size = 32
    train_num_workers = 8
    valid_batch_size = 8
    valid_num_workers = 2
    

    # train_batch_size = 4
    # train_num_workers = 0
    # valid_batch_size = 2
    # valid_num_workers = 0


    multiply_dataset = 100
    

    init_lr = 1e-3
    lr_changes_list = np.cumsum([40,20,10,5])
    # lr_changes_list = np.cumsum([1,1])

    
    gamma = 0.1
    max_epochs = lr_changes_list[-1]
    
    

    device ='cuda:0'
    
    
    patch_size = 128
    
    
    
    p = 0.8

    deformation = True
    scale_deform = 0.25
    shear_deform = 0.1
    rotate = True

    multipy = 0.1
    add = 0.1
    
    sharp = 0.5
    blur = 0.5
    


    filters = 64
    depth = 5
    
    weight_decay = 1e-6
    

    Gauss_and_Clahe = True
    data_use = 'all'