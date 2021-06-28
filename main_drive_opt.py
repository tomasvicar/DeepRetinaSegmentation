import logging
from shutil import rmtree
import os
import numpy as np
import matplotlib.pyplot as plt
import json


from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt import BayesianOptimization


from split_data import DataSpliter
from train import train
from config import Config
from test_fcn_vessels import test_fcn_vessels




def train_one_model(config,it):
    
    data_split = DataSpliter.split_data(data_type=DataSpliter.DATA_TYPE_VESSELS,seed=0*100)
    data_split['database_names'] = ['drive']
    
    if os.path.isdir(config.model_save_dir):
                rmtree(config.model_save_dir) 
                
                
    if not os.path.isdir(config.best_models_dir):
        os.mkdir(config.best_models_dir)
        
    if not os.path.isdir(config.model_save_dir):
        os.mkdir(config.model_save_dir)
        
    if not os.path.isdir(config.results_folder):
        os.mkdir(config.results_folder)


    database_name = data_split['database_names'][0]
                
    tmp_train = DataSpliter.filter_database(data_split['train'],database_name)
    tmp_valid = DataSpliter.filter_database(data_split['valid'],database_name)
    tmp_test = DataSpliter.filter_database(data_split['test'],database_name)

    init_model = None
    config.method = 'segmentation_separate' + database_name + str(it)
    config.model_name_load = init_model
    
    
    model_name = train(config,data_train=tmp_train,data_valid=tmp_valid)
    accs,aucs,dices,tps,fps,fns,tns = test_fcn_vessels('../' + config.method + '/' + database_name + str(it), config, model_name, tmp_test)
    
    return np.mean(aucs)



# def train_one_model(config,it):
    
#     data_split = DataSpliter.split_data(data_type=DataSpliter.DATA_TYPE_VESSELS,seed=0*100)
#     data_split['database_names'] = ['drive']
    
#     if os.path.isdir(config.model_save_dir):
#                 rmtree(config.model_save_dir) 
                
                
#     if not os.path.isdir(config.best_models_dir):
#         os.mkdir(config.best_models_dir)
        
#     if not os.path.isdir(config.model_save_dir):
#         os.mkdir(config.model_save_dir)
        
#     if not os.path.isdir(config.results_folder):
#         os.mkdir(config.results_folder)
        
#     database_name = data_split['database_names'][0]    
        
#     config.method = 'segmentation_pretrain' + database_name + str(it)
#     config.model_name_load = None
#     universal_model_name = train(config,data_train=data_split['train'],data_valid=data_split['valid'])
    

#     tmp_train = DataSpliter.filter_database(data_split['train'],database_name)
#     tmp_valid = DataSpliter.filter_database(data_split['valid'],database_name)
#     tmp_test = DataSpliter.filter_database(data_split['test'],database_name)

#     # init_model = None
#     config.method = 'segmentation_retrain' + database_name + str(it)
#     config.model_name_load = universal_model_name
    
    
#     model_name = train(config,data_train=tmp_train,data_valid=tmp_valid)
#     accs,aucs,dices,tps,fps,fns,tns = test_fcn_vessels('../' + config.method + '/' + database_name + str(it), config, model_name, tmp_test)
    
#     return np.mean(aucs)



    



class Wrapper(object):
    def __init__(self,iter_init=0):
        self.iter= iter_init

    def __call__(self, **params_in):
        
        config = Config()
        
        
        for key in list(params_in.keys()):
            
            
            setattr(config,key,params_in[key]) 

        
        config.init_lr = 10**(-config.init_lr)
        
        config.lr_changes_list = np.cumsum(np.round(config.lr_changes_list*1/2**(np.arange(4))).astype(np.int32))
        config.max_epochs = config.lr_changes_list[-1]
        
        # if config.loss>0.5:
        #     config.loss = 'bce'
        # else:
        #     config.loss = 'dice_loss'
            
        config.patch_size = int(32* np.round(config.patch_size))
        
        config.filters = int(np.round(config.filters))
        
        config.depth = int(np.round(config.depth))
        
        config.weight_decay = 10**(-config.weight_decay)
        
        if config.rotate>0.5:
            config.rotate = True
        else:
            config.rotate = False
        
        
        config.img_type
        
        if config.img_type<1:
            config.img_type = 'rgb'
            config.in_channels = 3
        elif config.img_type<2:
            config.img_type = 'green'
            config.in_channels = 1
        else:
            config.img_type = 'gray' 
            config.in_channels = 1
            
        if config.clahe>0.5:
            config.clahe = True
        else:
            config.clahe = False 

        self.iter = self.iter + 1
        return train_one_model(config, self.iter)









if __name__ == "__main__":
    


    logging.basicConfig(filename='debug.log',level=logging.INFO)
    try:
    # if True:

        pbounds = {'init_lr':[1,5], 
                   'lr_changes_list':[10,50],
                   'patch_size':[1,8],
                   'filters':[16,64],
                   'drop_out':[0,0.5],
                   'weight_decay':[2,7],
                   'scale_deform':[0,0.5],
                   'shear_deform':[0,0.3],
                   'rotate':[0,1],
                   'multipy':[0,0.4],
                   'add':[0,0.3],
                   'sharp':[0,2],
                   'blur':[0,2],
                   'img_type':[0,3],
                   'depth':[2,5],
                   'clahe':[0,1]
                    }
        
        
        
        optimizer_bayes = BayesianOptimization(f=Wrapper(0),pbounds=pbounds,random_state=0)
        
        logger_bayes = JSONLogger(path= '../opt_drive.json')
        optimizer_bayes.subscribe(Events.OPTIMIZATION_STEP, logger_bayes)
      
        optimizer_bayes.maximize(init_points=2,n_iter=100)

        
    except Exception as e:
        logging.critical(e, exc_info=True)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        