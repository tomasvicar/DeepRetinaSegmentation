import logging
from shutil import rmtree
import os
import numpy as np
import matplotlib.pyplot as plt

from split_data import DataSpliter
from train import train
from config import Config
from test_fcn_vessels import test_fcn_vessels


if __name__ == "__main__":
    
    
    
    resutls = dict()
    
    
    # logging.basicConfig(filename='debug.log',level=logging.INFO)
    # try:
    if True:
        for cv_iter in range(2):
            
            
            config = Config()
            
            data_split = DataSpliter.split_data(data_type=DataSpliter.DATA_TYPE_VESSELS,seed=cv_iter*100)
            
            if not resutls:
                for database_name in data_split['database_names']:
                    resutls[database_name] = {'ACC':[],
                                              'AUC':[],
                                              'DICE':[],
                                              'TP':[],
                                              'FP':[],
                                              'FN':[],
                                              'TN':[],
                                              }
            
            if os.path.isdir(config.model_save_dir):
                rmtree(config.model_save_dir) 
                
                
            if not os.path.isdir(config.best_models_dir):
                os.mkdir(config.best_models_dir)
                
            if not os.path.isdir(config.model_save_dir):
                os.mkdir(config.model_save_dir)
                
            if not os.path.isdir(config.results_folder):
                os.mkdir(config.results_folder)
            
            
            
            
            config.method = 'segmentation'
            config.model_name_load = None
            config.lr_changes_list = np.cumsum([2,2,2])
            config.max_epochs = config.lr_changes_list[-1]
            
            # universal_model_name = train(config,data_train=data_split['train'],data_valid=data_split['valid'])
            universal_model_name = '../best_models/segmentation_9_0.00001_gpu_0.00000_train_0.38325_valid_0.35431.pt'
            
            
            
            for database_name in data_split['database_names']:
                
                tmp_train = DataSpliter.filter_database(data_split['train'],database_name)
                tmp_valid = DataSpliter.filter_database(data_split['valid'],database_name)
                tmp_test = DataSpliter.filter_database(data_split['test'],database_name)
                
                
                accs,aucs,dices,tps,fps,fns,tns = test_fcn_vessels('../outputs_' + database_name , config, universal_model_name, tmp_test)
                
                resutls[database_name]['ACC'].append(accs)
                resutls[database_name]['AUC'].append(aucs)
                resutls[database_name]['DICE'].append(dices)
                resutls[database_name]['TP'].append(tps)
                resutls[database_name]['FP'].append(fps)
                resutls[database_name]['FN'].append(fns)
                resutls[database_name]['TN'].append(tns)
        
      

        
    # except Exception as e:
    #     logging.critical(e, exc_info=True)