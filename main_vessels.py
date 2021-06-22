import logging
from shutil import rmtree
import os
import numpy as np

from split_data import DataSpliter
from train import train
from config import Config
from test import test


if __name__ == "__main__":
    
    # logging.basicConfig(filename='debug.log',level=logging.INFO)
    # try:
    if True:
        
        config = Config()
        
        data_split = DataSpliter.split_data(data_type=DataSpliter.DATA_TYPE_VESSELS,seed=42)
        
        
        if os.path.isdir(config.model_save_dir):
            rmtree(config.model_save_dir) 
            
            
        if not os.path.isdir(config.best_models_dir):
            os.mkdir(config.best_models_dir)
            
        if not os.path.isdir(config.model_save_dir):
            os.mkdir(config.model_save_dir)
            
        if not os.path.isdir(config.results_folder):
            os.mkdir(config.results_folder)
        
        
        
        
        config.method = 'segmentation'
        config.model_name_load ='imagenet'
        config.lr_changes_list = np.cumsum([2,2,2])
        max_epochs = config.lr_changes_list[-1]
        
        universal_model_name = train(config,data_train=data_split['train'],data_valid=data_split['valid'])
        
        dice = test('../outputs', config, model_name=universal_model_name,data=data_split['test'])
        
        # for database_name in data_split['database_names']:
            
        #     tmp_train = DataSpliter.filter_database(data_split['train'],database_name)
        #     tmp_valid = DataSpliter.filter_database(data_split['valid'],database_name)
        #     tmp_test = DataSpliter.filter_database(data_split['test'],database_name)
            
            
        
        
      

        
    # except Exception as e:
    #     logging.critical(e, exc_info=True)