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
    



class Wrapper(object):
    def __init__(self,iter_init=0):
        self.iter= iter_init

    def __call__(self, **params_in):
        
        config = Config()
        
        
        for key in list(params_in.keys()):
            
            config[key] = params_in[key]
            
            setattr(config,key,params_in[key]) 

        self.iter = self.iter + 1
        return train_one_model(config, self.iter)









if __name__ == "__main__":
    


    logging.basicConfig(filename='debug.log',level=logging.INFO)
    try:
        pass
      

        
    except Exception as e:
        logging.critical(e, exc_info=True)