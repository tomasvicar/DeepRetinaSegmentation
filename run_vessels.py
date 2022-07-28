

import numpy as np
import matplotlib.pyplot as plt
import json
import sys
import os
from glob import glob
import shutil


from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt import BayesianOptimization


from DataSpliter import DataSpliter
from train import train
from config import Config
from valid_fcn_vessels import valid_fcn_vessels




def train_one_model(config,it):
    
    data_split = DataSpliter.split_data(data_type=DataSpliter.DATA_TYPE_VESSELS,config=config)
    if config.data_use == 'hrf':
        data_split['train'] = DataSpliter.filter_database(data_split['train'],'hrf')
        data_split['valid'] = DataSpliter.filter_database(data_split['valid'],'hrf')
    

    config.method = config.main_name + '_' + str(it)
    
    model_name = train(config,data_split=data_split)
    
    accs,aucs,dices,tps,fps,fns,tns = valid_fcn_vessels('../' + config.method + '/valid_results', config, model_name, data_split['valid'])
    
    return np.mean(dices)






if __name__ == "__main__":
    

    
    config = Config()
    
    config.data_path = sys.argv[1] + os.sep  + 'retina_segmentation_databases_25_gauss_clahe.hdf5'
    config.model_save_dir = sys.argv[2] + os.sep  + 'tmp'
    config.best_models_dir = sys.argv[2] + os.sep  + 'best_models'
    config.main_name = sys.argv[3] 
    config.results_folder = sys.argv[2] + os.sep  + 'results'
    final_dir = sys.argv[2] + os.sep  + 'final_dir'
    
    if config.main_name == 'norm_all':
        config.Gauss_and_Clahe = True
        config.data_use = 'all'
    elif config.main_name == 'orig_all':
        config.Gauss_and_Clahe = False
        config.data_use = 'all'
    elif config.main_name == 'norm_hrf':
        config.Gauss_and_Clahe = True
        config.data_use = 'hrf'
        config.multiply_dataset = 300
    elif config.main_name == 'orig_hrf':
        config.Gauss_and_Clahe = False
        config.data_use = 'hrf'
        config.multiply_dataset = 300
    else:
        raise Exception('incorect setup')
    
    if not os.path.isdir(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.isdir(config.best_models_dir):
        os.makedirs(config.best_models_dir) 
    if not os.path.isdir(config.results_folder):
        os.makedirs(config.results_folder) 
    if not os.path.isdir(final_dir):
        os.makedirs(final_dir) 
    
    value = train_one_model(config, 0)
  
    best_iter = 0
    best_name = config.best_models_dir + os.sep + config.main_name + '_' + str(best_iter)
    best_name = glob(best_name + '*.pt')[0]
    
    
    
    shutil.copy(best_name, final_dir)
    with open(final_dir + os.sep + config.main_name + '_valid_res.json', 'w') as json_file:
        json.dump(value, json_file)
    
    