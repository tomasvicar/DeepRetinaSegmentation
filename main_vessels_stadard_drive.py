import logging
from shutil import rmtree
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import sys

from split_data import DataSpliter
from train import train
from config import Config
from test_fcn_vessels import test_fcn_vessels
from measure_mean_std import measure_mean_std

if __name__ == "__main__":
    
    
    
    resutls_separate = dict()
    resutls_retrained = dict()
    resutls_universal = dict()
    
    # init_model = 'imagenet'
    # init_model = None
    
    if len(sys.argv)>1:
        output_folder = sys.argv[1]
    else:
        output_folder = '..'
    
    
    logging.basicConfig(filename=output_folder + '/debug.log', level=logging.INFO)
    try:
    # if True:
        # for init_model in ['none','imagenet','pretraining']:
        for init_model in ['none','imagenet','pretraining']:
            
            
            config = Config()
            
            config.model_save_dir = output_folder + '/tmp'
    
            config.best_models_dir = output_folder + '/best_models'
            
            config.results_folder = output_folder + '/results'
            
            
            
            data_split = DataSpliter.split_data(data_type=DataSpliter.DATA_TYPE_VESSELS,seed=0*100)
            data_split['database_names'] = ['drive']
            database_name = 'drive'
            
            
            tmp_train = DataSpliter.filter_database(data_split['train'],database_name)
            tmp_valid = DataSpliter.filter_database(data_split['valid'],database_name)
            tmp_test = DataSpliter.filter_database(data_split['test'],database_name)
            
            mean,std = measure_mean_std(config,tmp_train)
            config.pretrain_mean = mean 
            config.pretrain_std = std
                
            
                
            resutls_separate[database_name] = {'ACC':[],'AUC':[],'DICE':[],'TP':[],'FP':[],'FN':[],'TN':[],'ACC_mean':[],'AUC_mean':[]}
                    
                    
            
            if os.path.isdir(config.model_save_dir):
                rmtree(config.model_save_dir) 
                
                
            if not os.path.isdir(config.best_models_dir):
                os.mkdir(config.best_models_dir)
                
            if not os.path.isdir(config.model_save_dir):
                os.mkdir(config.model_save_dir)
                
            if not os.path.isdir(config.results_folder):
                os.mkdir(config.results_folder)
            
            
            if init_model == 'pretraining':
                
                config.method = 'pretraining'
                config.model_name_load = 'imagenet'
                config.multiply_dataset = 1
                
    
                init_modelx = train(config,data_train=data_split['pretrain_train'],data_valid=data_split['pretrain_valid'])
                
                
            else:
                init_modelx = init_model


                
            

            
            config.method = 'segmentation_separate' + database_name
            config.model_name_load = init_modelx
            config.multiply_dataset = 100
            
            # model_name = '../best_models/segmentation_separatedrive_8_0.00001_gpu_3.05870_train_0.09566_valid_0.12364.pt'
            model_name = train(config,data_train=tmp_train,data_valid=tmp_valid)
            accs,aucs,dices,tps,fps,fns,tns = test_fcn_vessels(config.results_folder + '/' + config.method + '/' + database_name + init_model, config, model_name, tmp_test)
            
            resutls_separate[database_name]['ACC'].append(accs)
            resutls_separate[database_name]['AUC'].append(aucs)
            resutls_separate[database_name]['DICE'].append(dices)
            resutls_separate[database_name]['TP'].append(tps)
            resutls_separate[database_name]['FP'].append(fps)
            resutls_separate[database_name]['FN'].append(fns)
            resutls_separate[database_name]['TN'].append(tns)
            resutls_separate[database_name]['ACC_mean'].append(np.mean(accs))
            resutls_separate[database_name]['AUC_mean'].append(np.mean(aucs))
                
                
            
            results = dict()
            results['resutls_separate'] = resutls_separate
            with open(config.results_folder + '/result_mult100_size128'  + init_model + '.json', 'w') as outfile:
                json.dump(results, outfile)    
            
          

        
    except Exception as e:
        logging.critical(e, exc_info=True)