import logging
from shutil import rmtree
import os
import numpy as np
import matplotlib.pyplot as plt
import json

from split_data import DataSpliter
from train import train
from config import Config
from test_fcn_vessels import test_fcn_vessels


if __name__ == "__main__":
    
    
    
    resutls_separate = dict()
    resutls_retrained = dict()
    resutls_universal = dict()
    
    init_model = None
    
    logging.basicConfig(filename='debug.log',level=logging.INFO)
    try:
    # if True:
        for cv_iter in range(1):
            
            config = Config()
            
            data_split = DataSpliter.split_data(data_type=DataSpliter.DATA_TYPE_VESSELS,seed=cv_iter*100)
            data_split['database_names'] = ['drive']
            
            if not resutls_separate:
                for database_name in data_split['database_names']:
                    resutls_separate[database_name] = {'ACC':[],'AUC':[],'DICE':[],'TP':[],'FP':[],'FN':[],'TN':[],}
                    resutls_retrained[database_name] = {'ACC':[],'AUC':[],'DICE':[],'TP':[],'FP':[],'FN':[],'TN':[],}
                    resutls_universal[database_name] = {'ACC':[],'AUC':[],'DICE':[],'TP':[],'FP':[],'FN':[],'TN':[],}
                    
                    
            
            if os.path.isdir(config.model_save_dir):
                rmtree(config.model_save_dir) 
                
                
            if not os.path.isdir(config.best_models_dir):
                os.mkdir(config.best_models_dir)
                
            if not os.path.isdir(config.model_save_dir):
                os.mkdir(config.model_save_dir)
                
            if not os.path.isdir(config.results_folder):
                os.mkdir(config.results_folder)
            
            

            
            config.method = 'segmentation_universal'
            config.model_name_load = init_model
            
            
            universal_model_name = train(config,data_train=data_split['train'],data_valid=data_split['valid'])
            
            
            
            for database_name in data_split['database_names']:
                
                tmp_train = DataSpliter.filter_database(data_split['train'],database_name)
                tmp_valid = DataSpliter.filter_database(data_split['valid'],database_name)
                tmp_test = DataSpliter.filter_database(data_split['test'],database_name)
                
                
                accs,aucs,dices,tps,fps,fns,tns = test_fcn_vessels('../' + config.method + '/' + database_name + str(cv_iter), config, universal_model_name, tmp_test)
                
                resutls_universal[database_name]['ACC'].append(accs)
                resutls_universal[database_name]['AUC'].append(aucs)
                resutls_universal[database_name]['DICE'].append(dices)
                resutls_universal[database_name]['TP'].append(tps)
                resutls_universal[database_name]['FP'].append(fps)
                resutls_universal[database_name]['FN'].append(fns)
                resutls_universal[database_name]['TN'].append(tns)
                
                
                config.method = 'segmentation_retrained' + database_name
                config.model_name_load = universal_model_name
                
                model_name = train(config,data_train=tmp_train,data_valid=tmp_valid)
                accs,aucs,dices,tps,fps,fns,tns = test_fcn_vessels('../' + config.method + '/' + database_name + str(cv_iter), config, model_name, tmp_test)
                
                resutls_retrained[database_name]['ACC'].append(accs)
                resutls_retrained[database_name]['AUC'].append(aucs)
                resutls_retrained[database_name]['DICE'].append(dices)
                resutls_retrained[database_name]['TP'].append(tps)
                resutls_retrained[database_name]['FP'].append(fps)
                resutls_retrained[database_name]['FN'].append(fns)
                resutls_retrained[database_name]['TN'].append(tns)
                
                
                config.method = 'segmentation_separate' + database_name
                config.model_name_load = init_model
                
                
                model_name = train(config,data_train=tmp_train,data_valid=tmp_valid)
                accs,aucs,dices,tps,fps,fns,tns = test_fcn_vessels('../' + config.method + '/' + database_name + str(cv_iter), config, model_name, tmp_test)
                
                resutls_separate[database_name]['ACC'].append(accs)
                resutls_separate[database_name]['AUC'].append(aucs)
                resutls_separate[database_name]['DICE'].append(dices)
                resutls_separate[database_name]['TP'].append(tps)
                resutls_separate[database_name]['FP'].append(fps)
                resutls_separate[database_name]['FN'].append(fns)
                resutls_separate[database_name]['TN'].append(tns)
                
                
            
            results = dict()
            results['resutls_universal'] = resutls_universal
            results['resutls_retrained'] = resutls_retrained
            results['resutls_separate'] = resutls_separate
            with open('../result_test3_bce_' + str(cv_iter) + '.json', 'w') as outfile:
                json.dump(results, outfile)    
        
      

        
    except Exception as e:
        logging.critical(e, exc_info=True)