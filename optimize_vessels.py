
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
    
    data_split = DataSpliter.split_data(data_type=DataSpliter.DATA_TYPE_VESSELS)

    config.method = config.main_name + '_' + str(it)
    
    model_name = train(config,data_split=data_split)
    
    accs,aucs,dices,tps,fps,fns,tns = valid_fcn_vessels('../' + config.method + '/valid_results', config, model_name, data_split['valid'])
    
    return np.mean(dices)



    



class Wrapper(object):
    def __init__(self, config, iter_init=0):
        self.iter= iter_init
        self.config = config

    def __call__(self, **params_in):
        
        config = self.config
        
        
        for key in list(params_in.keys()):
            
            
            setattr(config,key,params_in[key]) 

        
        config.init_lr = 10**(-config.init_lr)
            
        config.patch_size = int(32* np.round(config.patch_size))
        
        config.filters = int(np.round(config.filters))
        
        config.depth = int(np.round(config.depth))
        
        
        if config.rotate>0.5:
            config.rotate = True
        else:
            config.rotate = False
        
            

        self.iter = self.iter + 1
        return train_one_model(config, self.iter)









if __name__ == "__main__":
    


    pbounds = {'init_lr':[2,4], 
               'patch_size':[3,8],
               'filters':[16,80],
               'scale_deform':[0,0.4],
               'shear_deform':[0,0.3],
               'rotate':[0,1],
               'multipy':[0,0.3],
               'add':[0,0.3],
               'sharp_blur':[0,2],
               'depth':[2,5],
                }
    
    
    config = Config()
    
    config.data_path = sys.argv[1] + os.sep  + 'retina_segmentation_databases_25_gauss_clahe.hdf5'
    config.model_save_dir = sys.argv[2] + os.sep  + 'tmp'
    config.best_models_dir = sys.argv[2] + os.sep  + 'best_models'
    config.main_name = sys.argv[3] 
    
    
    if not os.path.isdir(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.isdir(config.best_models_dir):
        os.makedirs(config.best_models_dir) 
        
    final_dir = sys.argv[2] + os.sep  + 'final_dir'
    
    optimizer_bayes = BayesianOptimization(f=Wrapper(config),pbounds=pbounds,random_state=0)
    
    logger_bayes = JSONLogger(path= final_dir + os.sep +'/opt' + config.main_name + '.json')
    optimizer_bayes.subscribe(Events.OPTIMIZATION_STEP, logger_bayes)
  
    optimizer_bayes.maximize(init_points=config.init_points,n_iter=config.n_iter)
    
    best_iter = np.argmax([x['target'] for x in optimizer_bayes.res])
    best_name = config.best_models_dir + os.sep + config.main_name + '_' + str(best_iter)
    best_name = glob(best_name + '*.pt')[0]
    
    
    shutil.copy(best_name, final_dir)
    with open(final_dir + os.sep + config.main_name + '_valid_res.json', 'w') as json_file:
        json.dump(optimizer_bayes.max, json_file)
    
    
    
    
    

