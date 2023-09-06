from glob import glob
from bayes_opt import BayesianOptimization
import numpy as np
import copy
import shutil

from Config import Config
from train import train


class Wrapper(object):
    def __init__(self, config):
        self.config = config
        self.it = -1
        
    def __call__(self, **params):
        
        self.it = self.it + 1
        
        
        config = copy.deepcopy(self.config)
        config.method = config.method + '_' + str(self.it)
        config.init_lr = np.exp(1) ** params['init_lr']
        config.patch_size = 32 * int(params['patch_size_multiplier'])
        config.multipy = params['multipy']
        config.add = params['add']
        config.p = params['p']
        
        dice = train(config)
        return dice 
        

if __name__ == "__main__":
    
    opt_iters = 15
    opt_init_points = 3 
    
    
    pbounds = dict()
    pbounds['init_lr'] = [np.log(1e-4), np.log(1e-2)]
    pbounds['patch_size_multiplier'] = [6 , 10]
    pbounds['multipy'] = [0, 0.4]
    pbounds['add'] = [0, 0.4]
    pbounds['p'] = [0, 1]
    
            
    config = Config()       
    wrapper = Wrapper(config)
            
    optimizer = BayesianOptimization(f=wrapper,pbounds=pbounds,random_state=42)
    optimizer.maximize(init_points=opt_init_points,n_iter=opt_iters)    
    best_iter = np.argmax([x['target'] for x in optimizer.res])
    print('best iter:')
    print(best_iter)
        
    
    to_copy = '../' + config.method + '_' + str(best_iter)
    
    where_copy = '../' +  config.method + '_' + 'best'
    
    shutil.copytree(to_copy, where_copy)
    
    
    
    