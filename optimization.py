from glob import glob
from bayes_opt import BayesianOptimization
import numpy as np


from config import Config
from train import train



opt_iters = 20
opt_init_points = 5


pbounds = dict()
pbounds['init_lr'] = [np.log(1e-4), np.log(1e-2)]
pbounds['patch_size_multiplier'] = [6 , 12]
pbounds['multipy'] = [0, 0.4]
pbounds['add'] = [0, 0.4]
pbounds['p'] = [0, 1]


class Wrapper(object):
    def __init__(self, config):
        self.config = config
        self.it = -1
        
    def __call__(self, **params):
        
        self.it = self.it + 1
        
        
        config = self.config.copy() 
        config.method = config.method + '_' + str(self.it)
        config.init_lr = np.exp(1) ** params['init_lr']
        config.patch_size = 32 * params['patch_size_multiplier']
        config.multipy = params['multipy']
        config.add = params['add']
        config.p = params['p']
        
        train(config)
        
        
config = Config()       
wrapper = Wrapper(config)
        
optimizer = BayesianOptimization(f=wrapper,pbounds=pbounds,random_state=42)
optimizer.maximize(init_points=opt_init_points,n_iter=opt_iters)    
best_iter = np.argmax([x['target'] for x in optimizer.res])
print('best iter:')
print(best_iter)
    
    