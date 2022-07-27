import numpy as np
from config import Config
import os
import h5py




class DataSpliter:

    DATA_TYPE_VESSELS = 'Vessels'

    @staticmethod
    def filter_database(names,database_name):
        
        names_new = []
        for name in names:
            tmp = os.path.split(name)[1].split('_')[0]
            if tmp == database_name:
                names_new.append(name)
            
        
        return names_new
    
    

    @staticmethod
    def split_data(data_type=DATA_TYPE_VESSELS,seed=42,config=Config()):
        
        
        np.random.seed(seed)
        
        data_split = dict()
    
        data_split['train'] = []
        data_split['valid'] = []
        data_split['test'] = []
        
        
        with h5py.File(config.data_path,'r') as f:
            if data_type==DataSpliter.DATA_TYPE_VESSELS:
                names = f['Vessels'].keys()
                names = [name for name in names if name.endswith('ves') ]
                names = ['Vessels' + '/' + name for name in names]
            else:
                raise Exception('incorect data type')
        
        
        
        database_names = [os.path.split(name)[1].split('_')[0] for name in names]
        data_split['database_names'] = list(set(database_names))
        


        perm=np.random.permutation(len(database_names))   
     
        split_ind=np.array(config.split_ratio_train_valid_test)
        split_ind=np.floor(np.cumsum(split_ind/np.sum(split_ind)*len(names))).astype(int)
        
        train_ind=perm[:split_ind[0]]
        valid_ind=perm[split_ind[0]:split_ind[1]]         
        test_ind=perm[split_ind[1]:]  
    
        data_split['train'] = [names[k] for k in train_ind]
        data_split['valid'] = [names[k] for k in valid_ind]
        data_split['test'] = [names[k] for k in test_ind]
            

        
        return data_split




if __name__ == "__main__":
    data_split = DataSpliter.split_data()
    
    
    