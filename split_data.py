import numpy as np
from config import Config
import os
from glob import glob



class DataSpliter:

    DATA_TYPE_VESSELS = 'Vessels'
    
    
    
    
    @staticmethod
    def filter_database(names,database_name):
        
        names_new = []
        for name in names:
            tmp = os.path.split(name)[1].replace('.png','').split('_')[0]
            if tmp == database_name:
                names_new.append(name)
            
        
        return names_new

    

    @staticmethod
    def split_data(data_type=DATA_TYPE_VESSELS,seed=42,config=Config()):
        
        
        np.random.seed(seed)
        
        data_split = dict()
        
      
        names = glob(config.data_path + '/Pretraining/*.png')
        
        perm=np.random.permutation(len(names))   
             
        split_ind=np.array(config.split_ratio_pretrain_train_valid)
        split_ind=np.floor(np.cumsum(split_ind/np.sum(split_ind)*len(names))).astype(np.int)
        
        
        train_ind=perm[:split_ind[0]]
        valid_ind=perm[split_ind[0]:]
        
        
        data_split['pretrain_train'] = [names[k] for k in train_ind]
        data_split['pretrain_valid'] = [names[k] for k in valid_ind]
        
        
        
        
        
        
        if data_type==DataSpliter.DATA_TYPE_VESSELS:
            
            
            data_split['train'] = []
            data_split['valid'] = []
            data_split['test'] = []
            
            
            names = glob(config.data_path + '/Vessels/*ves.png')
            
            dataset_names = [os.path.split(name)[1].replace('.png','').split('_')[0] for name in names]
            dataset_splits = [os.path.split(name)[1].replace('.png','').split('_')[1] for name in names]
            
            data_split['database_names'] = list(set(dataset_names))
            
            for dataset_name in set(dataset_names):
                
                dataset_splits_selected = [y for x,y in zip(dataset_names,dataset_splits) if x==dataset_name]
                dataset_names_selected = [y for x,y in zip(dataset_names,names) if x==dataset_name]
                
                if any([x=='test' for x in  dataset_splits_selected]):
                # if False: ############################################################################
                                        
                    test_ind = []
                    other_ind = []
                    for ind, dataset_split in enumerate(dataset_splits_selected):
                        
                        if dataset_split=='test':
                            test_ind.append(ind)
                        else:
                            other_ind.append(ind)
                    
                    
                    perm=np.random.permutation(len(other_ind))   
                 
                    split_ind=np.array(config.split_ratio_train_valid_test[:2])
                    split_ind=np.floor(np.cumsum(split_ind/np.sum(split_ind)*len(other_ind))).astype(np.int)
                    
                    train_ind=perm[:split_ind[0]]
                    valid_ind=perm[split_ind[0]:]  
                    
                    train_ind = [other_ind[k] for k in train_ind]
                    valid_ind = [other_ind[k] for k in valid_ind]
                    
                    
                    
                else:
            

                    perm=np.random.permutation(len(dataset_names_selected))   
                 
                    split_ind=np.array(config.split_ratio_train_valid_test)
                    split_ind=np.floor(np.cumsum(split_ind/np.sum(split_ind)*len(dataset_names_selected))).astype(np.int)
                    
                    train_ind=perm[:split_ind[0]]
                    valid_ind=perm[split_ind[0]:split_ind[1]]         
                    test_ind=perm[split_ind[1]:]  
            
                data_split['train'].extend([dataset_names_selected[k] for k in train_ind])
                data_split['valid'].extend([dataset_names_selected[k] for k in valid_ind])
                data_split['test'].extend([dataset_names_selected[k] for k in test_ind])
            
            
            
            
            
            
            
            
            
        else:
            raise Exception('wrong data type')
        
        
        names_all = glob(config.data_path + '/Vessels/*.png')
        names_all_ves2plus =[x for x in names_all if not 'ves.png' in x]
        
        for tmp_data in [data_split['train'],data_split['valid'],data_split['test']]:
            
            tmp = tmp_data.copy()
            for tmp_sample in tmp:
                
                for tmp_sample_2plus in names_all_ves2plus:
                    
                    if tmp_sample[:-4] in tmp_sample_2plus:
                        tmp_data.append(tmp_sample_2plus)
                
                
                
        
        
        return data_split







if __name__ == "__main__":
    
    data_split = DataSpliter.split_data()
