

import numpy as np
import os
import h5py


class DataSpliter:
    
    VESSEL = 'vessel'
    VESSEL_CLASS = 'vessel_class'
    DISK = 'disk'
    CUP = 'cup'
    
    def __init__(self, data_fname_hdf5, train_valid_test_frac, mask_type_use, seed=42):
        self.data_fname_hdf5 = data_fname_hdf5
        self.train_valid_test_frac = train_valid_test_frac
        self.mask_type_use = mask_type_use
        self.seed = seed
        
        
    
    def get_dataset_dict(self):
        with h5py.File(self.data_fname_hdf5, "r") as file:
            dataset_dict = dict()
            for name in file.keys():
                
                info = dict()
                
                
                contains = []
                for mask_type in file[name].keys():
                    if mask_type != 'fov' and mask_type != 'img':
                        contains.append(mask_type)
                        
                if len(contains) == 0:
                    continue
                if len(set(contains).intersection(set(self.mask_type_use))) == 0:
                    continue
                
                info['masks'] = contains
                
                info['database_name'] = name.split('_')[0]
                info['split'] = name.split('_')[1]
                
                dataset_dict[name] = info
                
            
        return dataset_dict

      
    def na_splits_to_random(self, dataset_dict):
        
        databases = set([dataset_dict[key]['database_name'] for key in dataset_dict])
        
        for database in databases:
            
            names_database = [key for key in dataset_dict.keys() if dataset_dict[key]['database_name'] == database]
            names_database_na =  [key for key in names_database if dataset_dict[key]['split'] == 'na']
            
            N = len(names_database_na)
            perm = np.random.permutation(N)   
                 
            split_inds = np.array(self.train_valid_test_frac)
            split_inds = np.floor(np.cumsum(split_inds / np.sum(split_inds) * N)).astype(int)
            
            train_inds = perm[:split_inds[0]]
            valid_inds = perm[split_inds[0]:split_inds[1]]         
            test_inds = perm[split_inds[1]:]  
            
            for train_ind in train_inds:
                dataset_dict[names_database_na[train_ind]]['split'] = 'train'
            
            for valid_ind in valid_inds:
                dataset_dict[names_database_na[valid_ind]]['split'] = 'valid'
                
            for test_ind in test_inds:
                dataset_dict[names_database_na[test_ind]]['split'] = 'test'
            
        
        return dataset_dict


    def add_valid_if_not_exists_random(self, dataset_dict):
        databases = set([dataset_dict[key]['database_name'] for key in dataset_dict])
        
        for database in databases:
            
            names_database = [key for key in dataset_dict.keys() if dataset_dict[key]['database_name'] == database]
            
            names_database_valid =  [key for key in names_database if dataset_dict[key]['split'] == 'valid']
            if not len(names_database_valid):
                continue
            
            names_database_train =  [key for key in names_database if dataset_dict[key]['split'] == 'train']
            
            N = len(names_database_train)
            perm = np.random.permutation(N)   
                 
            split_inds = np.array(self.train_valid_test_frac[:2])
            split_inds = np.floor(np.cumsum(split_inds / np.sum(split_inds) * N)).astype(int)
            
            train_inds = perm[:split_inds[0]]
            valid_inds = perm[split_inds[0]:]          
            
            for train_ind in train_inds:
                dataset_dict[names_database_train[train_ind]]['split'] = 'train'
            
            for valid_ind in valid_inds:
                dataset_dict[names_database_train[valid_ind]]['split'] = 'valid'
                
        
        return dataset_dict

    def split_data(self):
        np.random.seed(self.seed)
        dataset_dict = self.get_dataset_dict()
        dataset_dict = self.na_splits_to_random(dataset_dict)
        dataset_dict = self.add_valid_if_not_exists_random(dataset_dict)
        
        return dataset_dict
        
    
if __name__ == "__main__":
    
    
    data_spliter = DataSpliter('../data_30.hdf5',
                               train_valid_test_frac=[0.7, 0.1, 0.2],
                               mask_type_use=[DataSpliter.VESSEL, DataSpliter.VESSEL_CLASS, DataSpliter.DISK, DataSpliter.CUP],
                               seed=42
                               )
    dataset_dict = data_spliter.split_data()
    