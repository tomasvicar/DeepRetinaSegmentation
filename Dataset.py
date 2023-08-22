from torch.utils import data
import numpy as np
import torch 
import time
import h5py
import matplotlib.pyplot as plt
import itertools

def rand(size=1):
    return torch.rand(size).numpy()

def randint(maxval, size=1):
    return torch.randint(maxval,(size,)).numpy()


class Dataset(data.Dataset):
    
    def __init__(self, dataset_dict, augment, config, data_type):
        
        self.dataset_dict = dataset_dict
        self.augment = augment
        self.config = config
        self.data_type = data_type
        
        
        databases = set([dataset_dict[key]['database_name'] for key in dataset_dict])
        self.name_groups_databases = dict()
        for database in databases:
            self.name_groups_databases[database] = [key for key in dataset_dict if dataset_dict[key]['database_name'] == database]
            
            
        mask_types = config.mask_type_use
        self.name_groups_mask_types = dict()
        for mask_type in mask_types:
            self.name_groups_mask_types[mask_type] = [key for key in dataset_dict if mask_type in dataset_dict[key]['masks']]    
        
        
        
        self.N = len(self.dataset_dict) * 100
        
        self.h5data_fname = config.dataset_fname
        self.h5data = None
        
        
        
    def __len__(self):
        return self.N
    
    
    def __getitem__(self, idx):
        
        
        
        
        