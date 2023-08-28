from torch.utils import data
import numpy as np
import torch 
import time
import h5py
import matplotlib.pyplot as plt
import itertools

def rand(size=1):
    tmp = torch.rand(size).numpy()
    if size == 1:
        return tmp[0]
    else:
        return tmp
    

def randint(maxval, size=1):
    tmp = torch.randint(maxval,(size,)).numpy()
    if size == 1:
        return tmp[0]
    else:
        return tmp
    
    
class RandomCropper():
    def __init__(self, in_shape, out_shape):
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.random_pos = [randint(in_shape[0] - out_shape[0]), randint(in_shape[1] - out_shape[1])]
             
    def crop(self, img):
        r = self.random_pos
        return img[r[0]:r[0] + self.out_shape[0], r[1]:r[1] + self.out_shape[1], ...]


def make_generator_infinite(generator):
    while True:
        for data in generator:
            yield data
    
    

class Dataset(data.Dataset):
    
    def __init__(self, dataset_dict, augment, config, data_type):
        
        self.dataset_dict = dataset_dict
        self.augment = augment
        self.config = config
        self.data_type = data_type
        
        
        self.databases = set([dataset_dict[key]['database_name'] for key in dataset_dict])
        self.name_groups_databases = dict()
        for database in self.databases:
            self.name_groups_databases[database] = [key for key in dataset_dict if dataset_dict[key]['database_name'] == database]
            
            
        self.mask_types = self.config.mask_type_use
        self.name_groups_mask_types = dict()
        for mask_type in self.mask_types:
            self.name_groups_mask_types[mask_type] = [key for key in dataset_dict if mask_type in dataset_dict[key]['masks']]    
        
        
        
        self.N = len(self.dataset_dict) * 10
        
        self.h5data_fname = self.config.dataset_fname
        self.h5data = None
        
        
        
    def __len__(self):
        return self.N
    
    
    def __getitem__(self, idx):
        
        if  self.h5data is None:
            self.h5data = h5py.File(self.h5data_fname, 'r')
            
        
        mask_type_idx = randint(len(self.mask_types))
        mask_type = self.mask_types[mask_type_idx]
        
        names_mask_type = self.name_groups_mask_types[mask_type]
        name_idx = randint(len(names_mask_type))
        name = names_mask_type[name_idx]
        
        mask_types_all_labelers = [x for x in self.dataset_dict[name]['masks'] if mask_type in x]
        mask_type_final = mask_types_all_labelers[randint(len(mask_types_all_labelers))]
        
        img_hdf5 = self.h5data[name]['img']
        mask_hdf5 = self.h5data[name][mask_type_final]
        
        
        in_shape = img_hdf5.shape
        out_shape = [self.config.patch_size, self.config.patch_size]
        
        random_croper = RandomCropper(in_shape, out_shape) 
        img = random_croper.crop(img_hdf5)
        mask = random_croper.crop(mask_hdf5)
        
        print(name)
        print(mask_type_final)
        print(mask.shape)
        
        
        img = img.astype(np.float32) / 255 - 0.5
        img = torch.from_numpy(np.transpose(img, (2, 0, 1)))
        
        mask = mask.astype(np.float32)
        if mask_type == 'ves_class':
            mask = 0.3 * mask[:, :, 0] + 0.6 *  mask[:, :, 1] + 0.9 *  mask[:, :, 2] # fix this
        mask = np.expand_dims(mask,2)
        
        mask = torch.from_numpy(np.transpose(mask, (2, 0, 1)))
        
        
        
        return img, mask, mask_type
        
        
        
        
if __name__ == "__main__":

    from torch.utils.data import DataLoader
    
    from Config import Config    
    from DataSpliter import DataSpliter
    
    
    
    config = Config()
    config.train_batch_size = 1

    dataset_dict = DataSpliter(config.dataset_fname, config.train_valid_test_frac, config.mask_type_use, config.seed).split_data()        
        
    dataset_dict_train = {key : value for key, value in dataset_dict.items() if value['split'] == 'train'}
    train_generator = Dataset(dataset_dict_train, augment=True, config=config, data_type='train')
    train_generator = DataLoader(train_generator, batch_size=config.train_batch_size, num_workers=config.train_num_workers, shuffle=True, drop_last=True)
    
    for it,(img, mask, mask_type) in enumerate(train_generator):
        
        img = np.transpose(img[0, :, :, :].numpy(),(1 ,2, 0)) + 0.5
        plt.imshow(img, vmin=0, vmax=1)
        plt.show()
        mask = mask[0, 0, :, :].numpy()
        plt.imshow(mask, vmin=0, vmax=1)
        plt.show()

    