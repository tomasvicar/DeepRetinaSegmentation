from torch.utils import data
import numpy as np
import torch 
import h5py
import matplotlib.pyplot as plt


from DataSpliter import DataSpliter
from augmentation import augmentation


class Dataset(data.Dataset):


    def __init__(self, names,augment,config,data_type=None):
       
        self.names = names
        self.augment = augment
        self.config = config
        self.data_type = data_type
        
        self.names = self.names
        
        self.N = len(self.names)

        self.h5data_file = config.data_path 

        self.h5data = None

    def __len__(self):
        return self.N*self.config.multiply_dataset


    def __getitem__(self, idx):
        
        if  self.h5data is None:
            self.h5data = h5py.File(self.h5data_file, 'r')

        idx = idx % self.N

        name_mask = self.names[idx]
        groups_mask = name_mask.split('/');
        
        name_img = '_'.join(name_mask.split('_')[:-1])
        if not self.config.Gauss_and_Clahe:
            name_img = name_img.replace('Vessels','Images').replace('Disc','Images').replace('Cup','Images')
        else:
            name_img = name_img.replace('Vessels','Images_Gauss_and_Clahe').replace('Disc','Images_Gauss_and_Clahe').replace('Cup','Images_Gauss_and_Clahe') + '_gc'
        groups_img = name_img.split('/');
        
        
        
        ###twice the size for aditional augmetnation with deformation without border
        if self.augment:
            in_size = self.h5data[groups_mask[0]][groups_mask[1]].shape
            out_size = [self.config.patch_size*2,self.config.patch_size*2]
        else: 
            in_size = self.h5data[groups_mask[0]][groups_mask[1]].shape
            out_size = [self.config.patch_size,self.config.patch_size]
        r1 = torch.randint(in_size[0] - out_size[0],(1,1)).view(-1).numpy()[0]
        r2 = torch.randint(in_size[1] - out_size[1],(1,1)).view(-1).numpy()[0]
        r = [r1, r2]

        mask = self.h5data[groups_mask[0]][groups_mask[1]][r[0]:r[0]+out_size[0],r[1]:r[1]+out_size[1]]
        mask = np.transpose(mask,(1,0))
        mask = (mask > 0).astype(np.uint8)
        
        img = self.h5data[groups_img[0]][groups_img[1]][1:2,r[0]:r[0]+out_size[0],r[1]:r[1]+out_size[1]]
        img = np.transpose(img,(2,1,0))
        img = img.astype(np.float64)/255 - 0.5
        
        
        if self.augment:
            img,mask = augmentation(img,mask,self.config)
        

        
        mask = mask.reshape([mask.shape[0],mask.shape[1],1])
        mask=torch.from_numpy(np.transpose(mask,(2,0,1)).astype(np.float32))
        img=torch.from_numpy(np.transpose(img,(2,0,1)).astype(np.float32))
        
        
        return img,mask




if __name__ == "__main__":


    from config import Config    
    config = Config()

    data_split = DataSpliter.split_data(data_type=DataSpliter.DATA_TYPE_VESSELS,seed=42)
    
    train_generator = Dataset(data_split['train'],augment=True,config=config)
    train_generator = data.DataLoader(train_generator,batch_size=1,num_workers= 0, shuffle=True,drop_last=True)
    
    for it,(img,mask) in enumerate(train_generator):
        
        plt.imshow(np.transpose(img[0,:,:,:].numpy(),(1,2,0))+0.5,vmin=0,vmax=1)
        plt.show()
        plt.imshow(np.transpose(mask[0,:,:,:].numpy(),(1,2,0))+0.5,vmin=0,vmax=1)
        plt.show()
        break