import numpy as np
from torch.utils import data

from dataset import Dataset
from split_data import DataSpliter

def measure_mean_std(config,data_train):
    
    train_generator = Dataset(data_train,augment=True,config=config,data_type='train')
    train_generator = data.DataLoader(train_generator,batch_size=config.train_batch_size,num_workers= config.train_num_workers, shuffle=True,drop_last=True)

    means = []
    stds = []
    
    for it,(img,mask) in enumerate(train_generator):
        print(it)
        img = img.detach().cpu().numpy() 
        
        for k in range(img.shape[0]):
            

            means.append(np.mean(img[0,...]))
            stds.append(np.std(img[0,...]))
    
        if it>200:
            break
    
    mean = np.mean(means)
    std = np.mean(stds)
    
    return mean,std



if __name__ == "__main__":

    
    
    from config import Config    
    
    config = Config()
    config.method = 'segmentation'
    
    
    config.train_num_workers = 0
    
    data_split = DataSpliter.split_data(data_type=DataSpliter.DATA_TYPE_VESSELS,seed=42)
    mean,std = measure_mean_std(config,data_split['train'])
    
    print(mean,std)
    
    
