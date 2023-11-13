from glob import glob
from DataSpliter import DataSpliter
import h5py
from skimage.io import imread
import os
from utils.get_dice import get_dice
import matplotlib.pyplot as plt
import numpy as np

# path = r"C:\Data\Vicar\retina_jednotna_sit\nnUNet_raw_predict\Dataset001_VOVessels\predictTs"
# data_type = DataSpliter.VESSEL
path = r"C:\Data\Vicar\retina_jednotna_sit\nnUNet_raw_predict\Dataset003_VODisk\predictTs"
data_type = DataSpliter.DISK


fname_hdf5 = r"C:\Data\Vicar\retina_jednotna_sit\data_25_normalized.hdf5"

data_spliter = DataSpliter(fname_hdf5,
                           train_valid_test_frac=[0.7, 0.1, 0.2],
                           mask_type_use=[data_type],
                           seed=42
                           )


dataset_dict = data_spliter.get_dataset_dict()

fnames = glob(path + '/*.png')

dices = []
with h5py.File(fname_hdf5, 'r') as h5data:
    
    
    for file_num, fname in enumerate(fnames):
        print(str(file_num) + ' / ' + str(len(fnames)))
        if ' (1).png' in fname:
            continue
        
        res = imread(fname)
        
        fname_tmp = os.path.basename(fname).replace('.png', '')
        gt = h5data[fname_tmp][data_type][...]
        
        
        # plt.imshow(res)
        dice = get_dice(gt, res > 0)
        dices.append(dice)
        
print(np.mean(dices))
        
        
        
        
        


 





