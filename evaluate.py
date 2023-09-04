import torch
import numpy as np
from scipy.signal import convolve2d 
import h5py


from Config import Config
from DataSpliter import DataSpliter


def evaluate(dataset_dict, model_name):
    
    torch.cuda.empty_cache()
    
    
    device = torch.device('cuda:0')

    model=torch.load(model_name)
    model.eval()
    model=model.to(device)
    
    
    patch_size = model.config.patch_size  ### larger->faster, but need more ram (gpu ram)
    border = int(model.config.patch_size / 10)
    
    
    weigth_window=2*np.ones((patch_size,patch_size))
    weigth_window=convolve2d(weigth_window,np.ones((border,border))/np.sum(np.ones((border,border))),'same')
    weigth_window=weigth_window-1
    weigth_window[weigth_window<0.01]=0.01
    
    
    with h5py.File(config.dataset_fname, "r") as file:
        
        
        for file_num, name in enumerate(file):
            print(name)
    
    
    
    
    
if __name__ == "__main__":
    
    model_name = r"C:\Data\Vicar\retina_jednotna_sit\tmp\best_models\test_2_0.00010_gpu_71275520.00000_train_0.73237_valid_0.73501.pt"
    
    model=torch.load(model_name)
    
    config = model.config
    dataset_dict = model.dataset_dict
    
    dataset_dict_test = {key : value for key, value in dataset_dict.items() if value['split'] == 'test'}
    dice = evaluate(dataset_dict_test, model_name)