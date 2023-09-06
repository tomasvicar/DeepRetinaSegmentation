import torch
import numpy as np
from scipy.signal import convolve2d 
import h5py
import matplotlib.pyplot as plt
import random

from Config import Config
from DataSpliter import DataSpliter
from predict_by_parts import predict_by_parts
from utils.get_dice import get_dice





def evaluate(dataset_dict, model_name):
    
    torch.cuda.empty_cache()
    
    
    device = torch.device('cuda:0')

    model=torch.load(model_name)
    model.eval()
    model=model.to(device)
    
    
    patch_size = model.config.patch_size  ### larger->faster, but need more ram (gpu ram)
    border = int(model.config.patch_size / 8)
    out_layers = len(model.config.mask_type_use)
    
    mask_types_use = model.config.mask_type_use
    
    dices = {mask_type_use : [] for mask_type_use in mask_types_use}
    
    with h5py.File(model.config.dataset_fname, "r") as file:
        
        
        for file_num, name in enumerate(dataset_dict):
            
            # print(str(file_num) + ' / ' + str(len(dataset_dict)))
            # print(name)
            
            img = file[name]['img'][...]
            img = img.astype(np.float32) / 255 - 0.5
            img = torch.from_numpy(np.transpose(img, (2, 0, 1)))
            
            prediction = predict_by_parts(model, img, crop_size=patch_size, out_layers=out_layers, border=border, W=None)
            
            
            
            for mask_type in file[name].keys():
                for mask_type_idx, mask_type_use in enumerate(mask_types_use):
                    
                    if not mask_type_use in mask_type:
                        continue
                    
                    mask = file[name][mask_type_use][...]
                    
                    prediction_current = prediction[mask_type_idx, :, :].detach().cpu().numpy() > 0
                    
                    dice = get_dice(mask, prediction_current)
                    dices[mask_type_use].append(dice)
                    
            
    return dices
    
    
if __name__ == "__main__":
    
    model_name = r"D:\retina_jednotna_sit\tmp\best_models\test_2_0.00010_gpu_71275520.00000_train_0.73237_valid_0.73501.pt"
    
    model=torch.load(model_name)
    
    config = model.config
    dataset_dict = model.dataset_dict
    
    dataset_dict_test = {key : value for key, value in dataset_dict.items() if value['split'] == 'test'}
    
    def random_sample_dict(d, n):
        if n > len(d):
            raise ValueError("Sample size exceeds dictionary size.")
        
        sampled_keys = random.sample(list(d.keys()), n)
        return {key: d[key] for key in sampled_keys}
    
    dataset_dict_test =  random_sample_dict(dataset_dict_test, 30)
    
    dices = evaluate(dataset_dict_test, model_name)
    dices_types = {key : np.mean(value) for key, value in dices.items()}
    dice = np.mean(np.array(list(dices_types.values())))
    print(dices)
    print(dices_types)
    print(dice)