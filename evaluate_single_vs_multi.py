import torch
import random
import numpy as np

from DataSpliter import DataSpliter
from evaluate import evaluate

models = [
    # r"D:\retina_jednotna_sit\test_best\best_models\_120_0.00020_gpu_84644864.00000_train_0.33739_valid_0.37708.pt",
    # r"D:\retina_jednotna_sit\test_vessels_seg_best\best_models\_117_0.00013_gpu_74945536.00000_train_0.30473_valid_0.31326.pt",
    # r"D:\retina_jednotna_sit\test_disk_0\best_models\_121_0.00080_gpu_40342528.00000_train_0.20196_valid_0.20841.pt",
    r"D:\retina_jednotna_sit\test_disk_1\best_models\_131_0.00000_gpu_54498304.00000_train_0.10829_valid_0.17050.pt",
    ]

mask_type = [
    # [DataSpliter.VESSEL, DataSpliter.DISK, DataSpliter.CUP],
    # [DataSpliter.VESSEL,],
    # [DataSpliter.DISK,],
    [DataSpliter.DISK,],
    ]


results = dict()

for model_ind, (model_name, mask_type_use) in enumerate(zip(models, mask_type)):
    
    model=torch.load(model_name)
    
    config = model.config
    dataset_dict = model.dataset_dict
    
    dataset_dict_test = {key : value for key, value in dataset_dict.items() if value['split'] == 'test'}
    
    
    dices = evaluate(dataset_dict_test, model_name, mask_type_use)
    dices_types = {key : np.mean(value) for key, value in dices.items()}
    dice = np.mean(np.array(list(dices_types.values())))
    # print(dices)
    print(dices_types)
    # print(dice)
    
    results['dices_' + str(model_ind)] = dices
    results['dices_types_' + str(model_ind)] = dices_types