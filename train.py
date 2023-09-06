import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from shutil import copyfile
import os
import numpy as np
from shutil import rmtree

from Config import Config
from DataSpliter import DataSpliter
from Dataset import Dataset
from utils.Log import Log
from utils.training_fcns import l1_loss, l2_loss, dice_loss_logit, bce_logit, dice_loss_logit_mask_type
from utils.get_dice import get_dice, get_dice_mask_type
from Unet import Unet
from evaluate import evaluate



def train(config):
    
    dataset_dict = DataSpliter(config.dataset_fname, config.train_valid_test_frac, config.mask_type_use, config.seed).split_data()
    
    
    device = torch.device(config.device)
    
    torch.cuda.empty_cache()
    
    
    
    dataset_dict_train = {key : value for key, value in dataset_dict.items() if value['split'] == 'train'}
    train_generator = Dataset(dataset_dict_train, augment=True, config=config, data_type='train')
    train_generator = DataLoader(train_generator, batch_size=config.train_batch_size, num_workers=config.train_num_workers, shuffle=True, drop_last=True)

    dataset_dict_valid = {key : value for key, value in dataset_dict.items() if value['split'] == 'valid'}
    valid_generator = Dataset(dataset_dict_valid, augment=False, config=config, data_type='valid')
    valid_generator = DataLoader(valid_generator, batch_size=config.valid_batch_size, num_workers=config.valid_num_workers, shuffle=True, drop_last=True)

    
    model = Unet(filters=config.filters, in_size=3, out_size=len(config.mask_type_use), do=config.drop_out, depth=config.depth)
    
    model.config = config
    model.dataset_dict = dataset_dict
    model = model.to(device)
    
    model.log = Log(names=['loss','dice'])
    
    
    optimizer = torch.optim.AdamW(model.parameters(),lr =config.init_lr ,betas= (0.9, 0.999),eps=1e-8,weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.lr_changes_list, gamma=config.gamma, last_epoch=-1)
    
    for epoch in range(config.max_epochs):
        
        model.train()
        N = len(train_generator)
        for it, (img, mask, mask_type) in enumerate(train_generator):
            
            # print(str(it) + ' / ' + str(N))
            
            img = img.to(torch.device(config.device))
             
            res=model(img)
            

            loss = dice_loss_logit_mask_type(res, mask, mask_type, config.mask_type_use)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
            dice = get_dice_mask_type(res, mask, mask_type, config.mask_type_use)
            
            model.log.append_train([loss.detach().cpu().numpy(), dice])
    
        model.eval()
        with torch.no_grad():
            for img, mask, mask_type in valid_generator:
                
                img = img.to(torch.device(config.device))
                
                res = model(img)
                
                
                loss = dice_loss_logit_mask_type(res, mask, mask_type, config.mask_type_use)
  
                    
                dice = get_dice_mask_type(res, mask, mask_type, config.mask_type_use)
                # print(round(dice, 2), end=" ")
                
                model.log.append_valid([loss.detach().cpu().numpy(),dice])
            
        
        model.log.save_and_reset()
    
        res=torch.sigmoid(res)
        
        res = res.detach().cpu().numpy()
        mask = mask.detach().cpu().numpy()
        img = img.detach().cpu().numpy()
        for k in range(res.shape[0]):
            plt.imshow(np.concatenate((img[k,0,:,:] + 0.5,res[k,0,:,:],mask[k,0,:,:]),axis=1),vmin=0,vmax=1)
            plt.show()
            plt.close()
            
            
        xstr = lambda x:"{:.5f}".format(x)
        lr=optimizer.param_groups[0]['lr']
        info= '_' + str(epoch) + '_' + xstr(lr) + '_gpu_' + xstr(np.max((model.log.model_memory))) + '_train_'  + xstr(model.log.train_log['loss'][-1]) + '_valid_' + xstr(model.log.valid_log['loss'][-1]) 
        
        print(info)
        
        tmp_folder = config.tmp_folder.replace('_method_', config.method)
        
        model_name = tmp_folder + os.sep + info  + '.pt'
        
        model.log.save_log_model_name(model_name)
        
        
        if not os.path.isdir(os.path.split(model_name)[0]):
            os.makedirs(os.path.split(model_name)[0])
        
        torch.save(model, model_name)
        
        model_name_plot = model_name.replace('.pt','loss.png')
        
        model.log.plot(model_name_plot)
        
        scheduler.step()
        
        
    tmp_folder = config.tmp_folder.replace('_method_', config.method)
    best_model_folder = config.best_model_folder.replace('_method_', config.method) 
    
    best_model_ind = np.argmin(model.log.valid_log['loss'])
    best_model_name = model.log.model_names[best_model_ind]   
    best_model_name_new = best_model_name.replace(tmp_folder, best_model_folder)
    

    if not os.path.isdir(os.path.split(best_model_name_new)[0]):
        os.makedirs(os.path.split(best_model_name_new)[0])
    
    copyfile(best_model_name,best_model_name_new)
    
    # if os.path.isdir(tmp_folder):
    #     rmtree(tmp_folder) 
        
    
    dataset_dict_test = {key : value for key, value in dataset_dict.items() if value['split'] == 'test'}
    dices = evaluate(dataset_dict_test, model_name)
    dices_types = {key : np.mean(value) for key, value in dices.items()}
    dice = np.mean(np.array(list(dices_types.values())))
    
    return dice
    
    
if __name__ == "__main__":
    config = Config()
    train(config)
    
    