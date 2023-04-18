import torch
from torch.utils import data
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil


from DataSpliter import DataSpliter
from config import Config
from Unet import Unet
from Log import Log
from Dataset import Dataset
from utils.dice_loss_logit import dice_loss_logit
from utils.get_dice import get_dice

def train(config,data_split):
    
    torch.cuda.empty_cache()
    
    device = torch.device(config.device)
    
    if not os.path.isdir(config.model_save_dir):
        os.makedirs(config.model_save_dir)
      
    if not os.path.isdir(config.best_models_dir):
        os.makedirs(config.best_models_dir)
    
    
    train_generator = Dataset(data_split['train'],augment=True,config=config,data_type='train')
    train_generator = data.DataLoader(train_generator,batch_size=config.train_batch_size,num_workers= config.train_num_workers, shuffle=True,drop_last=True)
    
    valid_generator = Dataset(data_split['valid'],augment=False,config=config,data_type='valid')
    valid_generator = data.DataLoader(valid_generator,batch_size=config.valid_batch_size, num_workers=config.valid_num_workers, shuffle=True,drop_last=True)

    
    
    model = Unet(filters=config.filters,in_size=1,out_size=1,depth=config.depth)
    model.config = config
    model = model.to(device)
    model.log = Log(names=['loss','dice'])
    
    
    optimizer = torch.optim.AdamW(model.parameters(),lr =config.init_lr ,betas= (0.9, 0.999),eps=1e-8,weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.lr_changes_list, gamma=config.gamma, last_epoch=-1)
    
    for epoch in range(config.max_epochs):
        
        model.train()
        N = len(train_generator);
        for it, (img,mask) in enumerate(train_generator):
            
            if (it % 100) == 0:
                print(str(it) + ' / ' + str(N))
            
            img = img.to(torch.device(config.device))
             
            res = model(img)

            loss = dice_loss_logit(res,mask)
                
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            dice = get_dice(res,mask)
            
            model.log.append_train([loss.detach().cpu().numpy(),dice])
            
                
            
        model.eval()
        with torch.no_grad():
            N = len(valid_generator);
            for it, (img, mask) in enumerate(valid_generator):
                
                if (it % 100) == 0:
                    print(str(it) + ' / ' + str(N))
                
                img = img.to(torch.device(config.device))
                
                res=model(img)
                
                loss=dice_loss_logit(res,mask)
                    
                dice = get_dice(res,mask)
                
                model.log.append_valid([loss.detach().cpu().numpy(),dice])
        
        res=torch.sigmoid(res)
        
        # for k in range(4):
        #     print(k)
        #     plt.imshow(img[k,1,:,:].detach().cpu().numpy() + 0.5)
        #     plt.show()
        #     plt.imshow(res[k,0,:,:].detach().cpu().numpy())
        #     plt.show()
    
        
        
        
        
        model.log.save_and_reset()
        
        xstr = lambda x:"{:.5f}".format(x)
        info = ''
        info += '_' + str(epoch) 
        info +=  '_' + xstr(optimizer.param_groups[0]['lr']) 
        info +=  '_gpu_' + xstr(torch.cuda.max_memory_allocated() / 10**9) 
        info +=  '_train_'  + xstr(model.log.train_log['loss'][-1]) 
        info +=  '_valid_' + xstr(model.log.valid_log['loss'][-1])
        print(info)
        
        model_name = config.model_save_dir + os.sep + config.method + info  + '.pt'
        model.log.save_log_model_name(model_name)
        torch.save(model,model_name)
    
        
        model_name_png = config.results_folder + os.sep + config.method + os.sep + config.method + info  + 'loss.png'
        if not os.path.isdir(os.path.split(model_name_png)[0]):
            os.makedirs(os.path.split(model_name_png)[0])
        model.log.plot(model_name_png)
        
        scheduler.step()
        
    best_model_ind = np.argmin(model.log.valid_log['loss'])
    best_model_name = model.log.model_names[best_model_ind]   
    best_model_name_new = best_model_name.replace(config.model_save_dir,config.best_models_dir)
    
    shutil.copyfile(best_model_name,best_model_name_new)
    
    # if os.path.isdir(config.model_save_dir):
    #     shutil.rmtree(config.model_save_dir) 
        
        
    return best_model_name_new


if __name__ == "__main__":
    
    data_split = DataSpliter.split_data()
    config = Config()
    
    model_name = train(config,data_split)
    
    
    
    