import os
import numpy as np
import torch
from torch.utils import data
import matplotlib.pyplot as plt
from shutil import copyfile
import segmentation_models_pytorch as smp
from shutil import rmtree

from dataset import Dataset
from utils.log import Log
from utils.gpu_logger import GpuLogger
from utils.training_fcns import l1_loss,l2_loss,dice_loss_logit,bce_logit
from utils.get_dice import get_dice
from unet import Unet




def train(config,data_train,data_valid):

    gpuLogger = GpuLogger()
    device = torch.device(config.device)
    
    torch.cuda.empty_cache()

    train_generator = Dataset(data_train,augment=True,config=config,data_type='train')
    train_generator = data.DataLoader(train_generator,batch_size=config.train_batch_size,num_workers= config.train_num_workers, shuffle=True,drop_last=True)

    valid_generator = Dataset(data_valid,augment=False,config=config,data_type='valid')
    valid_generator = data.DataLoader(valid_generator,batch_size=config.valid_batch_size, num_workers=config.valid_num_workers, shuffle=True,drop_last=False)

    
    if config.net_name == 'unet':
        model = Unet(filters=config.filters,in_size=config.in_channels,out_size=1,do=config.drop_out,depth=config.depth)
    
    else:
        if config.model_name_load == 'imagenet':
            model = smp.Unet(
                encoder_name=config.net_name,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights='imagenet',     # use `imagenet` pre-trained weights for encoder initialization
                in_channels=config.in_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=1,                      # model output channels (number of classes in your dataset)
            )
            
        
        
        elif config.model_name_load=='none':
            # model=Unet(config, filters=config.filters,in_size=config.in_size,out_size=config.out_size)
            model = smp.Unet(
                encoder_name=config.net_name,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights=None,     # use `imagenet` pre-trained weights for encoder initialization
                in_channels=config.in_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=1,                      # model output channels (number of classes in your dataset)
            )
            
        else:
            model=torch.load(config.model_name_load)
        
        
    model.config = config
    model = model.to(device)
    
    model.log = Log(names=['loss','dice'])
    
    
    
    optimizer = torch.optim.AdamW(model.parameters(),lr =config.init_lr ,betas= (0.9, 0.999),eps=1e-8,weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.lr_changes_list, gamma=config.gamma, last_epoch=-1)

    for epoch in range(config.max_epochs):
        
        model.train()
        for img,mask in train_generator:
            
            img = img.to(torch.device(config.device))
             
            res=model(img)
            
            
            if config.method == 'pretraining':
                loss=l2_loss(res,mask)
            else:
                if config.loss == 'bce':
                    loss = bce_logit(res,mask)
                elif config.loss == 'dice_loss':
                    loss=dice_loss_logit(res,mask)
                else:
                    raise Exception('incorect loss')
            
            optimizer.zero_grad()
            loss.backward()
            gpuLogger.save_gpu_memory()
            optimizer.step()
            gpuLogger.save_gpu_memory()
            
            dice = get_dice(res,mask)
            
            model.log.append_train([loss.detach().cpu().numpy(),dice])
            
                
            
        model.eval()
        with torch.no_grad():
            for img,mask in valid_generator:
                
                img = img.to(torch.device(config.device))
                
                res=model(img)
                
                
                if config.method == 'pretraining':
                    loss=l2_loss(res,mask)
                else:
                    if config.loss == 'bce':
                        loss = bce_logit(res,mask)
                    elif config.loss == 'dice_loss':
                        loss=dice_loss_logit(res,mask)
                    else:
                        raise Exception('incorect loss')
                    
                    
                dice = get_dice(res,mask)
                
                model.log.append_valid([loss.detach().cpu().numpy(),dice])
                gpuLogger.save_gpu_memory()
            
        
        model.log.save_and_reset()
        

        
        
        if not (config.method == 'pretraining'):
            res=torch.sigmoid(res)
            
        res = res.detach().cpu().numpy()
        mask = mask.detach().cpu().numpy()
        img = img.detach().cpu().numpy()
        for k in range(res.shape[0]):
            plt.imshow(np.concatenate((img[k,0,:,:],res[k,0,:,:],mask[k,0,:,:]),axis=1),vmin=0,vmax=1)
            plt.show()
            plt.close()
    
    
        xstr = lambda x:"{:.5f}".format(x)
        lr=optimizer.param_groups[0]['lr']
        info= '_' + str(epoch) + '_' + xstr(lr) + '_gpu_' + xstr(np.max(gpuLogger.measured_gpu_memory)) + '_train_'  + xstr(model.log.train_log['loss'][-1]) + '_valid_' + xstr(model.log.valid_log['loss'][-1]) 
        
        print(info)
        
        model_name=config.model_save_dir+ os.sep + config.method + info  + '.pt'
        
        model.log.save_log_model_name(model_name)
        
        torch.save(model,model_name)
        
        if not os.path.isdir(config.results_folder + os.sep + config.method):
            os.mkdir(config.results_folder + os.sep + config.method)
        
        model_name2=config.results_folder + os.sep + config.method + os.sep + config.method + info  + '.pt'
        
        model.log.plot(model_name2.replace('.pt','loss.png'))
        
        scheduler.step()
    
    
    # last_x_to_use = 10
    # last_x_to_use = len(model.log.valid_log['loss']) - last_x_to_use
    # best_model_ind = np.argmin(model.log.valid_log['loss'][last_x_to_use:]) + last_x_to_use
    
    best_model_ind = np.argmin(model.log.valid_log['loss'])
    best_model_name = model.log.model_names[best_model_ind]   
    best_model_name_new = best_model_name.replace(config.model_save_dir,config.best_models_dir)
    
    copyfile(best_model_name,best_model_name_new)
    
    if os.path.isdir(config.model_save_dir):
        rmtree(config.model_save_dir) 
        
    if not os.path.isdir(config.model_save_dir):
        os.mkdir(config.model_save_dir)


    return best_model_name_new



