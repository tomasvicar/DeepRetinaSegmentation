import os
from skimage.io import imsave
import numpy as np
import torch 
from scipy.signal import convolve2d 
from sklearn.metrics import roc_auc_score
import h5py
import matplotlib.pyplot as plt

def valid_fcn_vessels(save_folder, config, model_name, data_names):
    
    device = config.device

    model=torch.load(model_name)
    model.eval()
    model=model.to(device)
    
    patch_size = model.config.patch_size  
    border = 17
    
    
    weigth_window=2*np.ones((patch_size,patch_size))
    weigth_window=convolve2d(weigth_window,np.ones((border,border))/np.sum(np.ones((border,border))),'same')
    weigth_window=weigth_window-1
    weigth_window[weigth_window<0.01]=0.01
    
    
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    accs = [] 
    aucs = []
    dices = []
    tps = []
    fps = []
    fns = []
    tns = []
    
    
        
    for name_num,name_mask in enumerate(data_names):
        

        
        h5data_file = config.data_path
        
        with h5py.File(h5data_file,"r") as h5data:
            
            groups_mask = name_mask.split('/');
            
            name_img = '_'.join(name_mask.split('_')[:-1])
            if not config.Gauss_and_Clahe:
                name_img = name_img.replace('Vessels','Images').replace('Disc','Images').replace('Cup','Images')
            else:
                name_img = name_img.replace('Vessels','Images_Gauss_and_Clahe').replace('Disc','Images_Gauss_and_Clahe').replace('Cup','Images_Gauss_and_Clahe') + '_gc'
            groups_img = name_img.split('/');
            
            name_fov = '_'.join(name_mask.split('_')[:-1]) + '_fov'
            name_fov = name_fov.replace('Vessels','Fov').replace('Disc','Fov').replace('Cup','Fov')
            groups_fov = name_fov.split('/')
            
            
            
            
            mask = h5data[groups_mask[0]][groups_mask[1]][:,:]
            mask = np.transpose(mask,(1,0))
            mask = (mask > 0).astype(np.uint8)
            
            img = h5data[groups_img[0]][groups_img[1]][:,:,:]
            img = np.transpose(img,(2,1,0))
            img = img.astype(np.float64)/255 - 0.5
            
            
            
            fov = h5data[groups_fov[0]][groups_fov[1]][:,:]
            fov = np.transpose(fov,(1,0))
            fov = (fov>0).astype(np.uint8)
            
            
        
            name_save = save_folder + '/' + os.path.split(name_img)[1] + '.png'
        
        
        img_size=img.shape
        

        
        sum_img=np.zeros(img_size[0:2])
        count_img=np.zeros(img_size[0:2])
        
        corners=[]
        cx=0
        while cx<img_size[0]-patch_size: 
            cy=0
            while cy<img_size[1]-patch_size:
                
                corners.append([cx,cy])
                
                cy=cy+patch_size-border
            cx=cx+patch_size-border
           
        cx=0
        while cx<img_size[0]-patch_size:
            corners.append([cx,img_size[1]-patch_size])
            cx=cx+patch_size-border
            
        cy=0
        while cy<img_size[1]-patch_size:
            corners.append([img_size[0]-patch_size,cy])
            cy=cy+patch_size-border   
            
        corners.append([img_size[0]-patch_size,img_size[1]-patch_size])
        
        for corner in corners:
            
            subimg = img[corner[0]:corner[0]+patch_size,corner[1]:corner[1]+patch_size,:]

            subimg = subimg.astype(np.float32)
            
            subimg = torch.from_numpy(np.transpose(subimg,(2,0,1)).astype(np.float32))
            
            subimg = subimg.unsqueeze(0)
            
            subimg=subimg.to(device)
        
            res=model(subimg)
            
            res=torch.sigmoid(res).detach().cpu().numpy()
            
            
            sum_img[corner[0]:(corner[0]+patch_size),corner[1]:(corner[1]+patch_size)]=sum_img[corner[0]:(corner[0]+patch_size),corner[1]:(corner[1]+patch_size)]+res*weigth_window
    
            count_img[corner[0]:corner[0]+patch_size,corner[1]:corner[1]+patch_size]=count_img[corner[0]:corner[0]+patch_size,corner[1]:corner[1]+patch_size]+weigth_window
        
        final=sum_img/count_img
        
        X = (final>0.5).astype(np.float64)[fov==1]
        X_nonbinar = final.astype(np.float64)[fov==1]
        Y = (mask>0).astype(np.float64)[fov==1]
        
        TP = np.sum(((X==1)&(Y==1)).astype(np.float64))
        FP = np.sum(((X==1)&(Y==0)).astype(np.float64))
        FN = np.sum(((X==0)&(Y==1)).astype(np.float64))
        TN = np.sum(((X==0)&(Y==0)).astype(np.float64))
        
        dice = (2 * TP )/ ((2 * TP) + FP + FN)
        
        acc = (TP+TN) / (TP + FP + FN + TN)
        
        auc = roc_auc_score(Y,X_nonbinar)
        
        
        dices.append(dice)
        
        
        accs.append(acc)
        aucs.append(auc)
        dices.append(dice)
        tps.append(TP)
        fps.append(FP)
        fns.append(FN)
        tns.append(TN)
        
        
        imsave(name_save,(final*255).astype(np.uint8))
        
        
        
        
        
    return accs,aucs,dices,tps,fps,fns,tns



if __name__ == "__main__":
        
    from DataSpliter import DataSpliter
    
    model_name = r"C:\Data\Vicar\retina_vessels_segmentation\best_models\norm_all_1_4_0.00008_gpu_7930869760.00000_train_0.31525_valid_0.33181.pt"
     
    
    
    model=torch.load(model_name)
    config = model.config
    
    data_split = DataSpliter.split_data(data_type=DataSpliter.DATA_TYPE_VESSELS)
    
     
    accs,aucs,dices,tps,fps,fns,tns = valid_fcn_vessels('../' + 'tmpx' + '/valid_results', config, model_name, data_split['valid'])