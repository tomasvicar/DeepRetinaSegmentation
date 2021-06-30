import os
from skimage.io import imread
from skimage.io import imsave
import numpy as np
import torch 
from scipy.signal import convolve2d 
from sklearn.metrics import roc_auc_score
import cv2
from skimage.color import rgb2gray
import h5py


def test_fcn_vessels(save_folder, config, model_name, data_names):
    
    device = torch.device('cuda:0')

    model=torch.load(model_name)
    model.eval()
    model=model.to(device)
    
    patch_size = model.config.patch_size  ### larger->faster, but need more ram (gpu ram)
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
    
    
        
    for name_num,name in enumerate(data_names):
        

        
        h5data_file = model.config.data_path + '/' +  'dataset.hdf5'
        
        with h5py.File(h5data_file,"r") as h5data:
            
            tmp = name.split('/')
            mask = h5data[tmp[0]][tmp[1]][:,:]
            mask = np.transpose(mask,(1,0))
            mask = (mask>0).astype(np.uint8)
            name_tmp = '_'.join(name.split('_')[:-1])
            name_tmp = name_tmp.replace('Vessels','Images').replace('Disc','Images').replace('Cup','Images')
        
            name_save = save_folder + '/' + os.path.split(name_tmp)[1] + '.png'
        
            tmp = name_tmp.split('/') 
            img = h5data[tmp[0]][tmp[1]][:,:,:]
            img = np.transpose(img,(2,1,0))
            
            
            
            name_tmp = '_'.join(name.split('_')[:-1]) + '_fov'
            name_tmp = name_tmp.replace('Vessels','Fov').replace('Disc','Fov').replace('Cup','Fov')
            tmp = name_tmp.split('/')
            fov = h5data[tmp[0]][tmp[1]][:,:]
            fov = np.transpose(fov,(1,0))
            fov = (fov>0).astype(np.uint8)
        
        
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
            
            
            if model.config.img_type == 'rgb':
                pass
            elif model.config.img_type == 'green':
                subimg = subimg[:,:,1]
                subimg = np.expand_dims(subimg,2)
            elif model.config.img_type == 'gray':  
                subimg = rgb2gray(subimg)
                subimg = np.expand_dims(subimg,2)
            else:
                raise Exception('incorect image type')
                
                
            if model.config.clahe:
            
                if subimg.shape[2]==1:
                    
                    
                    clahe = cv2.createCLAHE(clipLimit=model.config.clahe_clip,tileGridSize=(model.config.clahe_grid,model.config.clahe_grid))
                    subimg = clahe.apply(subimg[:,:,0])
                    
                    subimg = np.expand_dims(subimg,2)
                    
                else:
                    
                    
                    planes = cv2.split(subimg)
            
                    clahe = cv2.createCLAHE(clipLimit=model.config.clahe_clip,tileGridSize=(model.config.clahe_grid,model.config.clahe_grid))
                    
                    planes[0] = clahe.apply(planes[0])
                    planes[1] = clahe.apply(planes[1])
                    planes[2] = clahe.apply(planes[2])
                    
                    subimg = cv2.merge(planes)

                
                
            subimg = subimg.astype(np.float32)/255 - 0.5
            
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
        
        
        
    