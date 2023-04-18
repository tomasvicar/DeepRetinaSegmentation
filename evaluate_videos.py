import os
from skimage.io import imsave, imread
import numpy as np
import torch 
from scipy.signal import convolve2d 
from sklearn.metrics import roc_auc_score
import h5py
import matplotlib.pyplot as plt
from glob import glob
from skimage.transform import resize
from FFFMPEGvideo_read_write import load_video,save_video
from Unet import Unet
from config import Config


model_name = r"..\manual_models\norm_all_0_69_0.00001_gpu_5.23258_train_0.30543_valid_0.32891.pt"
save_text = 'segmentation_norm_all'
filenames = glob('../data_duf/*_preprocessed_norm.avi',recursive=True)


# model_name = r"..\manual_models\orig_all_0_58_0.00010_gpu_5.24120_train_0.30183_valid_0.32764.pt"
# save_text = 'segmentation_orig_all'
# filenames = glob('../data_duf/*_preprocesed.avi',recursive=True)



# model_name = r"..\manual_models\orig_hrf_0_74_0.00000_gpu_5.23258_train_0.18337_valid_0.15574.pt"
# model_name = r"..\manual_models\norm_hrf_0_45_0.00010_gpu_5.24120_train_0.18426_valid_0.15602.pt"








orig_size = (970, 1224)

device = torch.device('cuda:0')
# device = torch.device('cpu')

# config = Config()
# model = Unet(filters=config.filters,in_size=3,out_size=1,depth=config.depth)
# model.load_state_dict(torch.load(model_name))

model=torch.load(model_name)
model.eval()
model=model.to(device)

config = model.config
# model.config = config

for file_num, filename in enumerate(filenames):

    print(str(file_num) + '/' + str(len(filenames)))    
    
    
    data_video, fps = load_video(filename)
    
    output = np.zeros((data_video.shape[0],orig_size[0],orig_size[1],1),dtype=np.uint8)

    for img_num in range(data_video.shape[0]):

    
        
        
        # filename_save_example = control_resutls_path + os.sep + os.path.split(filename_save)[1]
        
        
        patch_size = model.config.patch_size  
        border = 40
        
        
        weigth_window=2*np.ones((patch_size,patch_size))
        weigth_window=convolve2d(weigth_window,np.ones((border,border))/np.sum(np.ones((border,border))),'same')
        weigth_window=weigth_window-1
        weigth_window[weigth_window<0.01]=0.01
        
        
    
        img0 = data_video[img_num,:,:,:]
        img0 = np.concatenate((img0, img0, img0),axis=2)
        img = img0.copy()
        
        
        img = img.astype(np.float64)/255 - 0.5
    
    
        
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
        
        
        final_resized = ((resize(final, orig_size) > 0.5) * 255).astype(np.uint8)
        
        output[img_num,:,:,0] = final_resized
        
        # imsave(filename_save,final_resized)
        
        img_cont = img0.copy()
        img_cont[:,:,0][final > 0.5] = 255
        img_cont[:,:,1][final > 0.5] = 0
        img_cont[:,:,2][final > 0.5] = 0
        control = np.concatenate((img0,img_cont),1)
        
        # plt.imshow(control)
        # plt.show()
    
        # dsfsdfdsfsd
        # imsave(filename_save_example, control)
    
    
    save_video(filename.replace('.avi','') + save_text + '.avi', output, fps)
    
    
    
    
    
   
    
    

