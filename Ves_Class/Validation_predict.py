## dataloader for new data2 - annotated

#for training Unet of ACDC - ED, ES
import os
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import torch
from torch.utils import data
import torch.optim as optim
import glob
import random
import torchvision.transforms as T
import pandas as pd
import cv2
import pickle
import pydicom as dcm
from skimage import measure, morphology
from scipy.io import savemat


import Utilities as Util
import Loaders
# import Network_v9 as Network
import Network as Network
#
data_list_test=[]

# data_list_2_train, data_list_3_train = Loaders.CreateDataset()

# ## StT LABELLED - JOINT
# path_data = '/data/rj21/Data/Data_1mm/Joint'  # Linux bioeng358
# data_list = Loaders.CreateDataset_StT_J_dcm(os.path.normpath( path_data ))
# data_list_test = data_list_test + data_list
# # data_list_test = data_list_test + data_list[1028:1154]

# # ## StT LABELLED - P1-30
# path_data = '/data/rj21/Data/Data_StT_Labaled'  # Linux bioeng358
# data_list = Loaders.CreateDataset_StT_P_dcm(os.path.normpath( path_data ),'','_m')
# data_list_test = data_list_test + data_list
# # b = int(len(data_list)*0.55)
# # data_list_test = data_list[b+1:]

# # ## StT LABELLED - Anastazia Valid
path_data = '/data/rj21/Data/Data_StT_Anast_valid'  # Linux bioeng358
data_list = Loaders.CreateDataset_StT_P_dcm(os.path.normpath( path_data ),'','')
data_list_test = data_list_test + data_list
# b = int(len(data_list)*0.55)
# data_list_test = data_list[b+1:]

##  StT LABELLED - Alina data T2
# # path_data = '/data/rj21/Data/Data_T2_Alina/dcm_resaved'  # Linux bioeng358
# path_data = '/data/rj21/Data/Data_1mm/T2_alina'
# data_list = Loaders.CreateDataset_StT_P_dcm(os.path.normpath( path_data ),'','')
# # data_list_test = data_list_test + data_list
# b = int(len(data_list)*0.80)
# data_list_test = data_list_test + data_list[b+1:]


# # ## CLinic UnLabled data
# # path_data = '/data/rj21/Data/Test_data/Clin_Unl'  # Linux bioeng358
# path_data = '/data/rj21/Data/Data_StT_Unlabeled'  # Linux bioeng358
# # data_list = Loaders.CreateDataset_StT_UnL_dcm(path_data, '', 'T1-postGd-map' )
# data_list = Loaders.CreateDataset_StT_UnL_dcm(path_data, 'P199', 'map' )
# data_list_test = data_list_test + data_list
# # b = int(len(data_list)*0.80)
# # data_list_test = data_list_test + data_list[b+1:]


# version = "v8_3_9_1_1"
version = "v8_3_9_4"
# version = "v9_2_1_0"
# version = "v9_1_6_6"
# version = "v9_2_0_0"

net = torch.load(r"/data/rj21/MyoSeg/Models/net_" + version + ".pt")
net = net.cuda()

path_save = '/data/rj21/MyoSeg/valid/Val_clin_mat/'
version = version + ""

# save_name = 'Joint_valid'
# save_name = 'All_valid'
# save_name = 'alinas_valid'
save_name = 'Anast_pp'

try:
    os.mkdir(path_save)
except OSError as error:
    print(error) 

res_table = pd.DataFrame(data=[], columns=['Name','ID_pat','Slice' ,'Dataset','Seq', 'Dice','HD', 'ID_image'] )                                         

diceTe=[]
vel=[]
iii=0
velImg = 256;
params = (velImg,  velImg,velImg,  0,0,  0,0,0,0,   1.0,1.0,  1.0) 

for num in range(0,len(data_list_test),1):
# ind=(0,50)
# for num in ind:
# for num in range(0,50,1):    

    Imgs = torch.tensor(np.zeros((1,1,velImg,velImg) ), dtype=torch.float32)
    Masks = torch.tensor(np.zeros((1,1,velImg,velImg) ), dtype=torch.float32)

    
    current_index = data_list_test[num]['slice']
    img_path = data_list_test[num]['img_path']
    mask_path = data_list_test[num]['mask_path']
    nPat = data_list_test[num]['pat_name']
    file_name = data_list_test[num]['file_name']
    seq = data_list_test[num]['Seq']

    dataset = dcm.dcmread(img_path)
    img = dataset.pixel_array
    dataset = dcm.dcmread(mask_path)
    mask = dataset.pixel_array
    mask = mask==1
    
    vel.append(img.shape)

    if len(dataset.dir('PixelSpacing'))>0:
        resO = (dataset['PixelSpacing'].value[0:2])
    else:
        resO = (2.0, 2.0)
        
    resN = (params[11],params[11])
    
    # img, p_cut, p_pad = Util.crop_center_final(img,velImg,velImg)
    # mask, p_cut, p_pad = Util.crop_center_final(mask,velImg,velImg)

    img = torch.tensor(np.expand_dims(img, 0).astype(np.float32))
    mask = torch.tensor(np.expand_dims(mask, 0).astype(np.float32))
    # img = torch.tensor(img.astype(np.float32))
    # mask = torch.tensor(mask.astype(np.float32))
    
    img = Util.Resampling(img, resO, resN, 'bilinear')
    mask = Util.Resampling(mask, resO,  resN, 'nearest')
    imgO = img;
    
    # rot = (12,45,70,95,120,150,187,201,230,275,0)
    rot = (45,95,120,170,187,230,250,275,0)
        
    Res = np.zeros((len(rot),velImg,velImg))
    k=0
    for r in rot:
        # if r>0:
        #     params = (velImg,  velImg,velImg,  r,r,  -20,20,-20,20,   1.0,1.0,  1.0) 
        # else:
        #     params = (velImg,  velImg,velImg,  r,r,  0,0,0,0,   1.0,1.0,  1.0) 
        # r=0
        
        # params = (velImg,  velImg,velImg,  r,r,  -20,20,-20,20,   1.0,1.0,  1.0) 
        params = (velImg,  velImg,velImg,  r,r,  -20,20,-20,20,   1.0,1.0,  1.0) 

        
        augm_params=[]
        augm_params.append({'Output_size': params[0],
                            'Crop_size': random.randint(params[1],params[2]),
                            'Angle': random.randint(params[3],params[4]),
                            'Transl': (0,0),
                            'Scale': random.uniform(params[9],params[10]),
                            'Flip': False
                            })
        
        augm_params[0]['Angle'] = r
        augm_params[0]['Transl'] = (0,0)
        img = Util.augmentation2(imgO, augm_params) 
        
        augm_params[0]['Angle'] = 0
        if not r==0:
            augm_params[0]['Transl'] = (random.randint(params[5],params[6]),random.randint(params[7],params[8]))
        else:
            augm_params[0]['Transl'] =  (0,0)
        img = Util.augmentation2(img, augm_params)
        # plt.figure()
        # plt.imshow(img[0,:,:])
        # plt.show()
        
        Imgs[0,0,:,:] = img
            
        # sc = 
        with torch.no_grad(): 
            res = net( Imgs.cuda() )
            # res = torch.softmax(res,dim=1)
            res = torch.sigmoid(res)      # for V9_   
        
        augm_params[0]['Angle'] = 0
        augm_params[0]['Transl'] = (-augm_params[0]['Transl'][0],-augm_params[0]['Transl'][1])
        res = Util.augmentation2(res, augm_params) 
        
        augm_params[0]['Angle'] = -r
        augm_params[0]['Transl'] = (0,0)
        res = Util.augmentation2(res, augm_params) 
        
        # res = (res[0,0,:,:].detach().cpu().numpy()>0.5)
        # # .astype(np.dtype('float')),0) 
        # labelled = measure.label(res)
        # rp = measure.regionprops(labelled)
        # if not not rp:
        #     size = max([i.area for i in rp])
        #     resM = morphology.remove_small_objects(res, min_size=size-1)
        # else:
        #     resM=res
        
        Res[k,:,:] = res[0,0,:,:].detach().cpu().numpy()
        k=k+1
        
        # plt.figure
        # plt.imshow( Res[k-1,:,:] )    
        # plt.show()
        
    torch.cuda.empty_cache()
    
    mask = Util.augmentation2(mask, augm_params)
    mask = mask[0,:,:]>0.5
     
    tresh = 0.50
    # res = np.median((Res>0.5).astype(np.dtype('float')),0) 
    # res = np.mean(Res,0) >0.5 
    res = np.median(Res,0) > tresh
    # res = np.sum(Res,0) >0.5  
    # res = Res[8,:,:] >0.5
    # res = Res[8,:,:]>tresh
    
    # res=res>0.5
    # save_name = 'Clin_pp' + '_' + str(tresh)
    # plt.figure
    # plt.imshow( res )    
    # plt.show()
    
    
    # res = morphology.binary_erosion(res)
    labelled = measure.label(res)
    rp = measure.regionprops(labelled)
    if not not rp:
        size = max([i.area for i in rp])
        resM = morphology.remove_small_objects(res, min_size=size-1)
    else:
        resM=res
        
    resM = torch.tensor(resM)
    res = torch.tensor(res)
    # resM = resM[0,:,:].cpu()
    # res = res[0,:,:,].cpu()
    
    # plt.figure()
    # plt.imshow(resM)
    # plt.show()
        
    dice = Util.dice_coef( res, mask )  
    diceM = Util.dice_coef( resM, mask )         
 
    A = resM.detach().cpu().numpy()>tresh
    B = mask.detach().cpu().numpy()>0.5
    HD = np.max((Util.MASD_compute(A,B),Util.MASD_compute(B,A)))
              
    A = resM.detach().cpu().numpy()>tresh
    B = mask.detach().cpu().numpy()>0.5
    HDm = np.max((Util.MASD_compute(A,B),Util.MASD_compute(B,A)))
    
    # resO = (resN)
    # resN = (2.5, 2.5)
    
    # resMin = Util.Resampling(torch.unsqueeze(res, 0) , resO, resN, 'bilinear')
    # maskmin = Util.Resampling(torch.unsqueeze(mask, 0), resO,  resN, 'nearest')
    
    # A = resMin[0,:,:].detach().cpu().numpy()>0.5
    # B = maskmin[0,:,:].detach().cpu().numpy()>0.5
    # HDmin = np.max((Util.MASD_compute(A,B),Util.MASD_compute(B,A)))
    
    dif = torch.sum(  (res ^ resM)  )
    # print(dif.item())
    
    res_table.loc[(num,'Name')] =   nPat 
    res_table.loc[(num,'ID_pat')] =   data_list_test[num]['ID_pat']
    res_table.loc[(num,'ID_scan')] = data_list_test[num]['ID_scan']
    res_table.loc[(num,'Slice')] =   current_index 
    # res_table.loc[(num,'Dataset')] =   file_name.split('_')[0]
    res_table.loc[(num,'Dataset')] =   file_name.split('_')[0]
    res_table.loc[(num,'Seq')] =   seq
    res_table.loc[(num,'HD')] =   HD
    res_table.loc[(num,'Dice')] =   dice.item()
    res_table.loc[(num,'HD2')] =   HDm
    # res_table.loc[(num,'HDmin')] =   HDmin
    res_table.loc[(num,'Dice2')] =   diceM.item()
    res_table.loc[(num,'FP')] =   dif.item()
    res_table.loc[(num,'ID_image')] =   iii

    
    print(nPat)
    
    # Fig = plt.figure()
    # plt.imshow(img[0,:,:], cmap='gray')
    
    # for ii in range(0,2):  
    #     if ii==0:
    #         res = Res[8,:,:]>tresh
    #     else:
    #         res = np.median(Res,0) > tresh  
    
    res = np.median(Res,0) > tresh
 
    # resB = res.astype(np.dtype('uint8'))
    # dimg = cv2.dilate(resB, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)) )
    # ctr = dimg - resB
    
    # GT = (mask.detach().cpu().numpy()>0.5).astype(np.dtype('uint8'))
    # dimg = cv2.dilate(GT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)) )
    # ctrGT = dimg - GT
    
    # imgB = img[0,:,:].numpy().astype(np.dtype('float'))
    # imgB = (( imgB - imgB.min() ) / (imgB.max() - imgB.min()) )
    # RGB_imgB = cv2.cvtColor((imgB*255).astype(np.dtype('uint8')),cv2.COLOR_GRAY2RGB)
    
    # comp = RGB_imgB[:,:,0]
    # comp[ctr==1]=255
    # comp[ctrGT==1]=0
    # RGB_imgB[:,:,0] = comp
    
    # comp = RGB_imgB[:,:,1]
    # comp[ctr==1]=0
    # comp[ctrGT==1]=255
    # RGB_imgB[:,:,1] = comp
    
    # comp = RGB_imgB[:,:,2]
    # comp[ctr==1]=0
    # comp[ctrGT==1]=0
    # RGB_imgB[:,:,2] = comp
    
    # Fig = plt.figure()
    # plt.imshow(RGB_imgB)
    # # plt.show()
    # # plt.draw()
        
    # Fig.savefig( path_save + '/' + 'res_' + '_' +  nPat + file_name.split('_')[0] + '_' + seq + '_' + str(current_index) + '.png', dpi=150)
    # # Fig.savefig( path_save + '/' + 'res_' + "%.4f" % (dice.item()) + '_' +  nPat + file_name.split('_')[0] + '_' + seq + '_' + str(current_index) + '.png', dpi=150)
    # plt.close(Fig)
    
    mdic = {"segm_mask": res, "dcm_data": img[0,:,:].numpy(), "prob_maps": Res}
    savemat(  path_save + '/' + version[0:2] + '_' +  nPat + file_name.split('_')[0] + '_' + seq + '_' + current_index + '.mat', mdic)

res_table.Dice.mean()    
Util.save_to_excel(res_table, path_save + '/' , 'Res_' + save_name + '_' + version)