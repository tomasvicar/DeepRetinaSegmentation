#for training version 3, ACDC, StT new

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
import pickle
import pydicom as dcm
from scipy.stats import norm

import Utilities as Util
import Loaders
import Network as Network
# import Network_v9 as Network


lr         = 0.001
L2         = 0.000001
batch      = 8
step_size  = 200
sigma      = 0.7
lambda_Train = 1.0
num_ite    = 50
num_epch   = 500


batchTr = int(np.round(batch))
step_size = int(np.round(step_size))
num_ite = int(np.round(num_ite))
 
torch.cuda.empty_cache()   
 
net = Network.Net(enc_chs=(4,32,64,128,256), dec_chs=(256,128,64,32), head=(128), num_class=3)
# net = Network.AttU_Net(img_ch=1,output_ch=1)
Network.init_weights(net,init_type= 'xavier', gain=0.02)


# version = "net_v0_0_1"
# net = torch.load("/home/chmelikj/Documents/chmelikj/Ophtalmo/DeepRetinaSegmentation/Ves_Class/Models/" + version + ".pt")


version_new = "v0_0_4"


net = net.cuda()
optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=L2)
# optimizer = optim.SGD(net2.parameters(), lr=0.000001, weight_decay=0.0001, momentum= 0.8)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1, verbose=False)



data_list_1_train=[];data_list_1_test=[];

# # # ## Data OPTHALMO
path_data = '/home/chmelikj/Documents/chmelikj/Ophtalmo/Data/data_preprocessed_dicom_12'  # Linux bioeng358
data_list = Loaders.CreateDataset_dcm_with_DC(os.path.normpath( path_data ), '','')

b = int(len(data_list)*0.80)
data_list_1_train = data_list_1_train + data_list[1:b+1]
data_list_1_test = data_list_1_test + data_list[b:]



diceTr_Clin=[]; diceTr_Other=[]; diceTr_Cons=[]; diceTe_Clin=[];

D1 = np.zeros((len(data_list_1_train),2))
D1[:,0] = np.arange(0,len(data_list_1_train))


for epch in range(0,num_epch):
    
    mu1, sigma1 = len(data_list_1_train)/10 , sigma*len(data_list_1_train)
    
    net.train(mode=True)

    diceTr1=[]; diceTe1=[];
               
    # for num_ite in range(0,len(data_list_1_train)-batch-1, batch):
    # for num_ite in range(0,len(data_list_1_train)/batch):
    for n_ite in range(0,num_ite):
      
        ## Pro StT our dataset CLINIC
        Indx_Sort = Util.rand_norm_distrb(batchTr, mu1, sigma1, [0,len(data_list_1_train)]).astype('int')
        Indx_Orig = D1[Indx_Sort,0].astype('int')
        sub_set = list(map(data_list_1_train.__getitem__, Indx_Orig))
        
        # params = (256,  186,276,  -170,170,  -40,40,-40,40,  0.9,1.2,  1.0)
        # params = (128,  108,148, -170,170,  -10,10,-10,10)
        params = (380,  256,400,  -20,20,  -10,10,-10,10,  1.0,1.0 )
        
        loss_train, res, _, Masks = Network.Training.straightForward(sub_set, net, params, TrainMode=True, Contrast=False)

                                           
        # dice = torch.sum(Util.ToOneHot(Masks, numClass=3 ).cuda()  (res>0.5) )
         
        # m = Util.ToOneHot(Masks, numClass=3 ).cuda()
        # correct = (torch.round(res[:,[1,2],:,:]) == m[:,[1:2],:,:]).sum().item()
           
        # diceTr1.append( np.mean( dice.detach().cpu().numpy() ) )
        
        metric  = loss_train.detach().cpu().item() 
        
        diceTr1.append( metric )
        
        D1[np.array(Indx_Sort),1] = np.array(10-metric)

        del Masks, res

        
        D1 = D1[D1[:, 1].argsort()]
        

    ## backF - training
        net.train(mode=True)
        if epch>0:
            
            loss = lambda_Train*loss_train
            
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=1.0)
            optimizer.step()
    
    
    
    # pd = norm(mu1,sigma1)
    # plt.figure()
    # plt.plot(D1[:,1])
    # y = pd.pdf([np.linspace(0,np.size(D1,0),np.size(D1,0))]).T
    # plt.plot(y/y.max())
    # plt.ylim([0.0, 1.1])
    # plt.show()

    
    if epch>0:
        scheduler.step()
        
    net.train(mode=False)
   
    ### validation
    params = (380,  380,380,  -0,0,  -0,0,-0,0,    1.0,1.0,   1.0)
    # params = (128,  108,148, -170,170,  -10,10,-10,10)
    batchTe = 20
    random.shuffle(data_list_1_test)
    # for num in range(0,len(data_list_1_test), batchTe):
    for num in range(0,1):   
        sub_set = data_list_1_test[num:num+batchTe]
        with torch.no_grad():
            
            loss_test, resTe, ImgsTe, MasksTE = Network.Training.straightForward(sub_set, net, params, TrainMode=False, Contrast=False)       
            # _, resTe, ImgsTe, MasksTE = Network.Training.straightForwardFour(sub_set, net, params, TrainMode=False, Contrast=False)       
                         
        
        # dice = Util.dice_coef( resTe[:,0,:,:]>0.5, MasksTE[:,0,:,:].cuda() )                
        
        diceTe1.append(loss_test.detach().cpu().item() )


    torch.cuda.empty_cache() 
     

    diceTr_Clin.append(np.mean(diceTr1))
    diceTe_Clin.append(np.mean(diceTe1))
    
    # resTe = np.round( resTe.detach().cpu().numpy() )
    resTe = resTe.detach().cpu().numpy()

    
    # plt.figure
    # # plt.imshow(ImgsTe[0,0,:,:].detach().numpy(), cmap='gray')
    # plt.imshow(resTe[0,1,:,:], cmap='jet', alpha=0.5)
    # plt.imshow(resTe[0,2,:,:], cmap='jet', alpha=0.5)
    # plt.show()
    
    plt.figure
    plt.imshow(resTe[0,0,:,:], cmap='jet')
    plt.show()
    plt.figure
    plt.imshow(resTe[0,1,:,:], cmap='jet')
    plt.show()   
    plt.figure
    plt.imshow(resTe[0,2,:,:], cmap='jet')
    plt.show()
    plt.figure
    plt.imshow(MasksTE[0,0,:,:], cmap='jet')
    plt.show()
    plt.figure
    plt.imshow(ImgsTe[0,3,:,:], cmap='jet')
    plt.show()
   
    
    plt.figure()
    plt.plot(diceTr_Clin,label='Joint Train')
    plt.plot(diceTe_Clin,label='Joint Test')

    plt.ylim([0.1, 0.9])
    plt.legend()
    plt.show()    
    

torch.save(net, 'Models/net_' + version_new + '.pt')


# file_name = "Models/Res_net_" + version + ".pkl"
# open_file = open(file_name, "wb")
# pickle.dump([diceTr_Clin, diceTe_Clin, diceTr_Other, diceTr_Cons, HD_Te_Clin], open_file)
# open_file.close()

