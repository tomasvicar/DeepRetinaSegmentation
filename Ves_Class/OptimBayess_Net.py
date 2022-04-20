
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import random
import pickle
from bayes_opt import BayesianOptimization

import Utilities as Util
import Network
import Loaders



def get_value(**params):
    lr         = params['lr']
    batch      = params['batch']
    step_size  = params['step_size']
    sigma      = params['sigma']
    # lamda_cons = params['lambda_cons']
    num_ite    = params['num_ite']
    
    lambda_Cons = 0.01
    lambda_Other = params['lambda_Other']
    num_epch = 80

    
    batchTr = int(np.round(batch))
    step_size = int(np.round(step_size))
    num_ite = int(np.round(num_ite))
     
    net = torch.load(r"/data/rj21/MyoSeg/Models/net_v7_0_0.pt")
    # net = torch.load(r"/data/rj21/MyoSeg/Models/net_v3_0_0.pt")
    
    net = net.cuda()
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.00000001)
    # optimizer = optim.SGD(net2.parameters(), lr=0.000001, weight_decay=0.0001, momentum= 0.8)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1, verbose=False)
    
    
    data_list_2_train, data_list_3_train = Loaders.CreateDataset()
    random.shuffle(data_list_2_train)
    
    ## StT LABELLED - P1-30
    path_data = '/data/rj21/Data/Data_StT_Labaled'  # Linux bioeng358
    data_list = Loaders.CreateDataset_StT_P_dcm(os.path.normpath( path_data ), 'A','')
    b = int(len(data_list)*0.70)
    data_list_1_train = data_list[1:b]
    data_list_1_test = data_list[b+1:-1]
    data_list = Loaders.CreateDataset_StT_P_dcm(os.path.normpath( path_data ), 'P','')
    b = int(len(data_list)*0.70)
    data_list_1_train = data_list_1_train + data_list[1:b]
    data_list_1_test = data_list_1_test +  data_list[b+1:-1]
    
    
    diceTr_Clin=[]; diceTr_Other=[]; diceTr_Cons=[]; diceTe_Clin=[]; HD_Te_Clin=[]
    
    # num_iter = 60
    # batchTr = 24
    D1 = np.zeros((len(data_list_1_train),2))
    D1[:,0] = np.arange(0,len(data_list_1_train))
    D2 = np.zeros((len(data_list_2_train),2))
    D2[:,0] = np.arange(0,len(data_list_2_train))
       
    
    for epch in range(0,num_epch):
        mu1, sigma1 = len(data_list_1_train)/10 , sigma*len(data_list_1_train)
        mu2, sigma2 = len(data_list_2_train)/10 ,  sigma*len(data_list_2_train)
        
        net.train(mode=True)
        
        # if epch>10:
        #     sigma = 0.7
    
        diceTr1=[]; diceTr2=[]; diceTr3=[]; diceTe1=[]; diceTe2=[];  HD1=[]
        Inds1=[]; Inds2=[]; Inds4=[]; Inds5=[];   Inds6=[]
                   
        # for num_ite in range(0,len(data_list_1_train)-batch-1, batch):
        # for num_ite in range(0,len(data_list_1_train)/batch):
        for n_ite in range(0,num_ite):
          
            ## Pro StT our dataset CLINIC
            Indx_Sort = Util.rand_norm_distrb(batchTr, mu1, sigma1, [0,len(data_list_1_train)]).astype('int')
            Indx_Orig = D1[Indx_Sort,0].astype('int')
            sub_set = list(map(data_list_1_train.__getitem__, Indx_Orig))
            
            params = (128,  80,120,  -170,170,  -10,10,-10,10)
            loss_Clin, res, _, Masks = Network.Training.straightForward(sub_set, net, params, TrainMode=True, Contrast=False)
                                                       
            dice = Util.dice_coef_batch( res[:,0,:,:]>0.5, Masks[:,0,:,:].cuda() )                
            diceTr1.append( np.mean( dice.detach().cpu().numpy() ) )
            # Inds1.append(Indx)
            D1[np.array(Indx_Sort),1] = np.array(dice.detach().cpu().numpy())
            # D1 = D1[D1[:, 0].argsort()]
            
            # ## Pro Other dataset
            Indx_Sort = Util.rand_norm_distrb(batchTr, mu2, sigma2, [0,len(data_list_2_train)]).astype('int')
            Indx_Orig = D2[Indx_Sort,0].astype('int')
            sub_set = list(map(data_list_2_train.__getitem__, Indx_Orig))  
            
            loss_Other, res, _, Masks = Network.Training.straightForward(sub_set, net, params, TrainMode=True, Contrast=False)
                                                           
            dice = Util.dice_coef_batch( res[:,0,:,:]>0.5, Masks[:,0,:,:].cuda() )                
            diceTr2.append(dice.detach().cpu().numpy())
            D2[np.array(Indx_Sort),1] = np.array(dice.detach().cpu().numpy())
    
        
            D1 = D1[D1[:, 1].argsort()]
            D2 = D2[D2[:, 1].argsort()]
        
        ## Consistency regularization
            # params = (128,  80,120,  -170,170,  -10,10,-10,10)
            # batchCons = 16
            # Indx = np.random.randint(0,len(data_list_3_train),(batchCons,)).tolist()
            # sub_set = list(map(data_list_3_train.__getitem__, Indx))
            # loss_cons, Imgs_P, res, res_P = Network.Training.Consistency(sub_set, net, params, TrainMode=True, Contrast=False)
            # diceTr3.append(1 - loss_cons.detach().cpu().numpy())
            
        ## backF - training
            net.train(mode=True)
            if epch>0:
                loss = loss_Clin + lambda_Other*loss_Other
                # loss = loss_Clin + lambda_Other*loss_Other + lambda_Cons*loss_cons
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
    ### StT lab
    params = (128,  80,120,  -0,0,  0,0,0,0)
    batch = 256
    random.shuffle(data_list_1_test)
    # for num in range(0,len(data_list_4_test), batch):
    for num in range(0,2):   
        sub_set = data_list_1_test[num:num+batch]
        with torch.no_grad():
            _, resTE, ImgsTe, MasksTE = Network.Training.straightForward(sub_set, net, params, TrainMode=False, Contrast=False)       
                         
        dice = Util.dice_coef( resTE[:,0,:,:]>0.5, MasksTE[:,0,:,:].cuda() )                
        diceTe1.append(dice.detach().cpu().numpy())
         
        # for b in range(0,batch):
        #     A = resTE[b,0,:,:].detach().cpu().numpy()>0.5
        #     B = MasksTE[b,0,:,:].detach().cpu().numpy()>0.5
        #     HD1.append (np.max((Util.MASD_compute(A,B),Util.MASD_compute(B,A))))
    
    torch.cuda.empty_cache()  
    
    return np.mean(diceTe1)
        



# ---------------- Optimaliyace -----------------


# param_names=['lr']
# bounds_lw=[0.00001]
# bounds_up=[0.0001]
# pbounds=dict(zip(param_names, zip(bounds_lw,bounds_up)))  

    
pbounds = {'lr':[0.00001,0.005],
           'batch':[8,64],
           'sigma':[0.5,1.5],
           'step_size':[15,40],
            'num_ite' :[20, 80],
           'lambda_Other' :[0.2,1.0]
           # 'lambda_cons':[0.001,0.01]
           }  

optimizer = BayesianOptimization(f = get_value, pbounds=pbounds,random_state=1)  

optimizer.maximize(init_points=5,n_iter=20)

print(optimizer.max)

params=optimizer.max['params']
# print(params)

file_name = "Models/BO_Unet_v8_0.pkl"
# open_file = open(file_name, "wb")
# pickle.dump(optimizer, open_file)
# pickle.dump(params, open_file)
# open_file.close()

open_file = open(file_name, "wb")
pickle.dump(params, open_file)
open_file.close()

# open_file = open(file_name, "rb")
# data_list_test = pickle.load(open_file)
# open_file.close()